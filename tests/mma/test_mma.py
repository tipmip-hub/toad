"""Tests for MMA (multi-model aggregation) pipeline."""

import tempfile
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

from toad import MMA, TOAD
from toad.regridding import HealPixRegridder
from toad.shifts import ASDETECT


def setup_and_export_native(tmp_path: Path, n_runs: int = 3) -> list[str]:
    """Create clusterings, export as native format via export_for_mma, return paths."""
    td = TOAD("tutorials/test_data/synth_data.nc", time_dim="time")
    td.data = td.data.coarsen(lat=3, lon=3, time=3, boundary="trim").reduce(np.mean)

    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        td.compute_shifts(td.base_vars[0], method=ASDETECT(ignore_nan_warnings=True))

    paths = []
    for i in range(n_runs):
        p = tmp_path / f"native_{i}.nc"
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=5),
            time_weight=0.5 + i * 0.3,
            shift_threshold=0.8,
            export_for_mma=str(p),
            mma_grid="native",
        )
        paths.append(str(p))
    return paths


def setup_and_export_healpix(
    tmp_path: Path, n_runs: int = 3, nside: int = 16
) -> list[str]:
    """Create clusterings, export as HealPix format via export_for_mma, return paths."""
    td = TOAD("tutorials/test_data/synth_data.nc", time_dim="time")
    td.data = td.data.coarsen(lat=4, lon=4, time=3, boundary="trim").reduce(np.mean)

    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        td.compute_shifts(td.base_vars[0], method=ASDETECT(ignore_nan_warnings=True))

    paths = []
    for i in range(n_runs):
        p = tmp_path / f"healpix_{i}.nc"
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=5),
            time_weight=0.5 + i * 0.25,
            shift_threshold=0.8,
            regridder=HealPixRegridder(nside=nside),
            export_for_mma=str(p),
            mma_grid="healpix",
        )
        paths.append(str(p))
    return paths


def test_export_for_mma_native():
    """Test compute_clusters export_for_mma with mma_grid='native'."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_native(Path(tmp), n_runs=2)
        assert len(paths) == 2
        for p in paths:
            ds = __import__("xarray").open_dataset(p)
            assert "cluster" in ds
            assert "lat" in ds.coords or "latitude" in ds.coords
            ds.close()


def test_export_for_mma_healpix():
    """Test compute_clusters export_for_mma with mma_grid='healpix'."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=2, nside=8)
        assert len(paths) == 2
        for p in paths:
            ds = __import__("xarray").open_dataset(p)
            assert "cluster" in ds
            assert "hp_pixel" in ds["cluster"].dims
            assert ds.attrs.get("nside") == 8
            npix = 12 * 8**2
            assert ds["cluster"].shape == (ds.sizes["time"], npix)
            ds.close()


def test_export_for_mma_healpix_no_events():
    """HealPix MMA export must succeed when no points pass the shift threshold."""
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmp:
        td = TOAD("tutorials/test_data/synth_data.nc", time_dim="time")
        td.data = td.data.coarsen(lat=4, lon=4, time=3, boundary="trim").reduce(np.mean)
        td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
        if len(td.shift_vars) == 0:
            td.compute_shifts(
                td.base_vars[0], method=ASDETECT(ignore_nan_warnings=True)
            )

        path = Path(tmp) / "empty_healpix.nc"
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=5),
            shift_threshold=1e9,
            regridder=HealPixRegridder(nside=8),
            export_for_mma=str(path),
            mma_grid="healpix",
        )
        assert path.exists()
        with xr.open_dataset(path) as ds:
            assert "cluster" in ds
            assert ds.attrs.get("nside") == 8
            assert np.all(np.isnan(ds["cluster"].values))


def test_mma_native_files():
    """Test MMA with native-format cluster label files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_native(Path(tmp), n_runs=3)
        mma = MMA(paths, nside=None)
        rate = mma.cluster_occurrence_rate()
        assert 0 <= float(rate.min()) <= float(rate.max()) <= 1
        ds = mma.run_consensus(
            min_consensus=0.5,
            temporal_tolerance=0,
            spatial_tolerance=1,
            min_cluster_area=2,
        )
        assert "consensus_clusters" in ds
        assert "consensus_clusters_rate" in ds
        assert "time" in ds["consensus_clusters"].dims
        assert (
            "lat" in ds["consensus_clusters"].dims
            or "lon" in ds["consensus_clusters"].dims
        )
        labels = ds["consensus_clusters"].values
        assert np.any(np.isfinite(labels) & (labels >= 0)) or np.any(labels == -1)
        # Plot native format (synth_data has lat/lon)
        fig, ax = mma.plot_consensus_clusters(
            map_style={"projection": "plate_carree"},
        )
        assert fig is not None
        assert ax is not None


def test_mma_native_files_rejected_with_nside():
    """Test MMA raises when nside is passed with native-format exports."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_native(Path(tmp), n_runs=2)
        with __import__("pytest").raises(ValueError) as exc_info:
            MMA(paths, nside=16)
        assert "native" in str(exc_info.value).lower()
        assert "nside=None" in str(exc_info.value)


def test_mma_healpix_files():
    """Test MMA with HealPix-format cluster label files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=3, nside=8)
        mma = MMA(paths, nside=8)
        ds = mma.run_consensus(
            min_consensus=0.4,
            temporal_tolerance=0,
            spatial_tolerance=1,
            min_cluster_area=1,
        )
        assert "consensus_clusters" in ds
        assert "consensus_clusters_rate" in ds
        assert ds["consensus_clusters"].dims == ("time", "hp_pixel")
        npix = 12 * 8**2
        assert ds["consensus_clusters"].shape == (ds.sizes["time"], npix)
        assert mma.consensus_cluster_ids() == sorted(
            mma.get_shift_times_per_consensus_cluster().keys()
        )
        # Shift times from HealPix cluster (no native grid in export)
        times_by_cluster = mma.get_shift_times_per_consensus_cluster()
        assert isinstance(times_by_cluster, dict)
        rate = mma.cluster_occurrence_rate()
        assert rate.shape == (npix,)
        assert 0 <= rate.min() <= rate.max() <= 1
        summary = mma.get_consensus_summary()
        assert "cluster_id" in summary.columns
        assert "size" in summary.columns
        assert "mean_consensus_rate" in summary.columns
        assert "mean_mean_shift_time" in summary.columns
        assert "std_mean_shift_time" in summary.columns
        fig2, ax_map2, ax_right2 = mma.plot.consensus_overview(
            cluster_ids=range(5),
            kind="violins",
            map_style={"projection": "plate_carree"},
        )
        assert fig2 is not None
        assert ax_map2 is not None
        assert ax_right2 is not None


def test_mma_healpix_union_time_alignment():
    """MMA aligns exports with different time lengths via time_alignment='union'."""
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=2, nside=8)
        short_path = Path(tmp) / "healpix_short.nc"
        long_path = Path(paths[0])
        ds = xr.open_dataset(long_path)
        ds_short = ds.isel(time=slice(0, ds.sizes["time"] // 2))
        ds_short.to_netcdf(short_path)
        ds.close()

        mma = MMA([str(short_path), paths[1]], nside=8, time_alignment="union")
        assert mma._label_arrays[0].shape[0] == mma._label_arrays[1].shape[0]
        ds = mma.run_consensus(
            min_consensus=0.5,
            temporal_tolerance=0,
            spatial_tolerance=1,
            min_cluster_area=1,
        )
        assert ds.sizes["time"] == max(
            xr.open_dataset(short_path).sizes["time"],
            xr.open_dataset(paths[1]).sizes["time"],
        )


def test_mma_infinite_spatial_tolerance_roundtrip():
    """``spatial_tolerance=np.inf`` survives NetCDF round-trip and dilation."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=2, nside=8)
        mma = MMA(paths, nside=8)
        mma.run_consensus(
            min_consensus=0.5,
            temporal_tolerance=1,
            spatial_tolerance=np.inf,
            min_cluster_area=1,
        )
        consensus_path = Path(tmp) / "consensus_inf.nc"
        mma.data.to_netcdf(consensus_path)

        reloaded = MMA.from_consensus(str(consensus_path), load_exports=False)
        temporal, spatial = reloaded._consensus_tolerances()
        assert temporal == 1
        assert np.isinf(spatial)

        T = reloaded.data.sizes[reloaded._time_dim]
        npix = 12 * 8**2
        mask = np.zeros((T, npix), dtype=bool)
        mask[0, 0] = True
        dilated = reloaded._dilate_healpix_model_support(mask)
        assert dilated[0].all()


def test_mma_from_consensus():
    """Reload saved consensus netCDF via MMA.from_consensus."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=2, nside=8)
        mma = MMA(paths, nside=8)
        mma.run_consensus(
            min_consensus=0.5,
            temporal_tolerance=0,
            spatial_tolerance=1,
            min_cluster_area=1,
        )
        consensus_path = Path(tmp) / "consensus.nc"
        mma.data.to_netcdf(consensus_path)

        reloaded = MMA.from_consensus(str(consensus_path))
        assert "consensus_clusters" in reloaded.data
        assert reloaded.nside == 8
        assert reloaded._format == "healpix"
        assert len(reloaded.paths) == 2
        rate = reloaded.cluster_occurrence_rate()
        assert rate.shape == (12 * 8**2,)
        fig, ax = reloaded.plot_consensus_clusters(
            map_style={"projection": "plate_carree"},
        )
        assert fig is not None
        assert ax is not None
