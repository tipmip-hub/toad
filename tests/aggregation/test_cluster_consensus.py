import gc

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.shifts import ASDETECT
from toad.utils import _attrs


def _latest_consensus_labels(td: TOAD):
    """Consensus label DataArray from the most recent :meth:`TOAD.compute_consensus` run."""
    names = td.consensus_cluster_vars
    assert names, "expected at least one consensus_cluster variable on td"
    return td.data[names[-1]]


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test. Important otherwise get bus errors on some machines."""
    yield
    gc.collect()


def setup_irregular_grid():
    """Setup and coarsen irregular grid data."""
    td = TOAD("tutorials/test_data/sea_ice_irregular_grid.nc", time_dim="time")
    td.data = td.data.isel(
        i=slice(None, None, 2),
        j=slice(None, None, 2),
        time=slice(None, None, 2),
    )
    return td


def setup_native_grid():
    """Setup and coarsen native grid data."""
    td = TOAD("tutorials/test_data/garbe_2020_antarctica.nc", time_dim="GMST")
    td.data = td.data.coarsen(x=4, y=4, GMST=3, boundary="trim").reduce(np.mean)
    return td


def setup_regular_latlon_grid():
    """Setup and coarsen regular lat/lon grid data."""
    td = TOAD("tutorials/test_data/synth_data.nc", time_dim="time")
    td.data = td.data.coarsen(lat=3, lon=3, time=3, boundary="trim").reduce(np.mean)
    return td


def setup_synthetic_consensus_toad(
    cluster_fields: dict[str, np.ndarray],
    *,
    base_values: np.ndarray | None = None,
    add_latlon: bool = False,
    cluster_id_attrs: dict[str, np.ndarray] | None = None,
) -> TOAD:
    """Build a tiny deterministic TOAD object with manual cluster fields."""
    first = next(iter(cluster_fields.values()))
    T, y_len, x_len = first.shape
    coords = {
        "time": np.arange(T, dtype=np.float32),
        "y": np.arange(y_len, dtype=np.int32),
        "x": np.arange(x_len, dtype=np.int32),
    }
    if base_values is None:
        base_values = np.broadcast_to(
            (np.arange(T, dtype=np.float32) + 1.0).reshape(T, 1, 1),
            (T, y_len, x_len),
        ).copy()

    ds = xr.Dataset(
        {
            "foo": xr.DataArray(
                np.asarray(base_values, dtype=np.float32),
                coords=coords,
                dims=("time", "y", "x"),
            )
        }
    )

    if add_latlon:
        lat = np.broadcast_to(
            np.linspace(-20.0, 20.0, y_len, dtype=np.float32).reshape(y_len, 1),
            (y_len, x_len),
        )
        lon = np.broadcast_to(
            np.linspace(0.0, 30.0, x_len, dtype=np.float32).reshape(1, x_len),
            (y_len, x_len),
        )
        ds = ds.assign_coords(
            {
                "lat": (("y", "x"), lat),
                "lon": (("y", "x"), lon),
            }
        )

    for name, labels in cluster_fields.items():
        arr = np.asarray(labels, dtype=np.float32)
        cluster_ids = (
            cluster_id_attrs[name]
            if cluster_id_attrs is not None and name in cluster_id_attrs
            else np.unique(arr[np.isfinite(arr)].astype(np.int32))
        )
        da = xr.DataArray(arr, coords=coords, dims=("time", "y", "x"), name=name)
        da.attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.asarray(cluster_ids, dtype=np.int32),
            }
        )
        ds[name] = da

    return TOAD(ds, time_dim="time")


def _build_td_with_consensus(*, min_consensus: float = 0.5) -> TOAD:
    """Native-grid TOAD with shifts, four clusterings, and one consensus run."""
    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    for tsf in (0.25, 0.5, 1.0, 1.5):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )
    td.compute_consensus(
        min_consensus=min_consensus,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
    )
    return td


@pytest.fixture(scope="module")
def td_with_consensus():
    return _build_td_with_consensus()


def non_noise_cluster_ids(da_clusters: xr.DataArray) -> np.ndarray:
    """Return sorted non-noise cluster ids present in a consensus field."""
    values = np.asarray(da_clusters.values)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.array([], dtype=np.int32)
    ids = np.unique(valid.astype(np.int32))
    return ids[ids >= 0]


def test_min_cluster_area_filter_removes_small_clusters():
    """Post-hoc consensus filter drops clusters with small spatial footprint."""
    from toad.postprocessing.aggregation import _filter_consensus_labels_min_size

    coords = {"time": [0, 1], "y": [0, 1], "x": [0, 1, 2]}
    lab = np.full((2, 2, 3), -1, dtype=np.int64)
    lab[:, 0, 0] = 0  # cluster 0: one spatial cell, two timesteps (area 1)
    lab[:, 1, :] = 1  # cluster 1: three spatial cells × two timesteps (area 3)
    cons = np.ones_like(lab, dtype=np.float32) * 0.9
    da_lab = xr.DataArray(lab, coords=coords, dims=("time", "y", "x"))
    da_cons = xr.DataArray(cons, coords=coords, dims=("time", "y", "x"))
    out_lab, out_cons = _filter_consensus_labels_min_size(
        da_lab, da_cons, min_cluster_area=3, time_dim="time"
    )
    assert np.sum(out_lab.values == 0) == 6
    assert np.sum(out_lab.values == -1) == 6
    np.testing.assert_allclose(out_cons.values, cons)


def test_rate_independent_of_min_consensus_with_min_cluster_area():
    """Member-support rate must not change when only min_consensus differs."""
    fields = {}
    for i in range(5):
        fields[f"foo_r{i + 1}_cluster"] = np.full((4, 2, 4), -1, dtype=np.float32)

    # Three members agree on a compact blob; two on a larger blob elsewhere.
    fields["foo_r1_cluster"][1, 0, 0:2] = 0
    fields["foo_r2_cluster"][1, 0, 0:2] = 0
    fields["foo_r3_cluster"][1, 0, 0:2] = 0
    fields["foo_r4_cluster"][2, 1, 2:4] = 0
    fields["foo_r5_cluster"][2, 1, 2:4] = 0

    def run(min_consensus: float):
        td = setup_synthetic_consensus_toad(fields)
        td.compute_consensus(
            min_consensus=min_consensus,
            temporal_tolerance=0,
            spatial_tolerance=0,
            min_cluster_area=2,
            show_progress=False,
        )
        return td.data["cluster_consensus_rate"].values

    rate_low = run(0.6)
    rate_high = run(0.8)
    np.testing.assert_allclose(rate_low, rate_high, equal_nan=True)


def test_member_support_retains_voxels_with_enough_dilated_votes():
    """Native voxels with enough dilated member support are kept and labelled."""
    fields = {}
    for i in range(4):
        fields[f"foo_r{i + 1}_cluster"] = np.full((4, 1, 3), -1, dtype=np.float32)

    fields["foo_r1_cluster"][1, 0, 0:2] = 0
    fields["foo_r2_cluster"][2, 0, 0:2] = 0
    fields["foo_r3_cluster"][1, 0, 1:3] = 0

    td = setup_synthetic_consensus_toad(fields)
    td.compute_consensus(
        min_consensus=0.75,
        temporal_tolerance=1,
        spatial_tolerance=0,
        min_cluster_area=None,
        show_progress=False,
    )

    da = td.data["cluster_consensus"]
    assert da.attrs["consensus_method"] == "member_support"
    assert da.attrs["min_consensus_members"] == 3
    assert non_noise_cluster_ids(da).tolist() == [0]

    expected = np.full((4, 1, 3), -1, dtype=np.float64)
    expected[1, 0, 1] = 0
    expected[2, 0, 1] = 0
    np.testing.assert_array_equal(da.values, expected)

    rate = td.data["cluster_consensus_rate"]
    assert np.allclose(rate.values[da.values == 0], 0.75)


def test_member_support_connects_voxels_across_temporal_gap():
    """Retained voxels separated by a tolerated time gap share one consensus id."""
    fields = {}
    for i in range(2):
        fields[f"foo_r{i + 1}_cluster"] = np.full((4, 1, 1), -1, dtype=np.float32)

    fields["foo_r1_cluster"][0, 0, 0] = 0
    fields["foo_r2_cluster"][2, 0, 0] = 0

    td = setup_synthetic_consensus_toad(fields)
    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=2,
        spatial_tolerance=0,
        min_cluster_area=None,
        show_progress=False,
    )

    da = td.data["cluster_consensus"]
    assert non_noise_cluster_ids(da).tolist() == [0]
    expected = np.full((4, 1, 1), -1, dtype=np.float64)
    expected[0, 0, 0] = 0
    expected[2, 0, 0] = 0
    np.testing.assert_array_equal(da.values, expected)


def test_member_support_discards_voxels_below_vote_threshold():
    """Native event voxels below the distinct-member threshold are discarded."""
    fields = {}
    for i in range(4):
        fields[f"foo_r{i + 1}_cluster"] = np.full((4, 1, 3), -1, dtype=np.float32)

    fields["foo_r1_cluster"][1, 0, 0:2] = 0
    fields["foo_r2_cluster"][2, 0, 0:2] = 0
    fields["foo_r3_cluster"][1, 0, 2] = 0

    td = setup_synthetic_consensus_toad(fields)
    td.compute_consensus(
        min_consensus=0.75,
        temporal_tolerance=1,
        spatial_tolerance=0,
        min_cluster_area=None,
        show_progress=False,
    )

    da = td.data["cluster_consensus"]
    assert non_noise_cluster_ids(da).size == 0
    assert np.all(da.values[np.isfinite(da.values)] == -1)


def test_compute_consensus_end_to_end_on_irregular_grid():
    """End-to-end consensus on a curvilinear grid with summary rate checks."""
    td = setup_irregular_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")

    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))

    for tsf in (0.5, 1.0, 1.5, 2.0):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )

    td.compute_consensus(
        min_consensus=0.8,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
    )
    cv = td.consensus_cluster_vars[-1]
    summary_df = td.aggregate.consensus_summary(cv)
    clusters = td.data[cv]
    rate = td.data[f"{cv}_rate"]

    assert td.time_dim in clusters.dims
    assert clusters.shape == rate.shape
    assert "pooled_median_shift_time" in summary_df.columns

    cvals = np.asarray(clusters.values, dtype=np.float64)
    assert np.all(np.isfinite(cvals) | (cvals == -1) | np.isnan(cvals))

    unique_clusters = non_noise_cluster_ids(clusters)
    if unique_clusters.size:
        assert len(summary_df) == len(unique_clusters)
        assert set(summary_df["cluster_id"].values) == set(unique_clusters)
        assert np.all(np.isfinite(summary_df["median_median_shift_time"].values))
    else:
        assert summary_df.empty


def test_shift_time_distribution_matches_summary_median(td_with_consensus):
    """Spatial median from shift-time distribution matches the summary table."""
    td = td_with_consensus
    dc = _latest_consensus_labels(td)
    summary_df = td.aggregate.consensus_summary(str(dc.name))
    dist_ds, df_cell = td.aggregate.consensus_shift_time_distribution(
        dc,
    )
    assert "spatial_median_transition_time" in dist_ds.data_vars
    assert dist_ds.sizes["cluster_var"] == len(td.cluster_vars)
    assert len(summary_df) > 0
    assert len(df_cell) > 0
    cid = int(summary_df["cluster_id"].iloc[0])
    sm = dist_ds["spatial_median_transition_time"].sel(consensus_cluster_id=float(cid))
    tab = float(
        summary_df.loc[
            summary_df["cluster_id"] == cid, "median_median_shift_time"
        ].iloc[0]
    )
    np.testing.assert_allclose(float(np.nanmedian(sm.values)), tab, rtol=1e-4)


def test_consensus_cluster_timeseries_matches_manual_mean(td_with_consensus):
    """Consensus-cluster timeseries matches a manually masked spatial mean."""
    from toad.utils.cluster_consensus_utils import _consensus_input_support_mask

    td = td_with_consensus
    dc = _latest_consensus_labels(td)
    summary_df = td.aggregate.consensus_summary(str(dc.name))
    assert len(summary_df) > 0
    cid = int(summary_df["cluster_id"].iloc[0])

    out = td.aggregate.consensus_cluster_timeseries(dc, cid, aggregation="mean")
    first_cluster_var = next(
        cvar
        for cvar in dc.attrs["cluster_vars"]
        if bool(
            (
                (dc == cid)
                & _consensus_input_support_mask(
                    td,
                    dc,
                    cvar,
                    spatial_dims=tuple(td.space_dims),
                    time_dim=td.time_dim,
                )
            )
            .any()
            .item()
        )
    )
    base_var = td.data[first_cluster_var].attrs[_attrs.BASE_VARIABLE]
    support_mask = _consensus_input_support_mask(
        td,
        dc,
        first_cluster_var,
        spatial_dims=tuple(td.space_dims),
        time_dim=td.time_dim,
    )
    footprint = ((dc == cid) & support_mask).any(dim=td.time_dim)
    expected = td._aggregate_spatial(td.data[base_var].where(footprint), method="mean")
    np.testing.assert_allclose(
        out[first_cluster_var].values,
        expected.values,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_native_edges_from_mask_stitches_meridian():
    from toad.utils.cluster_consensus_utils import _native_edges_from_mask

    mask = np.ones((2, 3), dtype=bool)
    flat_idx = np.arange(6, dtype=np.int64).reshape(2, 3)

    rows, cols = _native_edges_from_mask(mask, flat_idx, stitch_longitude=True)
    edges = set(zip(rows, cols))

    assert (0, 2) in edges
    assert (0, 5) in edges
    assert (2, 3) in edges


def test_stitch_meridian_auto_detection_and_resolution():
    from toad.utils.cluster_consensus_utils import (
        infer_stitch_meridian,
        resolve_stitch_meridian,
    )

    td_global = setup_regular_latlon_grid()
    spatial_dims = tuple(td_global.space_dims)
    assert infer_stitch_meridian(td_global.data, spatial_dims) is True
    assert resolve_stitch_meridian(
        True, dataset=td_global.data, spatial_dims=spatial_dims
    )
    assert not resolve_stitch_meridian(
        False, dataset=td_global.data, spatial_dims=spatial_dims
    )
    assert resolve_stitch_meridian(
        "auto", dataset=td_global.data, spatial_dims=spatial_dims
    )

    td_regional = setup_synthetic_consensus_toad(
        {"c1": np.full((1, 2, 4), -1, dtype=np.float32)},
        add_latlon=True,
    )
    assert (
        infer_stitch_meridian(td_regional.data, tuple(td_regional.space_dims)) is False
    )

    td_native = setup_native_grid()
    assert infer_stitch_meridian(td_native.data, tuple(td_native.space_dims)) is False


def test_stitch_meridian_merges_seam_split_clusters():
    """Explicit meridian stitching merges wraparound clusters split at the seam."""
    labels = np.array([[[0, -1, -1, 0], [0, -1, -1, 0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels, "c2": labels})

    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        stitch_meridian=False,
        show_progress=False,
        output_label="cons_no",
    )
    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        stitch_meridian=True,
        show_progress=False,
        output_label="cons_yes",
    )

    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_no"]), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_yes"]), np.array([0])
    )
    assert td.data["cons_no"].attrs["stitch_meridian"] == 0
    assert td.data["cons_no"].attrs["stitch_meridian_applied"] == 0
    assert td.data["cons_yes"].attrs["stitch_meridian"] == 1
    assert td.data["cons_yes"].attrs["stitch_meridian_applied"] == 1


def test_spatial_tolerance_merges_clusters_separated_by_two_cell_gap():
    """Raised spatial tolerance bridges native events two grid cells apart."""
    labels_a = np.array([[[0, -1, -1]]], dtype=np.float32)
    labels_b = np.array([[[-1, -1, 0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels_a, "c2": labels_b})

    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=1,
        min_cluster_area=None,
        show_progress=False,
        output_label="cons_split",
    )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=2,
        min_cluster_area=None,
        show_progress=False,
        output_label="cons_merged",
    )

    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_split"]), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_merged"]), np.array([0])
    )


def test_temporal_tolerance_merges_clusters_separated_by_two_step_gap():
    """Raised temporal tolerance bridges native events two timesteps apart."""
    labels_a = np.array([[[0]], [[-1]], [[-1]], [[-1]]], dtype=np.float32)
    labels_b = np.array([[[-1]], [[-1]], [[0]], [[-1]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels_a, "c2": labels_b})

    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=0,
        min_cluster_area=None,
        show_progress=False,
        output_label="cons_split",
    )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=2,
        spatial_tolerance=0,
        min_cluster_area=None,
        show_progress=False,
        output_label="cons_merged",
    )

    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_split"]), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_merged"]), np.array([0])
    )


def test_all_noise_inputs_yield_no_consensus_clusters():
    labels = np.full((2, 1, 2), -1, dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels, "c2": labels})

    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
    )
    cv = td.consensus_cluster_vars[-1]
    summary_df = td.aggregate.consensus_summary(cv)

    assert np.all(td.data[cv].values == -1)
    assert np.all(td.data[f"{cv}_rate"].values == 0)
    assert summary_df.empty


def test_high_min_consensus_yields_all_noise():
    """Strict threshold leaves all native events as noise but keeps partial rate values."""
    labels_a = np.array([[[0, 0, -1, -1]]], dtype=np.float32)
    labels_b = np.array([[[-1, -1, 0, 0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels_a, "c2": labels_b})

    td.compute_consensus(
        min_consensus=0.75,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
    )
    cv = td.consensus_cluster_vars[-1]
    rate = td.data[f"{cv}_rate"]
    summary_df = td.aggregate.consensus_summary(cv)

    assert np.all(td.data[cv].values == -1)
    assert summary_df.empty
    np.testing.assert_allclose(rate.values[0, 0, 0:2], 0.5)
    np.testing.assert_allclose(rate.values[0, 0, 2:4], 0.5)


def test_compute_consensus_rejects_invalid_parameters():
    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=10),
        time_weight=0.5,
        shift_threshold=0.8,
    )
    with pytest.raises(ValueError, match="temporal_tolerance"):
        td.compute_consensus(
            min_consensus=0.5,
            temporal_tolerance=-1,
            spatial_tolerance=0,
            show_progress=False,
        )
    with pytest.raises(ValueError, match="spatial_tolerance"):
        td.compute_consensus(
            min_consensus=0.5,
            temporal_tolerance=0,
            spatial_tolerance=-1,
            show_progress=False,
        )
    with pytest.raises(ValueError, match="stitch_meridian"):
        td.compute_consensus(
            min_consensus=0.5,
            temporal_tolerance=0,
            spatial_tolerance=0,
            stitch_meridian="yes",
            show_progress=False,
        )


def test_consensus_rate_map_runs():
    """Smoke test for consensus rate map plotting."""
    import matplotlib

    matplotlib.use("Agg")

    fields = {}
    for i in range(3):
        fields[f"foo_r{i + 1}_cluster"] = np.full((2, 1, 2), -1, dtype=np.float32)
    fields["foo_r1_cluster"][0, 0, 0] = 0
    fields["foo_r2_cluster"][0, 0, 0] = 0
    fields["foo_r3_cluster"][1, 0, 1] = 0

    td = setup_synthetic_consensus_toad(fields, add_latlon=True)
    td.compute_consensus(
        min_consensus=0.67,
        temporal_tolerance=0,
        spatial_tolerance=0,
        min_cluster_area=None,
        show_progress=False,
    )

    # Disable borders to avoid a second Natural Earth download (flaky in CI).
    map_style = {"borders": False}

    fig, ax = td.plot.consensus_rate_map(time_reduce="max", map_style=map_style)
    assert fig is not None
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.close(fig)

    fig_h, ax_h = td.plot.consensus_rate_map(
        time_reduce="max",
        map_style=map_style,
        colorbar_orientation="horizontal",
        colorbar_location="left",
        colorbar_shrink=0.38,
        colorbar_pad=0.04,
        colorbar_aspect=28.0,
    )
    assert fig_h is not None
    assert ax_h is not None
    assert len(fig_h.axes) >= 2
    plt.close(fig_h)
