"""The ``regridder=`` argument accepts only None or a HealPixRegridder (audit 2.6)."""

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import HDBSCAN

from toad import TOAD
from toad.clustering import _clone_regridder
from toad.regridding import HealPixRegridder
from toad.regridding.base import BaseRegridder
from toad.utils import _attrs


def _ds():
    """Global 5-degree grid with one positive and one negative event block."""
    time = np.arange(12, dtype=float)
    lat = np.arange(-87.5, 90.0, 5.0)
    lon = np.arange(2.5, 360.0, 5.0)
    sh = np.zeros((time.size, lat.size, lon.size))
    sh[3, 20:26, 10:16] = 1.0
    sh[8, 8:14, 40:46] = -1.0
    ds = xr.Dataset(
        {"x_dts": (("time", "lat", "lon"), sh)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["x_dts"].attrs[_attrs.BASE_VARIABLE] = "x"
    ds["x_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT
    return ds


class IdentityRegridder(BaseRegridder):
    def regrid(self, coords, weights, space_dims_size):
        return coords, weights

    def regrid_clusters_back(self, cluster_labels):
        return cluster_labels

    def map_orig_to_regrid(self, coords_2d):
        return np.arange(len(coords_2d))


def _cluster(td, **kwargs):
    td.compute_clusters(
        "x_dts",
        method=HDBSCAN(min_cluster_size=5, allow_single_cluster=True),
        **kwargs,
    )


def test_custom_regridder_raises_type_error():
    td = TOAD(_ds(), log_level="ERROR")
    with pytest.raises(TypeError, match="Custom regridders are no longer supported"):
        _cluster(td, regridder=IdentityRegridder())
    with pytest.raises(TypeError, match="Custom regridders"):
        _cluster(td, regridder=HealPixRegridder)  # a class, not an instance


def test_clone_keeps_every_setting():
    class MyHealPix(HealPixRegridder):
        def __init__(self, extra=3, **kwargs):
            super().__init__(**kwargs)
            self.extra = extra

    rg = MyHealPix(extra=7, nside=16)
    rg.df_healpix = "stale"
    clone = _clone_regridder(rg)
    assert type(clone) is MyHealPix
    assert (clone.extra, clone.nside) == (7, 16)
    assert clone.df_healpix.empty


@pytest.mark.parametrize("shift_direction", ["both", "positive"])
def test_passed_regridder_carries_final_labels(shift_direction):
    td = TOAD(_ds(), log_level="ERROR")
    rg = HealPixRegridder(nside=16)
    _cluster(td, regridder=rg, shift_direction=shift_direction)
    assert rg.nside == 16
    df = rg.df_healpix
    assert len(df) > 0 and "cluster" in df.columns

    # Every pixel cluster id is an id of the output field.
    out_ids = set(td.get_cluster_ids("x_dts_cluster").tolist())
    assert set(df["cluster"][df["cluster"] >= 0].astype(int)) == out_ids

    # The point at a cell centre carries the label of that cell.
    c = td.data["x_dts_cluster"]
    t_idx = int(df["time"].iloc[0])
    labels_t = c.isel(time=t_idx).values
    cells = np.argwhere(np.isfinite(labels_t))
    lat2d, lon2d = np.meshgrid(c.lat.values, c.lon.values, indexing="ij")
    pix = rg.map_orig_to_regrid(
        np.column_stack([lat2d[tuple(cells.T)], lon2d[tuple(cells.T)]])
    )
    n_checked = 0
    for (i, j), p in zip(cells, pix):
        match = df[(df["time"] == t_idx) & (df["hp_pix"] == p)]
        if len(match):
            assert int(match["cluster"].iloc[0]) == int(labels_t[i, j])
            n_checked += 1
    assert n_checked > 0


def test_reusing_a_regridder_gives_the_same_result():
    rg = HealPixRegridder()
    out = []
    for _ in range(2):
        td = TOAD(_ds(), log_level="ERROR")
        _cluster(td, regridder=rg)
        out.append(td.data["x_dts_cluster"].values)
    assert rg.nside is None
    np.testing.assert_array_equal(out[0], out[1])
