"""Consensus summary centre longitude for clusters crossing the grid seam (item 4.5)."""

import logging

import numpy as np
import pytest
import xarray as xr

from toad import TOAD
from toad.utils import _attrs

logging.getLogger("TOAD").setLevel(logging.ERROR)

T, NY, NX = 4, 6, 36


def _summary(lon, cols):
    lat = np.linspace(-75, 75, NY)
    a = np.full((T, NY, NX), -1.0)
    a[1, 2:4, cols] = 0
    coords = {"time": np.arange(T, dtype=float), "lat": lat, "lon": lon}
    ds = xr.Dataset({"foo": (("time", "lat", "lon"), np.zeros((T, NY, NX)))}, coords)
    for n in ["c1", "c2"]:
        da = xr.DataArray(a.copy(), dims=("time", "lat", "lon"), coords=ds.coords)
        da.attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.array([0]),
            }
        )
        ds[n] = da
    td = TOAD(ds, time_dim="time")
    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
        min_cluster_area=None,
    )
    df = td.aggregate.consensus_summary()
    assert len(df) == 1
    return float(df["mean_lon"].iloc[0]), float(df["mean_lat"].iloc[0])


def test_greenwich_cluster_on_0_360_grid():
    lon = np.arange(0, 360, 10.0)
    mean_lon, _ = _summary(lon, [0, 1, NX - 2, NX - 1])  # 340, 350, 0, 10
    assert mean_lon == pytest.approx(355.0)


def test_dateline_cluster_on_minus180_180_grid():
    lon = np.arange(-180, 180, 10.0)
    # -180, -170, 160, 170 -> centre at 175 (the dateline cluster spans 160 to 190)
    mean_lon, _ = _summary(lon, [0, 1, NX - 2, NX - 1])
    assert mean_lon == pytest.approx(175.0)


@pytest.mark.parametrize(
    "lon",
    [np.arange(0, 360, 10.0), np.arange(-180, 180, 10.0), np.arange(5, 365, 10.0)],
)
@pytest.mark.parametrize(
    "cols", [[3, 4, 5, 6], [0, 1, 2], [NX - 3, NX - 2, NX - 1], [17]]
)
def test_non_crossing_clusters_keep_arithmetic_mean(lon, cols):
    mean_lon, mean_lat = _summary(lon, cols)
    assert mean_lon == pytest.approx(np.float32(np.mean(lon[cols])), abs=0)
    assert mean_lat == pytest.approx(np.float32(np.mean(np.linspace(-75, 75, NY)[2:4])))
