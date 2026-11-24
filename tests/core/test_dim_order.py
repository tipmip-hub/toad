"""TOAD normalises the dimension order to (time, lat-like, lon-like) on creation."""

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import DBSCAN  # type: ignore

from toad import TOAD

NT, NY, NX = 40, 6, 9


def _standard(ydim: str, xdim: str) -> xr.Dataset:
    """(time, y, x) data where two separate patches shift at different times."""
    rng = np.random.default_rng(1)
    t = np.arange(NT, dtype=float)
    data = 0.01 * rng.standard_normal((NT, NY, NX))
    data[:, :3, :4] += (t >= 12).astype(float)[:, None, None]
    data[:, 3:, 5:] -= 2 * (t >= 27).astype(float)[:, None, None]
    if ydim == "lat":
        ycoord, xcoord = np.linspace(-20, 20, NY), np.linspace(0, 80, NX)
    else:
        ycoord, xcoord = np.arange(NY), np.arange(NX)
    return xr.Dataset(
        {"v": (("time", ydim, xdim), data)},
        coords={"time": t, ydim: ycoord, xdim: xcoord},
    )


def _run(ds: xr.Dataset) -> TOAD:
    td = TOAD(ds, log_level="WARNING")
    td.compute_shifts(show_progress=False)
    eps = 2500 if "lat" in ds.dims else 3  # lat/lon grids cluster in km
    td.compute_clusters(method=DBSCAN(eps=eps, min_samples=3))
    return td


@pytest.mark.parametrize(
    "ydim, xdim, order",
    [
        ("lat", "lon", ("time", "lon", "lat")),
        ("lat", "lon", ("lat", "lon", "time")),
        ("y", "x", ("time", "x", "y")),
    ],
)
def test_nonstandard_order_gives_same_clusters(ydim, xdim, order):
    std = _standard(ydim, xdim)
    td_std = _run(std)

    permuted = std.transpose(*order)
    td = _run(permuted)

    # caller's dataset keeps its own order
    assert permuted["v"].dims == order
    assert td.space_dims == [ydim, xdim]
    assert td.data["v"].dims == ("time", ydim, xdim)

    cv = td_std.cluster_vars[0]
    assert td.cluster_vars == [cv]
    assert td.data[cv].dims == ("time", ydim, xdim)
    assert len(td_std.get_cluster_ids()) >= 2
    np.testing.assert_array_equal(td.data[cv].values, td_std.data[cv].values)
    np.testing.assert_array_equal(td.data["v_dts"].values, td_std.data["v_dts"].values)


def test_standard_order_is_untouched():
    ds = _standard("lat", "lon")
    td = TOAD(ds, log_level="WARNING")
    assert td.data["v"].dims == ("time", "lat", "lon")
    assert td.data["v"].variable._data is ds["v"].variable._data
    assert td.space_dims == ["lat", "lon"]


def test_unrecognised_spatial_names_keep_relative_order():
    ds = _standard("y", "x").rename({"y": "b", "x": "a"}).transpose("b", "time", "a")
    td = TOAD(ds, log_level="WARNING")
    assert td.data["v"].dims == ("time", "b", "a")
    assert td.space_dims == ["b", "a"]
    assert ds["v"].dims == ("b", "time", "a")
