"""Consensus results must not depend on the storage order of the spatial dims."""

import logging

import numpy as np
import pytest
import xarray as xr

from toad import TOAD
from toad.utils import _attrs

logging.getLogger("TOAD").setLevel(logging.ERROR)

T, NLAT, NLON = 4, 9, 36


def _build(order, lab):
    lon = np.arange(NLON) * 10.0
    lat = np.linspace(-80, 80, NLAT)
    coords = {"time": np.arange(T, dtype=np.float32), "lat": lat, "lon": lon}
    dims = ("time", "lat", "lon")
    ds = xr.Dataset(
        {"foo": xr.DataArray(np.ones(lab.shape, np.float32), coords=coords, dims=dims)}
    )
    for n in ("a", "b"):
        da = xr.DataArray(lab, coords=coords, dims=dims, name=n)
        da.attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.unique(lab[lab >= 0]).astype(np.int32),
            }
        )
        ds[n] = da
    return TOAD(ds.transpose(*order), time_dim="time")


def _run(order, lab, stitch):
    td = _build(order, lab)
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        stitch_meridian=stitch,
        show_progress=False,
        min_cluster_area=None,
    )
    da = td.data[td.consensus_cluster_vars[-1]]
    return td, da


def _seam_and_pole_labels():
    lab = -np.ones((T, NLAT, NLON), np.float32)
    lab[:, 4, [0, 1, NLON - 2, NLON - 1]] = 0  # crosses the 0/360 seam
    lab[:, 0, 10] = 1  # southernmost row
    lab[:, NLAT - 1, 10] = 1  # northernmost row, same longitude
    return lab


@pytest.mark.parametrize("stitch", ["auto", True])
@pytest.mark.parametrize(
    "order", [("time", "lat", "lon"), ("time", "lon", "lat"), ("lon", "lat", "time")]
)
def test_seam_is_stitched_and_poles_are_not_for_any_dim_order(order, stitch):
    lab = _seam_and_pole_labels()
    td, da = _run(order, lab, stitch)
    assert da.attrs["stitch_meridian_applied"] == 1
    assert da.dims == (td.time_dim, *td.space_dims)
    arr = da.transpose("time", "lat", "lon").values
    seam_ids = np.unique(arr[:, 4, [0, 1, NLON - 2, NLON - 1]])
    assert seam_ids.size == 1 and seam_ids[0] >= 0
    # The two polar cells must stay two clusters (latitude is never wrapped).
    assert arr[0, 0, 10] >= 0 and arr[0, NLAT - 1, 10] >= 0
    assert arr[0, 0, 10] != arr[0, NLAT - 1, 10]
    assert np.unique(arr[arr >= 0]).size == 3


def test_lon_first_matches_lat_first_after_transpose():
    lab = _seam_and_pole_labels()
    _, ref = _run(("time", "lat", "lon"), lab, True)
    _, alt = _run(("time", "lon", "lat"), lab, True)
    np.testing.assert_array_equal(
        alt.transpose("time", "lat", "lon").values, ref.values
    )


def test_longitude_last_spatial_dims_curvilinear():
    from toad.utils.cluster_consensus_utils import longitude_last_spatial_dims

    lon2d, lat2d = np.meshgrid(np.arange(36) * 10.0, np.linspace(-80, 80, 9))
    ds = xr.Dataset(
        coords={
            "lon": (("j", "i"), lon2d),
            "lat": (("j", "i"), lat2d),
        }
    )
    assert longitude_last_spatial_dims(ds, ("j", "i")) == ("j", "i")
    assert longitude_last_spatial_dims(ds, ("i", "j")) == ("j", "i")
    # No longitude coordinate: order is left alone.
    assert longitude_last_spatial_dims(xr.Dataset(), ("x", "y")) == ("x", "y")
