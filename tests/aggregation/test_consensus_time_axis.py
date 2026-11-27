"""Shift-time samples must not assume that time is the first axis."""

import numpy as np
import pytest
import xarray as xr

from toad import TOAD
from toad.utils import _attrs
from toad.utils.cluster_consensus_utils import consensus_shift_time_distribution


def _build(order, T, ny, nx):
    lab = -np.ones((T, ny, nx), np.float32)
    lab[T - 5 :, 1:3, 1:3] = 0  # event at the last five time steps
    coords = {
        "time": np.arange(T, dtype=np.float32),
        "y": np.arange(ny),
        "x": np.arange(nx),
    }
    dims = ("time", "y", "x")
    ds = xr.Dataset(
        {"foo": xr.DataArray(np.ones(lab.shape, np.float32), coords=coords, dims=dims)}
    )
    for n in ("a", "b"):
        da = xr.DataArray(lab, coords=coords, dims=dims, name=n)
        da.attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.array([0], np.int32),
            }
        )
        ds[n] = da
    return TOAD(ds.transpose(*order), time_dim="time")


@pytest.mark.parametrize("sizes", [(30, 8, 9), (12, 12, 12)])
@pytest.mark.parametrize("order", [("time", "y", "x"), ("y", "x", "time")])
def test_label_shift_time_samples_any_time_axis(order, sizes):
    T = sizes[0]
    td = _build(order, *sizes)
    df = td.aggregate.label_shift_time_samples("a")
    assert len(df) == 20
    np.testing.assert_array_equal(np.unique(df["transition_time"]), np.arange(T - 5, T))
    dists = td.aggregate.label_shift_time_distributions("a")
    np.testing.assert_array_equal(np.unique(dists[0]), np.arange(T - 5, T))


def test_consensus_shift_time_distribution_time_last_labels():
    td = _build(("time", "y", "x"), 12, 12, 12)
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        stitch_meridian=False,
        show_progress=False,
    )
    da = td.data[td.consensus_cluster_vars[-1]]
    _, df_ref = consensus_shift_time_distribution(td, da)
    _, df_alt = consensus_shift_time_distribution(td, da.transpose("y", "x", "time"))
    np.testing.assert_array_equal(
        np.unique(df_alt["transition_time"]), np.arange(7, 12)
    )
    assert len(df_alt) == len(df_ref)
