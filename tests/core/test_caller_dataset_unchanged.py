"""TOAD must not modify the dataset object the caller passed in."""

import numpy as np
import xarray as xr

from toad import TOAD


def _dataset() -> xr.Dataset:
    data = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), data)},
        coords={"time": np.arange(3), "lat": np.arange(4), "lon": np.arange(5)},
        attrs={"title": "original"},
    )
    ds["v"].attrs["units"] = "m"
    ds["v"].encoding["zlib"] = False
    ds["v"].encoding["dtype"] = "float64"
    return ds


def _snapshot(ds: xr.Dataset):
    return (
        list(ds.variables),
        dict(ds.attrs),
        {k: dict(ds[k].attrs) for k in ds.variables},
        {k: dict(ds[k].encoding) for k in ds.variables},
        ds["v"].values.copy(),
    )


def _assert_same(before, after):
    assert before[0] == after[0]
    assert before[1] == after[1]
    assert before[2] == after[2]
    assert before[3] == after[3]
    np.testing.assert_array_equal(before[4], after[4])


def test_preprocess_variable_does_not_modify_caller_dataset():
    ds = _dataset()
    before = _snapshot(ds)
    td = TOAD(ds, log_level="WARNING")
    td.preprocess.preprocess_variable("v", lambda x: x > 10)
    assert np.isnan(td.data["v"].values[0, 0, 0])
    _assert_same(before, _snapshot(ds))


def test_save_does_not_modify_caller_dataset(tmp_path):
    ds = _dataset()
    before = _snapshot(ds)
    td = TOAD(ds, log_level="WARNING")
    td.data["extra"] = td.data["v"] * 2
    td.data.attrs["note"] = "added by toad"
    td.save(path=str(tmp_path / "out.nc"))
    _assert_same(before, _snapshot(ds))


def test_dataarray_input_does_not_modify_caller_encoding(tmp_path):
    da = _dataset()["v"]
    da.encoding["zlib"] = False
    td = TOAD(da, log_level="WARNING")
    td.save(path=str(tmp_path / "out.nc"))
    assert da.encoding.get("zlib") is False
