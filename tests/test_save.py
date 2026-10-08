"""Tests for TOAD.save atomic netCDF writes."""

import numpy as np
import xarray as xr

from toad import TOAD


def test_save_suffix_overwrites_while_destination_open(tmp_path):
    """save(suffix=...) must replace an existing *_toad.nc open for read."""
    src = tmp_path / "mlotst_annualmax.nc"
    out = tmp_path / "mlotst_annualmax_toad.nc"

    ds = xr.Dataset(
        {"mlotst": (("time", "lat", "lon"), np.ones((3, 4, 5), dtype=np.float32))},
        coords={
            "time": np.arange(3),
            "lat": np.arange(4),
            "lon": np.arange(5),
        },
    )
    ds.to_netcdf(src)

    td = TOAD(str(src))
    td.save("toad")

    reader = xr.open_dataset(out, engine="netcdf4")
    try:
        td.data["mlotst"].values[0, 0, 0] = 42.0
        td.save("toad")
    finally:
        reader.close()

    with xr.open_dataset(out, engine="netcdf4") as saved:
        assert float(saved["mlotst"].values[0, 0, 0]) == 42.0


def test_save_overwrite_inplace_while_open(tmp_path):
    src = tmp_path / "data.nc"
    ds = xr.Dataset(
        {"mlotst": (("time", "lat", "lon"), np.ones((2, 3, 4), dtype=np.float32))},
        coords={"time": np.arange(2), "lat": np.arange(3), "lon": np.arange(4)},
    )
    ds.to_netcdf(src)

    td = TOAD(str(src))
    td.save(overwrite=True)
    reader = xr.open_dataset(src, engine="netcdf4")
    try:
        td.data["mlotst"].values[:] = 7.0
        td.save(overwrite=True)
    finally:
        reader.close()

    with xr.open_dataset(src, engine="netcdf4") as saved:
        assert float(saved["mlotst"].values[0, 0, 0]) == 7.0
