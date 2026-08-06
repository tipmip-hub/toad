"""Tests for TOAD netCDF open behaviour."""

from unittest.mock import patch

import numpy as np
import xarray as xr
from toad import TOAD


def _minimal_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"mlotst": (("time", "lat", "lon"), np.zeros((2, 3, 4)))},
        coords={
            "time": [0, 1],
            "lat": np.arange(3.0),
            "lon": np.arange(4.0),
        },
    )


def test_open_path_uses_cftime_coder_by_default():
    ds = _minimal_dataset()
    with patch("os.path.exists", return_value=True):
        with patch("xarray.open_dataset", return_value=ds) as mock_open:
            TOAD("/fake/path.nc", log_level="CRITICAL")
    _, kwargs = mock_open.call_args
    decode_times = kwargs["decode_times"]
    assert isinstance(decode_times, xr.coders.CFDatetimeCoder)
    assert decode_times.use_cftime is True


def test_open_path_respects_decode_times_false():
    ds = _minimal_dataset()
    with patch("os.path.exists", return_value=True):
        with patch("xarray.open_dataset", return_value=ds) as mock_open:
            TOAD("/fake/path.nc", log_level="CRITICAL", decode_times=False)
    _, kwargs = mock_open.call_args
    assert kwargs["decode_times"] is False
