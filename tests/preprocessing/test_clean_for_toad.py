import numpy as np
import xarray as xr

from toad import TOAD
from toad.preprocessing import clean_for_toad


def test_clean_for_toad_drops_static_grid_geometry():
    """CESM-style POP grids carry TLAT/TLONG as 2D data vars without time."""
    ds = xr.Dataset(
        {
            "mlotst": (("time", "nlat", "nlon"), np.ones((3, 4, 5), dtype=np.float32)),
            "TLAT": (("nlat", "nlon"), np.zeros((4, 5))),
            "TLONG": (("nlat", "nlon"), np.zeros((4, 5))),
        },
        coords={"time": np.arange(3)},
    )
    cleaned = clean_for_toad(ds)
    assert list(cleaned.data_vars) == ["mlotst"]
    assert cleaned["mlotst"].dims == ("time", "nlat", "nlon")


def test_toad_auto_clean_cesm_like_grid():
    ds = xr.Dataset(
        {
            "mlotst": (("time", "nlat", "nlon"), np.ones((3, 4, 5), dtype=np.float32)),
            "TLAT": (("nlat", "nlon"), np.zeros((4, 5))),
        },
        coords={"time": np.arange(3)},
    )
    td = TOAD(ds, auto_clean=True)
    assert list(td.data.data_vars) == ["mlotst"]


def test_clean_for_toad_drops_cesm_pop_tlat_on_yx():
    """CESM POP annualmax files put TLAT on (y, x) while mlotst uses (nlat, nlon)."""
    ds = xr.Dataset(
        {
            "mlotst": (("time", "nlat", "nlon"), np.ones((3, 4, 5), dtype=np.float32)),
            "TLAT": (("y", "x"), np.zeros((4, 5))),
        },
        coords={"time": np.arange(3), "y": np.arange(4), "x": np.arange(5)},
    )
    cleaned = clean_for_toad(ds)
    assert list(cleaned.data_vars) == ["mlotst"]
    assert cleaned["mlotst"].dims == ("time", "nlat", "nlon")
    assert "y" not in cleaned.dims and "x" not in cleaned.dims
