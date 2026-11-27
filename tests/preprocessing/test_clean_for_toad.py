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


def test_clean_for_toad_keeps_nemo_nav_lat_nav_lon():
    """auto_clean must keep the 2-D lat/lon coords that lat/lon detection supports."""
    ny, nx = 4, 5
    nav_lat, nav_lon = np.meshgrid(np.linspace(-10, 10, ny), np.linspace(0, 40, nx))
    ds = xr.Dataset(
        {"tos": (("time", "y", "x"), np.ones((3, ny, nx), dtype=np.float32))},
        coords={
            "time": np.arange(3),
            "nav_lat": (("y", "x"), nav_lat.T),
            "nav_lon": (("y", "x"), nav_lon.T),
        },
    )
    cleaned = clean_for_toad(ds)
    assert "nav_lat" in cleaned.coords and "nav_lon" in cleaned.coords

    td = TOAD(ds, auto_clean=True, log_level="WARNING")
    assert "nav_lat" in td.data.coords and "nav_lon" in td.data.coords
