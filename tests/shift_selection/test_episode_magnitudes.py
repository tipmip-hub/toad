import numpy as np
import pytest
import xarray as xr

from toad.utils.shift_selection_utils import (
    _plateau_magnitude,
    collect_episode_magnitudes,
)


def test_plateau_magnitude_window_means():
    # peak at 3: pre [0:2)=100, post [5:7)=150  with L=3, G=1
    base = np.array([100.0, 100.0, 100.0, 125.0, 150.0, 150.0, 150.0])
    mag = _plateau_magnitude(base, peak=3, L=3, G=1)
    assert mag == pytest.approx(50.0)


def test_collect_episode_magnitudes_local_episodes():
    time = np.arange(10, dtype=float)
    var = np.array(
        [100.0, 100.0, 100.0, 100.0, 100.0, 130.0, 130.0, 130.0, 130.0, 130.0]
    )
    dts = np.array([0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0])

    ds = xr.Dataset(
        {
            "test": (("time",), var),
            "test_dts": (("time",), dts),
        },
        coords={"time": time},
    )

    mags = collect_episode_magnitudes(
        ds["test_dts"],
        ds["test"],
        "time",
        shift_threshold=0.5,
        L=3,
        G=1,
    )
    assert mags.size == 1
    # peak at index 4: pre mean[1:3]=100, post mean[6:8]=130
    assert mags[0] == pytest.approx(30.0)
