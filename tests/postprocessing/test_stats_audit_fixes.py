"""Regression tests for the cluster statistics and scores fixed after the 2026-11 audit."""

import numpy as np
import pytest
import xarray as xr

from toad import TOAD
from toad.postprocessing.stats.general import GeneralStats


def make_td(values, labels, time=None, lat=None, lon=None, shifts=None):
    """TOAD object with a handmade cluster variable ``v_dts_cluster`` (dims time, lat, lon)."""
    dims = ("time", "lat", "lon")
    T, NY, NX = values.shape
    coords = {
        "time": np.arange(T, dtype=float) if time is None else time,
        "lat": np.arange(NY, dtype=float) if lat is None else lat,
        "lon": np.arange(NX, dtype=float) if lon is None else lon,
    }
    ds = xr.Dataset({"v": (dims, values)}, coords=coords)
    ds["v_dts"] = (dims, np.zeros_like(values) if shifts is None else shifts)
    ds["v_dts"].attrs.update({"variable_type": "shift", "base_variable": "v"})
    ids = np.unique(labels[np.isfinite(labels)]).astype(int)
    ds["v_dts_cluster"] = (dims, labels.astype(float))
    ds["v_dts_cluster"].attrs.update(
        {
            "variable_type": "cluster",
            "base_variable": "v",
            "shifts_variable": "v_dts",
            "cluster_ids": ids,
            "method_name": "manual",
            "shift_threshold": 0.5,
        }
    )
    return TOAD(ds, log_level="WARNING")


class _FakeTD:
    """Minimal stand-in that returns fixed cell series from get_timeseries."""

    def __init__(self, Y):
        self.Y = Y

    def get_timeseries(self, *args, **kwargs):
        return self.Y


def _single_series_td(series, n_cells=4):
    T = len(series)
    v = np.broadcast_to(np.asarray(series, float)[:, None, None], (T, 2, n_cells // 2))
    lab = np.full(v.shape, np.nan)
    lab[T // 2] = 0
    return make_td(v.copy(), lab)


# --- 4.1 / 4.2: documented values of score_heaviside and score_consistency ---


def test_heaviside_documented_values():
    t = np.arange(100)
    step = _single_series_td((t >= 50).astype(float))
    linear = _single_series_td(t / 99.0)
    assert step.stats("v_dts_cluster").general.score_heaviside(0) == pytest.approx(
        0.25, abs=1e-3
    )
    assert linear.stats("v_dts_cluster").general.score_heaviside(0) == pytest.approx(
        0.0, abs=1e-12
    )


def test_consistency_documented_degenerate_values():
    rng = np.random.default_rng(0)
    two = GeneralStats(_FakeTD(rng.normal(size=(2, 50))), "x").score_consistency(0)
    three = GeneralStats(_FakeTD(rng.normal(size=(3, 50))), "x").score_consistency(0)
    assert two == pytest.approx(1.0)
    assert three == pytest.approx(np.sqrt(2.0))
