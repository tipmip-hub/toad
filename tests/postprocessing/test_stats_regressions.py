"""Regression tests for the cluster statistics and scores."""

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


# --- documented values of score_heaviside and score_consistency ---


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


# --- steepest_gradient returns the time coordinate in its own type ---


def _jump_td(time):
    T = 40
    t = np.arange(T)
    v = np.broadcast_to(
        ((t >= 20).astype(float) + np.linspace(0, 0.01, T))[:, None, None], (T, 4, 4)
    ).copy()
    lab = np.full(v.shape, np.nan)
    lab[15:25, :2, :2] = 0
    return make_td(v, lab, time=time)


def test_steepest_gradient_numeric_time():
    td = _jump_td(None)
    out = td.stats("v_dts_cluster").time.steepest_gradient(0)
    assert isinstance(out, float) and out == 20.0


def test_steepest_gradient_datetime64_time():
    import pandas as pd

    times = pd.date_range("2000-01-01", periods=40, freq="YS").values
    td = _jump_td(times)
    out = td.stats("v_dts_cluster").time.steepest_gradient(0)
    assert np.datetime64(out, "D") == np.datetime64("2020-01-01")


def test_steepest_gradient_cftime_time_and_all_stats():
    import cftime

    times = np.array([cftime.DatetimeNoLeap(2000 + i, 1, 1) for i in range(40)])
    td = _jump_td(times)
    ts = td.stats("v_dts_cluster").time
    assert ts.steepest_gradient(0) == cftime.DatetimeNoLeap(2020, 1, 1)
    stats = ts.all_stats(0)
    assert stats["steepest_gradient"] == cftime.DatetimeNoLeap(2020, 1, 1)


# --- score_overview passes each score only the arguments it accepts ---


def _two_cluster_td():
    rng = np.random.default_rng(4)
    T, NY, NX = 50, 4, 6
    t = np.arange(T)
    v = (t >= 25).astype(float)[:, None, None] + 0.1 * rng.normal(size=(T, NY, NX))
    v[:, :, 3:] = (t / T)[:, None, None] + 0.1 * rng.normal(size=(T, NY, 3))
    lab = np.full((T, NY, NX), np.nan)
    lab[25, :, :3] = 0
    lab[30, :, 3:] = 1
    return make_td(v, lab)


def test_score_overview_kwargs_reach_only_accepting_scores():
    g = _two_cluster_td().stats("v_dts_cluster").general
    cols = ["heaviside", "consistency", "spatial_autocorrelation", "nonlinearity"]
    ov = g.score_overview(aggregation="median")
    assert not ov[cols].isna().any().any()
    np.testing.assert_allclose(ov["aggregate_score"], ov[cols].prod(axis=1))
    assert ov["heaviside"].iloc[0] == g.score_heaviside(0, aggregation="median")
    assert ov["consistency"].iloc[0] == g.score_consistency(0)
    with pytest.raises(TypeError):
        g.score_overview(not_a_score_argument=1)


def test_score_overview_without_kwargs_and_nan_component(monkeypatch):
    g = _two_cluster_td().stats("v_dts_cluster").general
    cols = ["heaviside", "consistency", "spatial_autocorrelation", "nonlinearity"]
    ov = g.score_overview()
    expected = [
        float(
            np.prod(
                [
                    g.score_heaviside(c),
                    g.score_consistency(c),
                    g.score_spatial_autocorrelation(c),
                    g.score_nonlinearity(c),
                ]
            )
        )
        for c in (0, 1)
    ]
    assert ov["aggregate_score"].tolist() == expected
    assert not ov[cols].isna().any().any()

    def boom(self, cluster_id):
        raise RuntimeError("fail")

    monkeypatch.setattr(GeneralStats, "score_consistency", boom)
    ov2 = g.score_overview()
    assert ov2["consistency"].isna().all()
    assert ov2["aggregate_score"].isna().all()
