"""Tests for converting ASDETECT ``timescale`` into segment lengths."""

import logging

import cftime
import numpy as np
import pytest

from toad.shifts import ASDETECT


def _old_lengths(times, timescale):
    """Conversion used up to 1.0.10: first time step, truncated."""
    dt = np.diff(times)[0]
    lmin = int(timescale[0] / dt)
    lmax = min(int(timescale[1] / dt), len(times) // 3)
    return lmin, lmax


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def toad_warnings():
    handler = _Records()
    logger = logging.getLogger("TOAD")
    level = logger.level  # other tests may leave the TOAD logger at ERROR
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    yield handler.messages
    logger.removeHandler(handler)
    logger.setLevel(level)


@pytest.mark.parametrize(
    "times, timescale",
    [
        # GMST axes whose first step is a hair above or at 0.02
        (np.arange(-1, 12, 0.02), (0.5, 3.5)),
        (np.arange(0, 12, 0.02), (0.5, 3.5)),
        (np.linspace(0, 12, 601), (0.5, 3.5)),
        (np.arange(0.01, 12, 0.02), (0.5, 3.5)),
        (np.arange(1850.0, 2101.0), (10.0, 40.0)),
        (np.arange(0, 400) * 86400.0, (10 * 86400.0, 60 * 86400.0)),
    ],
)
def test_even_spacing_unchanged(times, timescale, toad_warnings):
    assert ASDETECT(timescale=timescale)._get_segment_lengths(times) == _old_lengths(
        times, timescale
    )
    assert not toad_warnings


def test_paper_antarctic_example_lengths():
    # The paper quotes lmin=24, lmax=174 for timescale (0.5, 3.5) on a GMST axis.
    times = np.arange(-1, 12, 0.02)
    assert ASDETECT(timescale=(0.5, 3.5))._get_segment_lengths(times) == (24, 174)


def _monthly_seconds(calendar, years=30):
    dates = [
        cftime.datetime(2000 + y, m, 15, calendar=calendar)
        for y in range(years)
        for m in range(1, 13)
    ]
    return np.array([(d - dates[0]).total_seconds() for d in dates], dtype=np.float64)


@pytest.mark.parametrize("calendar", ["noleap", "proleptic_gregorian"])
def test_calendar_months_use_mean_step_and_warn_once(calendar, toad_warnings):
    times = _monthly_seconds(calendar)
    year = 365.25 * 24 * 3600
    method = ASDETECT(timescale=(year, 5 * year))
    assert method._get_segment_lengths(times) == (12, 60)
    method._get_segment_lengths(times)
    rng = np.random.default_rng(0)
    method.fit_predict(rng.normal(size=times.size), times)
    assert len(toad_warnings) == 1
    assert "vary slightly" in toad_warnings[0]


@pytest.mark.parametrize("run_parallel", [False, True])
def test_compute_shifts_monthly_cftime(run_parallel, toad_warnings):
    import xarray as xr

    from toad import TOAD

    time = xr.date_range(
        "2000-01-15", periods=360, freq="MS", calendar="noleap", use_cftime=True
    )
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"x": (("time", "lat", "lon"), rng.normal(size=(360, 3, 3)))},
        coords=dict(time=time, lat=[0.0, 1.0, 2.0], lon=[0.0, 1.0, 2.0]),
    )
    td = TOAD(ds, log_level="WARNING")
    year = 365.25 * 24 * 3600
    td.compute_shifts(
        "x",
        method=ASDETECT(timescale=(year, 5 * year)),
        run_parallel=run_parallel,
        n_jobs=2,
        show_progress=False,
    )
    attrs = td.data["x_dts"].attrs
    assert (attrs["shifts_lmin"], attrs["shifts_lmax"]) == ("12", "60")
    assert sum("vary slightly" in m for m in toad_warnings) == 1


def test_irregular_spacing_raises():
    times = np.concatenate([np.arange(0.0, 50.0), np.arange(52.0, 150.0)])
    with pytest.raises(ValueError, match="regular temporal spacing"):
        ASDETECT(timescale=(5.0, 20.0))._get_segment_lengths(times)


def test_unrepresentative_first_step_raises():
    times = np.concatenate([[0.0, 0.1], np.arange(1.0, 150.0)])
    with pytest.raises(ValueError, match="regular temporal spacing"):
        ASDETECT(timescale=(5.0, 20.0))._get_segment_lengths(times)


def test_irregular_spacing_without_timescale_is_allowed():
    times = np.concatenate([np.arange(0.0, 50.0), np.arange(52.0, 150.0)])
    assert ASDETECT(lmin=5, lmax=20)._get_segment_lengths(times) == (5, 20)
