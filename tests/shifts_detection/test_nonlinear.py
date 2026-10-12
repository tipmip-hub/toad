"""Tests for NONLINEAR (optional pyunicorn)."""

import numpy as np
import pytest

pyunicorn = pytest.importorskip("pyunicorn")

from toad.shifts import NONLINEAR, add_recurrence_diagnostic  # noqa: E402


class TestNonlinear:
    def test_fit_predict_single_output(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=120)
        ts[60:] += 2.5
        times = np.arange(120, dtype=float)
        m = NONLINEAR(window_size=40, window_step=5, lmin=5, lmax=15)
        out = m.fit_predict(ts, times)
        assert out.shape == (120,)
        assert np.isfinite(out).all()

    def test_recurrence_1d_finite(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=120)
        times = np.arange(120, dtype=float)
        m = NONLINEAR(window_size=40, window_step=5)
        rec = m.recurrence_1d(ts, times)
        assert rec.shape == (120,)
        assert np.isfinite(rec).any()

    def test_compute_shifts_integration(self, tmp_path):
        import xarray as xr

        from toad import TOAD

        time = np.arange(120)
        data = np.random.default_rng(0).normal(size=(120, 2, 2))
        data[60:, :, :] += 2.5
        path = tmp_path / "nl.nc"
        xr.Dataset(
            {"ts": (("time", "lat", "lon"), data)},
            coords={
                "time": time,
                "lat": np.array([0.0, 1.0]),
                "lon": np.array([0.0, 1.0]),
            },
        ).to_netcdf(path)

        td = TOAD(str(path))
        td.compute_shifts(
            "ts",
            NONLINEAR(window_size=40, window_step=5, lmin=5, lmax=15),
            overwrite=True,
            run_parallel=False,
            show_progress=False,
        )
        assert "ts_dts" in td.data
        assert np.isfinite(td.data["ts_dts"].values).any()

        m = NONLINEAR(window_size=40, window_step=5, lmin=5, lmax=15)
        name = add_recurrence_diagnostic(td, "ts", m, overwrite=True)
        assert name == "ts_transitivity"
        assert name in td.data
        assert np.isfinite(td.data[name].values).any()
