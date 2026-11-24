"""Tests for compute_shifts and its handling of method objects."""

import numpy as np
import pytest
import xarray as xr

from toad import TOAD
from toad.shifts import EDGE
from toad.shifts import _clone_method


def _step_dataset(nt=90, ny=6, nx=7, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(nt, ny, nx))
    for j in range(ny):
        for i in range(nx):
            data[20 + 3 * j + i :, j, i] += 5.0
    return xr.Dataset(
        {"x": (("time", "lat", "lon"), data)},
        coords=dict(
            time=np.arange(nt, dtype=float),
            lat=np.linspace(-60, 60, ny),
            lon=np.linspace(0, 300, nx),
        ),
    )


def _run(ds, method, run_parallel):
    td = TOAD(ds.copy(), log_level="ERROR")
    td.compute_shifts(
        "x",
        method=method,
        run_parallel=run_parallel,
        show_progress=False,
        n_jobs=2,
    )
    return td.data["x_dts"]


class TestCloneMethod:
    def test_clone_keeps_none_attributes(self):
        m = EDGE(lmin=5, lmax=15, lcutoff=2, smoothing_scale=None)
        clone = _clone_method(m)
        assert clone is not m
        assert clone.smoothing_scale is None
        assert vars(clone) == vars(m)

    def test_edge_no_smoothing_serial_matches_parallel(self):
        ds = _step_dataset()
        kw = dict(lmin=5, lmax=15, lcutoff=2, smoothing_scale=None)
        serial = _run(ds, EDGE(**kw), run_parallel=False)
        parallel = _run(ds, EDGE(**kw), run_parallel=True)
        # the parallel path returns float32, so compare to float32 precision
        np.testing.assert_array_equal(serial.values != 0, parallel.values != 0)
        np.testing.assert_allclose(serial.values, parallel.values, rtol=1e-6)
        # and "no smoothing" really differs from "auto" on this data
        auto = _run(ds, EDGE(lmin=5, lmax=15, lcutoff=2), run_parallel=False)
        assert not np.array_equal(serial.values, auto.values)


if __name__ == "__main__":
    pytest.main([__file__])
