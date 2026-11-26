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


class TestMethodNotModified:
    def test_edge_reused_on_series_of_different_length(self):
        rng = np.random.default_rng(1)
        short = rng.normal(size=100)
        long = rng.normal(size=300)
        long[150:] += 4.0

        reused = EDGE()
        reused.fit_predict(short, np.arange(100.0))
        assert reused.lmin is None and reused.lmax is None
        assert reused.lcutoff is None

        expected = EDGE().fit_predict(long, np.arange(300.0))
        np.testing.assert_array_equal(
            reused.fit_predict(long, np.arange(300.0)), expected
        )

    @pytest.mark.parametrize("run_parallel", [False, True])
    def test_compute_shifts_leaves_method_unchanged(self, run_parallel):
        from toad.shifts import ASDETECT

        ds = _step_dataset()
        for method in (EDGE(), ASDETECT(lmax=1000)):  # lmax capped -> warn-once
            before = dict(vars(method))
            _run(ds, method, run_parallel=run_parallel)
            assert vars(method) == before

    def test_edge_attrs_record_resolved_lengths(self):
        ds = _step_dataset()
        out = _run(ds, EDGE(), run_parallel=False)
        assert out.attrs["shifts_lmin"] == "9"
        assert out.attrs["shifts_lmax"] == "18"
        assert out.attrs["shifts_lcutoff"] == "1"


class TestOverwrite:
    def test_overwrite_drops_dependent_clusters(self):
        import logging

        from sklearn.cluster import HDBSCAN  # type: ignore

        from toad.shifts import ASDETECT

        ds = _step_dataset()
        ds["y"] = ds["x"] * 2.0
        td = TOAD(ds, log_level="WARNING")
        td.compute_shifts(
            "x", method=ASDETECT(), show_progress=False, run_parallel=False
        )
        td.compute_shifts(
            "y", method=ASDETECT(), show_progress=False, run_parallel=False
        )
        td.compute_clusters("x_dts", method=HDBSCAN(min_cluster_size=5))
        td.compute_clusters("y_dts", method=HDBSCAN(min_cluster_size=5))
        assert set(td.cluster_vars) == {"x_dts_cluster", "y_dts_cluster"}

        records = []
        handler = logging.Handler(level=logging.WARNING)
        handler.emit = lambda r: records.append(r.getMessage())  # type: ignore
        logging.getLogger("TOAD").addHandler(handler)
        try:
            td.compute_shifts(
                "x",
                method=ASDETECT(lmin=10),
                overwrite=True,
                show_progress=False,
                run_parallel=False,
            )
        finally:
            logging.getLogger("TOAD").removeHandler(handler)

        assert "x_dts" in td.data
        assert "x_dts_cluster" not in td.data
        assert "y_dts_cluster" in td.data  # other shift variable: untouched
        assert any("x_dts_cluster" in m for m in records)


if __name__ == "__main__":
    pytest.main([__file__])


def _clustered_td():
    from sklearn.cluster import HDBSCAN  # type: ignore

    from toad.shifts import ASDETECT

    td = TOAD(_step_dataset(), log_level="ERROR")
    td.compute_shifts("x", method=ASDETECT(), show_progress=False, run_parallel=False)
    td.compute_clusters("x_dts", method=HDBSCAN(min_cluster_size=5))
    return td


def test_overwrite_also_drops_consensus_built_from_the_old_clusters():
    from toad.shifts import ASDETECT

    td = _clustered_td()
    td.compute_consensus(min_consensus=0.5, temporal_tolerance=1, spatial_tolerance=1)
    consensus = [str(v) for v in td.data.data_vars if "consensus" in str(v)]
    assert consensus
    td.compute_shifts(
        "x",
        method=ASDETECT(),
        overwrite=True,
        show_progress=False,
        run_parallel=False,
    )
    assert "x_dts" in td.data
    assert not [v for v in consensus if v in td.data]
    assert td.cluster_vars == []


def test_failing_detector_with_overwrite_keeps_shifts_and_clusters():
    from toad.shifts.methods.base import ShiftsMethod

    class Failing(ShiftsMethod):
        def fit_predict(self, values_1d, times_1d):
            raise RuntimeError("detector failed")

    td = _clustered_td()
    before = td.data["x_dts"].copy()
    with pytest.raises(RuntimeError, match="detector failed"):
        td.compute_shifts(
            "x",
            method=Failing(),
            overwrite=True,
            show_progress=False,
            run_parallel=False,
        )
    assert "x_dts_cluster" in td.data
    xr.testing.assert_identical(td.data["x_dts"], before)


def test_custom_method_object_is_used_as_given_in_the_serial_path():
    from toad.shifts.methods.base import ShiftsMethod

    class Counting(ShiftsMethod):
        def __init__(self):
            self.calls = 0

        def fit_predict(self, values_1d, times_1d):
            self.calls += 1
            return np.zeros_like(values_1d, dtype=float)

    method = Counting()
    td = TOAD(_step_dataset(), log_level="ERROR")
    td.compute_shifts("x", method=method, show_progress=False, run_parallel=False)
    assert method.calls > 0
