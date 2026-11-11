"""Tests for ASDETECT shift detection method."""

import numpy as np
import pytest
import xarray as xr

from toad import TOAD
from toad.shifts import ASDETECT


@pytest.fixture(scope="module")
def coarsened_toad():
    """Load and coarsen test data once per module for efficiency."""
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=15, lon=20, time=3, boundary="trim").reduce(np.mean)
    return td


class TestASDetect:
    """Tests for ASDETECT shift detection."""

    @pytest.mark.parametrize(
        "segmentation,expected_mean,expected_std",
        [
            pytest.param("original", 0.03111361, 0.18984994, id="centered"),
            pytest.param("two_sided", 0.04242268, 0.17294617, id="two_sided"),
        ],
    )
    def test_asdetect_on_real_data(
        self, coarsened_toad, segmentation, expected_mean, expected_std
    ):
        """Test ASDETECT with different segmentation modes on synthetic data."""
        td = coarsened_toad

        td.compute_shifts(
            "ts",
            ASDETECT(segmentation=segmentation),
            overwrite=True,
            run_parallel=False,
        )
        shifts = td.get_shifts("ts")

        np.testing.assert_allclose(
            shifts.mean().values, expected_mean, rtol=2e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            shifts.std().values, expected_std, rtol=2e-5, atol=1e-6
        )

    def test_asdetect_two_sided_unit(self):
        """Unit test for two_sided segmentation with known input/output."""
        np.random.seed(4)
        data = np.random.randn(50)
        data[20:] += 20  # Clear shift at index 20

        shifts = ASDETECT(segmentation="two_sided").fit_predict(
            data, np.arange(len(data), dtype=np.float64)
        )

        expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.04166667,
                0.04166667,
                0.04166667,
                0.125,
                0.16666667,
                0.20833333,
                0.375,
                0.45833333,
                0.54166667,
                0.58333333,
                0.75,
                0.75,
                0.75,
                0.70833333,
                0.58333333,
                0.54166667,
                0.375,
                0.375,
                0.25,
                0.20833333,
                0.125,
                0.125,
                0.08333333,
                0.08333333,
                0.04166667,
                0.04166667,
                0.0,
                0.0,
                -0.04166667,
                -0.04166667,
                -0.04166667,
                -0.04166667,
                -0.04166667,
                -0.04166667,
                -0.04166667,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        np.testing.assert_allclose(shifts, expected, atol=1e-6)

    def test_asdetect_leading_trailing_nan_matches_finite_core(self):
        """Leading/trailing NaN padding is trimmed; core matches a finite series."""
        np.random.seed(4)
        data = np.random.randn(50)
        data[20:] += 20

        expected = ASDETECT(segmentation="two_sided").fit_predict(
            data, np.arange(len(data), dtype=np.float64)
        )

        padded = np.full(60, np.nan)
        padded[5:55] = data
        shifts = ASDETECT(segmentation="two_sided").fit_predict(
            padded, np.arange(len(padded), dtype=np.float64)
        )

        np.testing.assert_allclose(shifts[5:55], expected, atol=1e-6)
        np.testing.assert_allclose(shifts[:5], 0.0)
        np.testing.assert_allclose(shifts[55:], 0.0)

    def test_asdetect_internal_nan_returns_zeros(self):
        """Internal NaN gaps still yield an all-zero detection series."""
        data = np.arange(30, dtype=np.float64)
        data[15] = np.nan
        shifts = ASDETECT(segmentation="two_sided").fit_predict(
            data, np.arange(len(data), dtype=np.float64)
        )
        np.testing.assert_allclose(shifts, 0.0)

    def test_compute_shifts_persists_resolved_lmax(self, coarsened_toad):
        """Resolved lmax (default n//3) is written to attrs, including under joblib."""
        td = coarsened_toad
        n_time = td.data.sizes[td.time_dim]

        td.compute_shifts(
            "ts",
            ASDETECT(segmentation="two_sided"),
            overwrite=True,
            run_parallel=False,
        )
        shifts = td.get_shifts("ts")
        assert int(shifts.attrs["shifts_lmax"]) == n_time // 3
        assert int(shifts.attrs["shifts_lmin"]) == 5

        # Parallel path clones the method; attrs must still record resolved values.
        td.compute_shifts(
            "ts",
            ASDETECT(lmin=5, lmax=8, segmentation="two_sided"),
            overwrite=True,
            run_parallel=True,
            n_jobs=2,
        )
        shifts = td.get_shifts("ts")
        assert int(shifts.attrs["shifts_lmax"]) == 8
        assert int(shifts.attrs["shifts_lmin"]) == 5

    def test_automatic_lmax_follows_each_series(self):
        """A reused detector must not keep lmax from the previous series."""
        method = ASDETECT()
        for n in (30, 90):
            data = np.zeros((n, 2, 2), dtype=np.float64)
            data[:, 0, 0] = np.linspace(0.0, 1.0, n)
            ds = xr.Dataset(
                {"ts": (("time", "lat", "lon"), data)},
                coords={
                    "time": np.arange(n),
                    "lat": np.array([0.0, 1.0]),
                    "lon": np.array([0.0, 1.0]),
                },
            )
            td = TOAD(ds, log_level="ERROR")
            td.compute_shifts(method=method, run_parallel=False, show_progress=False)
            shifts = td.get_shifts("ts")
            assert int(shifts.attrs["shifts_lmax"]) == n // 3
            assert method.lmax is None


class TestASDetectPrecision:
    """float32 input is detected in float64 and returned as float32."""

    @pytest.mark.parametrize("segmentation", ["two_sided", "original"])
    @pytest.mark.parametrize(
        "times",
        [
            np.arange(1850, 2000, dtype=np.int64),
            np.arange(1850, 2000, dtype=np.float64),
        ],
        ids=["int_years", "float_years"],
    )
    def test_float32_matches_float64(self, segmentation, times):
        rng = np.random.default_rng(7)
        # Large mean and small variability, like a temperature in K.
        values = (288.0 + 0.3 * rng.standard_normal(times.size).cumsum()).astype(
            np.float32
        )
        method = ASDETECT(segmentation=segmentation)
        got = method.fit_predict(values, times)
        expected = method.fit_predict(values.astype(np.float64), times)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, expected.astype(np.float32))

    def test_float32_compute_shifts_matches_float64(self):
        rng = np.random.default_rng(8)
        n = 120
        data = (288.0 + 0.3 * rng.standard_normal((n, 3, 4)).cumsum(axis=0)).astype(
            np.float32
        )
        coords = {
            "time": np.arange(1850, 1850 + n),
            "lat": np.arange(3.0),
            "lon": np.arange(4.0),
        }
        out = {}
        for dtype in (np.float32, np.float64):
            ds = xr.Dataset(
                {"ts": (("time", "lat", "lon"), data.astype(dtype))}, coords=coords
            )
            td = TOAD(ds, log_level="ERROR")
            td.compute_shifts("ts", run_parallel=False, show_progress=False)
            out[dtype] = td.get_shifts("ts").values
        np.testing.assert_array_equal(
            out[np.float32], out[np.float64].astype(np.float32)
        )


class TestASDetectHelpers:
    """The allocation-free helpers must match the plain numpy computations."""

    def test_median_matches_numpy(self):
        from toad.shifts.methods.asdetect import median

        rng = np.random.default_rng(0)
        for _ in range(500):
            x = rng.standard_normal(int(rng.integers(1, 120)))
            if rng.random() < 0.5:
                x = np.round(x, 1)  # ties
            original = x.copy()
            assert median(x) == np.median(x)
            np.testing.assert_array_equal(x, original)  # input left untouched

    def test_linear_fit_matches_numpy_sums(self):
        from toad.shifts.methods.asdetect import linear_fit

        def reference(x, y):
            # Sequential sums, as numba's np.sum, with numpy's array dtypes.
            n = len(x)
            sum_x, sum_y = x.cumsum()[-1], y.cumsum()[-1]
            sum_xy, sum_x2 = (x * y).cumsum()[-1], (x * x).cumsum()[-1]
            den = n * sum_x2 - sum_x * sum_x
            return (n * sum_xy - sum_x * sum_y) / den if den != 0 else 0.0

        rng = np.random.default_rng(1)
        for _ in range(200):
            n = int(rng.integers(3, 80))
            y = rng.standard_normal(n).cumsum()
            x = np.round(np.linspace(0.2, 4.0, n), 2)
            assert linear_fit(x, y) == reference(x, y)

    @pytest.mark.parametrize(
        "x_dtype,y_dtype",
        [
            (np.float64, np.float64),
            (np.float64, np.float32),
            (np.int64, np.float64),
            (np.int64, np.float32),
        ],
    )
    def test_linear_fit_matches_numba_np_sum(self, x_dtype, y_dtype):
        """Same sums and dtypes as the numba ``np.sum`` formulation.

        Under numba, an int64 array times a float32 array is float32, so the
        x*y sum must stay in float32 for int times with float32 values.
        """
        from numba import njit

        from toad.shifts.methods.asdetect import linear_fit

        @njit
        def np_sum_fit(x, y):
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            den = n * sum_x2 - sum_x * sum_x
            return np.float64((n * sum_xy - sum_x * sum_y) / den if den != 0 else 0.0)

        rng = np.random.default_rng(2)
        for _ in range(100):
            n = int(rng.integers(3, 80))
            y = (rng.standard_normal(n).cumsum() * 100).astype(y_dtype)
            x = (np.arange(n) + 1850).astype(x_dtype)
            assert linear_fit(x, y) == np_sum_fit(x, y)
