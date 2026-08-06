"""Tests for ASDETECT shift detection method."""

import numpy as np
import pytest

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
