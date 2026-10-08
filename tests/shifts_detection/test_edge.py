"""Tests for EDGE shift detection method."""

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.shifts import EDGE
from toad.shifts.methods.edge import construct_detection_ts


@pytest.fixture(scope="module")
def coarsened_toad():
    """Load and coarsen test data once per module for efficiency."""
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=15, lon=20, time=3, boundary="trim").reduce(np.mean)
    return td


class TestEDGE:
    """Tests for EDGE shift detection."""

    def test_returns_raw_abruptness_below_sigma_cut(self):
        """EDGE keeps sub-threshold abruptness; clustering applies shift_threshold."""
        values = np.array(
            list(np.linspace(0, 1, 40)) + list(np.linspace(1, 1.5, 60)),
            dtype=np.float64,
        )
        result = construct_detection_ts(
            values_1d=values,
            lmin=5,
            lmax=15,
            lcutoff=2,
            alpha=0.4,
            smoothing_scale=3,
            gradient_threshold="relative",
            gradient_threshold_multiplier=0.5,
        )

        assert np.count_nonzero(result) > 0
        assert np.any((np.abs(result) > 0) & (np.abs(result) <= 4))
        assert np.count_nonzero(np.where(np.abs(result) > 4, result, 0)) == 0

    def test_nan_series_returns_zeros(self):
        """NaN input returns all zeros, consistent with ASDETECT."""
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
        result = construct_detection_ts(
            values_1d=values,
            lmin=2,
            lmax=3,
            lcutoff=1,
            alpha=0.4,
            smoothing_scale=None,
        )
        assert result.shape == values.shape
        assert np.all(result == 0)

    def test_edge_on_real_data(self, coarsened_toad):
        """EDGE runs on synthetic dataset and returns abruptness scores."""
        td = coarsened_toad
        td.compute_shifts(
            "ts",
            EDGE(lmin=5, lmax=15, lcutoff=2, smoothing_scale=3),
            overwrite=True,
            run_parallel=False,
        )
        shifts = td.get_shifts("ts")

        assert shifts.attrs["method_name"] == "EDGE"
        assert "abruptness_threshold" not in shifts.attrs
        assert np.isfinite(shifts.max().values)

    def test_edge_clustering_with_shift_threshold_four(self, coarsened_toad):
        """4-sigma filtering is applied via shift_threshold in compute_clusters."""
        td = coarsened_toad
        td.drop_clusters()
        td.compute_shifts(
            "ts",
            EDGE(lmin=5, lmax=15, lcutoff=2, smoothing_scale=3),
            overwrite=True,
            run_parallel=False,
        )

        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=5),
            shift_threshold=4,
        )

        cluster_var = td.cluster_vars[0]
        assert cluster_var in td.data
        assert td.data[cluster_var].attrs["shift_threshold"] == 4
