"""Tests for cluster statistics functions in toad.postprocessing.stats."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.shifts import ASDETECT


@pytest.fixture(scope="module")
def td_with_clusters():
    """Create a TOAD object with clusters for testing."""
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=3, lon=3, boundary="trim").reduce(np.mean)
    td.drop_clusters()
    td.compute_clusters(method=HDBSCAN(min_cluster_size=10))
    return td


class TestTimeStats:
    """Test time-related statistics functions."""

    def test_time_stats_values(self, td_with_clusters):
        """Test that time stats return expected values for first cluster."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]

        time_stats = td.stats(cluster_var).time

        # Test specific values
        assert time_stats.start_timestep(cid) >= 0
        assert time_stats.end_timestep(cid) >= time_stats.start_timestep(cid)
        assert time_stats.duration_timesteps(cid) == (
            time_stats.end_timestep(cid) - time_stats.start_timestep(cid)
        )
        assert time_stats.duration(cid) >= 0
        assert time_stats.std(cid) >= 0
        assert time_stats.membership_peak_density(cid) > 0

    def test_iqr_ordering(self, td_with_clusters):
        """Test that IQR bounds are correctly ordered."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]

        lower, upper = td.stats(cluster_var).time.iqr(cid, 0.25, 0.75)
        assert lower <= upper

    def test_all_stats_completeness(self, td_with_clusters):
        """Test that all_stats returns expected keys."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]

        all_stats = td.stats(cluster_var).time.all_stats(cid)

        expected_keys = {"start", "end", "duration", "mean", "median", "std"}
        assert expected_keys.issubset(all_stats.keys())

    def test_compute_transition_time(self, td_with_clusters):
        """Test compute_transition_time returns valid DataArray."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        shifts_var = td.data[cluster_var].attrs.get("shifts_variable")

        if shifts_var:
            result = td.stats(shifts_var).time.compute_transition_time()
            assert isinstance(result, xr.DataArray)
            assert result.name == "transition_time"
            # Should have same spatial shape as data
            assert result.dims == td.data[shifts_var].dims[1:]

    def test_compute_transition_time_without_clusters(self):
        """Shift-only stats work before compute_clusters (no cluster labels required)."""
        td = TOAD("tutorials/test_data/synth_data.nc")
        td.data = td.data.coarsen(lat=3, lon=3, boundary="trim").reduce(np.mean)
        td.drop_clusters()
        td.compute_shifts(method=ASDETECT(ignore_nan_warnings=True))
        shifts_var = td.shift_vars[0]

        result = td.stats(shifts_var).time.compute_transition_time()
        assert isinstance(result, xr.DataArray)
        assert result.name == "transition_time"
        assert result.dims == td.data[shifts_var].dims[1:]

    def test_activity_vs_pooled_transition_time(self, td_with_clusters):
        """Both time summaries exist; pooled uses per-cell peak-shift times."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]
        time_stats = td.stats(cluster_var).time

        activity = time_stats.median_activity_time(cid)
        pooled = time_stats.pooled_median_transition_time(cid)
        assert activity is not None
        assert np.isfinite(float(pooled)) or isinstance(pooled, (int, float))

        summary = time_stats.summary(cluster_ids=[cid])
        assert list(summary.columns) == [
            "cluster_id",
            "median_activity_time",
            "pooled_median_transition_time",
            "pooled_std_transition_time",
            "n_transition_cells",
            "start",
            "end",
        ]
        assert summary.loc[0, "cluster_id"] == cid
        assert summary.loc[0, "n_transition_cells"] > 0


class TestSpaceStats:
    """Test space-related statistics functions."""

    def test_space_stats_values(self, td_with_clusters):
        """Test that space stats return consistent values."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]

        space_stats = td.stats(cluster_var).space

        # Mean and median should return (lat, lon) tuples
        mean = space_stats.mean(cid)
        median = space_stats.median(cid)
        std = space_stats.std(cid)

        assert len(mean) == 2 and all(np.isfinite(v) for v in mean)
        assert len(median) == 2 and all(np.isfinite(v) for v in median)
        assert len(std) == 2 and all(v >= 0 for v in std)

        # Footprint area should be positive
        area = space_stats.footprint_cumulative_area(cid)
        assert area > 0

    def test_central_point_inside_bounds(self, td_with_clusters):
        """Test that central point is within data bounds."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]

        lat, lon = td.stats(cluster_var).space.central_point_for_labeling(cid)

        # Should be within data coordinate ranges
        lat_range = (float(td.data.lat.min()), float(td.data.lat.max()))
        lon_range = (float(td.data.lon.min()), float(td.data.lon.max()))

        assert lat_range[0] <= lat <= lat_range[1]
        assert lon_range[0] <= lon <= lon_range[1]


class TestScoreUnits:
    """Unit tests for score functions with known inputs/outputs."""

    def test_score_nonlinearity_known_values(self):
        """Test score_nonlinearity with synthetic data of known nonlinearity."""
        from toad.postprocessing.stats.general import GeneralStats

        # Create a mock TOAD-like object with controlled data
        class MockTOAD:
            numeric_time_values = np.arange(100, dtype=float)

            def get_timeseries(
                self, var, cluster_id, aggregation, percentile, normalize
            ):
                # Return a perfect linear trend (should have ~0 nonlinearity)
                return type("DA", (), {"values": np.linspace(0, 1, 100)})()

        stats = GeneralStats(MockTOAD(), "test_var")
        score = stats.score_nonlinearity(cluster_id=0)

        # Perfect linear trend should have very low RMSE (close to 0)
        np.testing.assert_allclose(score, 0.0, atol=1e-10)

    def test_score_nonlinearity_step_function(self):
        """Test that a step function has higher nonlinearity than linear."""
        from toad.postprocessing.stats.general import GeneralStats

        class MockTOAD:
            numeric_time_values = np.arange(100, dtype=float)

            def get_timeseries(
                self, var, cluster_id, aggregation, percentile, normalize
            ):
                # Step function at midpoint
                data = np.zeros(100)
                data[50:] = 1.0
                return type("DA", (), {"values": data})()

        stats = GeneralStats(MockTOAD(), "test_var")
        score = stats.score_nonlinearity(cluster_id=0)

        # Step function deviates significantly from linear fit
        assert score > 0.1  # Should have meaningful nonlinearity

    def test_score_spatial_autocorrelation_identical_series(self):
        """Test that identical time series have R²=1."""
        from toad.postprocessing.stats.general import GeneralStats

        class MockTOAD:
            def get_timeseries(self, var, cluster_id):
                # Return 5 identical time series
                data = np.tile(np.sin(np.linspace(0, 2 * np.pi, 50)), (5, 1))
                return data

        stats = GeneralStats(MockTOAD(), "test_var")
        score = stats.score_spatial_autocorrelation(cluster_id=0)

        # Identical series should have perfect correlation
        np.testing.assert_allclose(score, 1.0, atol=1e-10)


class TestGeneralStats:
    """Test scoring functions with specific expected values."""

    def test_score_values_in_valid_ranges(self, td_with_clusters):
        """Test that all scores return values in expected ranges."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]

        general = td.stats(cluster_var).general

        # Nonlinearity: RMSE, should be non-negative
        nonlinearity = general.score_nonlinearity(cid)
        assert nonlinearity >= 0

        # Heaviside: standardized score, should be non-negative
        heaviside = general.score_heaviside(cid)
        assert heaviside >= 0

        # Consistency: non-negative (0-1 typical, can exceed 1 for very consistent)
        consistency = general.score_consistency(cid)
        assert consistency >= 0

        # Spatial autocorrelation: R², should be 0-1
        spatial_autocorr = general.score_spatial_autocorrelation(cid)
        assert 0 <= spatial_autocorr <= 1

    def test_score_heaviside_with_fit(self, td_with_clusters):
        """Test that score_heaviside returns fit array when requested."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]

        score, fit = td.stats(cluster_var).general.score_heaviside(
            cid, return_score_fit=True
        )

        assert isinstance(score, float)
        assert isinstance(fit, np.ndarray)
        assert len(fit) == len(td.data.time)

    def test_score_overview_structure(self, td_with_clusters):
        """Test that score_overview returns complete DataFrame."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]

        df = td.stats(cluster_var).general.score_overview()
        cluster_ids = td.get_cluster_ids(cluster_var)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(cluster_ids)

        # Check required columns exist
        required_cols = {
            "cluster_id",
            "heaviside",
            "consistency",
            "spatial_autocorrelation",
            "nonlinearity",
            "size",
            "aggregate_score",
        }
        assert required_cols.issubset(df.columns)

        # All cluster_ids should be present
        assert set(df["cluster_id"]) == set(cluster_ids)


class TestClusterSummary:
    """Test cluster overview summary table."""

    def test_cluster_summary_minimal(self, td_with_clusters):
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cluster_ids = td.get_cluster_ids(cluster_var)

        df_stats = td.stats(cluster_var).cluster_summary()
        df_alias = td.cluster_summary(cluster_var)

        assert isinstance(df_stats, pd.DataFrame)
        pd.testing.assert_frame_equal(df_stats, df_alias)

        assert len(df_stats) == len(cluster_ids)
        assert set(df_stats["cluster_id"]) == set(cluster_ids)

        required_cols = {
            "cluster_id",
            "start_time",
            "end_time",
            "duration_timesteps",
            "size",
            "footprint_area",
            "median_time",
            "center_lat",
            "center_lon",
        }
        assert required_cols.issubset(df_stats.columns)

        row = df_stats.loc[df_stats["cluster_id"] == cluster_ids[0]].iloc[0]
        assert row["size"] > 0
        assert row["footprint_area"] > 0
        assert row["duration_timesteps"] >= 0

    def test_cluster_summary_extended(self, td_with_clusters):
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cid = td.get_cluster_ids(cluster_var)[0]

        df = td.cluster_summary(cluster_var, extended=True, cluster_ids=[cid])

        extended_cols = {
            "iqr_68_start",
            "iqr_68_end",
            "avg_amplitude",
            "max_amplitude",
            "pooled_median_transition_time",
            "pooled_std_transition_time",
            "n_transition_cells",
            "variable",
            "method",
            "shift_threshold",
        }
        assert extended_cols.issubset(df.columns)
        assert df.loc[0, "cluster_id"] == cid
        assert df.loc[0, "n_transition_cells"] > 0
        assert df.loc[0, "method"] == "HDBSCAN"


class TestSmartInference:
    """Test that cluster var vs base var give identical results."""

    def test_scores_identical_for_base_and_cluster_var(self, td_with_clusters):
        """Test smart inference produces identical scores."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        base_var = "ts"
        cid = td.get_cluster_ids(cluster_var)[0]

        # All score functions should give identical results
        for method, kwargs in [
            ("score_nonlinearity", {"aggregation": "mean"}),
            ("score_heaviside", {"aggregation": "mean"}),
            ("score_consistency", {}),
            ("score_spatial_autocorrelation", {}),
        ]:
            score_base = getattr(td.stats(base_var).general, method)(cid, **kwargs)
            score_cluster = getattr(td.stats(cluster_var).general, method)(
                cid, **kwargs
            )

            np.testing.assert_allclose(
                score_base,
                score_cluster,
                rtol=1e-10,
                err_msg=f"{method}: base={score_base}, cluster={score_cluster}",
            )


class TestMultipleClusterVariables:
    """Test handling of multiple cluster variables."""

    def test_scores_work_with_multiple_cluster_vars(self):
        """Test scoring works when multiple cluster variables exist."""
        # Use fresh TOAD instance to avoid mutating shared fixture
        td = TOAD("tutorials/test_data/synth_data.nc")
        td.data = td.data.coarsen(lat=3, lon=3, boundary="trim").reduce(np.mean)
        td.drop_clusters()

        # Create two cluster variables
        td.compute_clusters(method=HDBSCAN(min_cluster_size=10))
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=5),
            output_label="second_cluster",
        )

        assert len(td.cluster_vars) == 2

        # Both should work independently
        for cluster_var in td.cluster_vars:
            cluster_ids = td.get_cluster_ids(cluster_var)
            if len(cluster_ids) > 0:
                score = td.stats(cluster_var).general.score_nonlinearity(cluster_ids[0])
                assert np.isfinite(score)
