"""Tests for basic plotting functions in toad.plotting."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD

# Use non-interactive backend for tests
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def td_with_clusters():
    """Create a TOAD object with clusters for testing plotting functions.

    Uses a very coarse dataset to keep tests fast.
    """
    td = TOAD("tutorials/test_data/synth_data.nc")
    # Make data very coarse for fast tests
    td.data = td.data.coarsen(lat=5, lon=5, boundary="trim").reduce(np.mean)
    td.drop_clusters()
    td.compute_clusters(method=HDBSCAN(min_cluster_size=5))
    return td


class TestClusterMap:
    """Test cluster_map plotting function."""

    def test_cluster_map_returns_figure_and_axes(self, td_with_clusters):
        """Test that cluster_map returns a figure and axes."""
        td = td_with_clusters
        fig, ax = td.plot.cluster_map()

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_cluster_map_single_cluster(self, td_with_clusters):
        """Test cluster_map with a single cluster ID."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            # Convert to Python int to avoid numpy.int64 type issues
            fig, ax = td.plot.cluster_map(cluster_ids=int(cluster_ids[0]))
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_cluster_map_multiple_clusters(self, td_with_clusters):
        """Test cluster_map with multiple cluster IDs."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) >= 2:
            fig, ax = td.plot.cluster_map(cluster_ids=list(cluster_ids[:2]))
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_cluster_map_with_subplots(self, td_with_clusters):
        """Test cluster_map with subplots=True."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) >= 2:
            fig, axes = td.plot.cluster_map(
                cluster_ids=list(cluster_ids[:2]), subplots=True
            )
            assert fig is not None
            assert axes is not None
            assert isinstance(axes, np.ndarray)
            plt.close(fig)

    def test_cluster_map_with_custom_cmap(self, td_with_clusters):
        """Test cluster_map with a custom colormap."""
        td = td_with_clusters
        fig, ax = td.plot.cluster_map(cmap="viridis")

        assert fig is not None
        plt.close(fig)

    def test_cluster_map_with_provided_geoax(self, td_with_clusters):
        """Test cluster_map with a pre-created GeoAxes."""
        import cartopy.crs as ccrs

        td = td_with_clusters
        fig_external, ax_external = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        fig, ax = td.plot.cluster_map(ax=ax_external)

        # When ax is provided, fig should be None
        assert fig is None
        assert ax is ax_external
        plt.close(fig_external)


class TestTimeseries:
    """Test timeseries plotting function."""

    def test_timeseries_returns_figure_and_axes(self, td_with_clusters):
        """Test that timeseries returns a figure and axes."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            # Convert to Python int to avoid numpy.int64 type issues
            fig, ax = td.plot.timeseries(cluster_ids=int(cluster_ids[0]))
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_timeseries_single_cluster(self, td_with_clusters):
        """Test timeseries with a single cluster ID."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            fig, ax = td.plot.timeseries(cluster_ids=int(cluster_ids[0]))
            assert fig is not None
            plt.close(fig)

    def test_timeseries_multiple_clusters(self, td_with_clusters):
        """Test timeseries with multiple cluster IDs."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) >= 2:
            fig, ax = td.plot.timeseries(cluster_ids=list(cluster_ids[:2]))
            assert fig is not None
            plt.close(fig)

    def test_timeseries_with_subplots(self, td_with_clusters):
        """Test timeseries with subplots=True."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) >= 2:
            fig, axes = td.plot.timeseries(
                cluster_ids=list(cluster_ids[:2]), subplots=True
            )
            assert fig is not None
            assert axes is not None
            plt.close(fig)

    def test_timeseries_with_median(self, td_with_clusters):
        """Test timeseries with median line."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            fig, ax = td.plot.timeseries(
                cluster_ids=int(cluster_ids[0]),
                plot_median=True,
                plot_trajectories=False,
            )
            assert fig is not None
            plt.close(fig)

    def test_timeseries_with_trajectory_range(self, td_with_clusters):
        """Test timeseries with trajectory range shading."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            fig, ax = td.plot.timeseries(
                cluster_ids=int(cluster_ids[0]),
                plot_trajectory_range=True,
            )
            assert fig is not None
            plt.close(fig)

    def test_timeseries_with_map(self, td_with_clusters):
        """Test timeseries with embedded map."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            fig, result = td.plot.timeseries(
                cluster_ids=int(cluster_ids[0]),
                plot_map=True,
            )
            assert fig is not None
            plt.close(fig)

    def test_timeseries_shared_ylabel(self, td_with_clusters):
        """Test shared y-label beside timeseries subplots."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) >= 2:
            fig, result = td.plot.timeseries(
                cluster_ids=list(cluster_ids[:2]),
                plot_map=True,
                shared_ylabel="Test quantity (units)",
            )
            assert fig is not None
            label_texts = [
                t.get_text()
                for t in fig.texts
                if t.get_text() == "Test quantity (units)"
            ]
            assert len(label_texts) == 1
            plt.close(fig)

    def test_timeseries_with_provided_ax(self, td_with_clusters):
        """Test timeseries with a pre-created axes."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            fig_external, ax_external = plt.subplots()

            fig, ax = td.plot.timeseries(
                cluster_ids=int(cluster_ids[0]), ax=ax_external
            )

            # When ax is provided, fig should be None
            assert fig is None
            assert ax is ax_external
            plt.close(fig_external)


class TestOverview:
    """Test overview plotting function."""

    def test_overview_returns_figure_and_dict(self, td_with_clusters):
        """Test that overview returns a figure and dict with axes."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            fig, axes_dict = td.plot.overview(cluster_ids=list(cluster_ids[:1]))

            assert fig is not None
            assert isinstance(axes_dict, dict)
            assert "map" in axes_dict
            assert "timeseries" in axes_dict
            plt.close(fig)

    def test_overview_single_cluster(self, td_with_clusters):
        """Test overview with a single cluster."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            # Convert to Python int to avoid numpy.int64 type issues
            fig, axes_dict = td.plot.overview(cluster_ids=int(cluster_ids[0]))
            assert fig is not None
            assert axes_dict["map"] is not None
            plt.close(fig)

    def test_overview_multiple_clusters(self, td_with_clusters):
        """Test overview with multiple clusters."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) >= 2:
            fig, axes_dict = td.plot.overview(cluster_ids=list(cluster_ids[:2]))
            assert fig is not None
            assert "timeseries" in axes_dict
            plt.close(fig)

    def test_overview_timeseries_mode(self, td_with_clusters):
        """Test overview in timeseries mode (default)."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            fig, axes_dict = td.plot.overview(
                cluster_ids=list(cluster_ids[:1]),
                mode="timeseries",
            )
            assert fig is not None
            plt.close(fig)

    def test_overview_aggregated_mode(self, td_with_clusters):
        """Test overview in aggregated mode."""
        td = td_with_clusters
        cluster_ids = td.get_cluster_ids(td.cluster_vars[0])

        if len(cluster_ids) > 0:
            fig, axes_dict = td.plot.overview(
                cluster_ids=list(cluster_ids[:1]),
                mode="aggregated",
            )
            assert fig is not None
            plt.close(fig)

    def test_overview_no_clusters_returns_figure(self, td_with_clusters):
        """Noise-only / empty cluster sets still get a map + placeholder panel."""
        from toad.utils import _attrs

        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        original_ids = td.data[cluster_var].attrs[_attrs.CLUSTER_IDS]
        td.data[cluster_var].attrs[_attrs.CLUSTER_IDS] = np.array([-1], dtype=np.int64)
        try:
            fig, axes_dict = td.plot.overview(
                cluster_ids=range(6),
                mode="aggregated",
                figsize=(8, 6),
            )
            assert fig is not None
            assert isinstance(axes_dict, dict)
            assert axes_dict["map"] is not None
            assert axes_dict["timeseries"] is not None
            texts = [t.get_text() for t in axes_dict["timeseries"][0].texts]
            assert "No clusters identified" in texts
            plt.close(fig)
        finally:
            td.data[cluster_var].attrs[_attrs.CLUSTER_IDS] = original_ids


class TestPlottingWithExplicitVar:
    """Test plotting functions with explicit variable specification."""

    def test_cluster_map_with_var(self, td_with_clusters):
        """Test cluster_map with explicit variable name."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]

        fig, ax = td.plot.cluster_map(var=cluster_var)
        assert fig is not None
        plt.close(fig)

    def test_timeseries_with_var(self, td_with_clusters):
        """Test timeseries with explicit variable name."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cluster_ids = td.get_cluster_ids(cluster_var)

        if len(cluster_ids) > 0:
            # Convert to Python int to avoid numpy.int64 type issues
            fig, ax = td.plot.timeseries(
                var=cluster_var, cluster_ids=int(cluster_ids[0])
            )
            assert fig is not None
            plt.close(fig)

    def test_timeseries_shift_window_without_cluster_ids(self, td_with_clusters):
        """Shift window applies when plotting all data without cluster_ids."""
        td = td_with_clusters
        var = td.base_vars[0]
        expected = td.get_timeseries(
            var=var,
            cluster_id=None,
            aggregation="raw",
            timeseries_window="shift",
            normalize="max_each",
        )

        fig, ax = td.plot.timeseries(
            timeseries_window="shift",
            normalize="max_each",
            plot_cluster_duration=False,
            max_trajectories=len(expected),
            trajectory_ids=list(range(len(expected))),
        )

        plotted = np.stack([line.get_ydata() for line in ax.get_lines()], axis=0)
        assert plotted.shape == expected.values.shape
        assert np.allclose(plotted, expected.values, equal_nan=True)
        plt.close(fig)

    def test_overview_with_var(self, td_with_clusters):
        """Test overview with explicit variable name."""
        td = td_with_clusters
        cluster_var = td.cluster_vars[0]
        cluster_ids = td.get_cluster_ids(cluster_var)

        if len(cluster_ids) > 0:
            # Convert to Python int to avoid numpy.int64 type issues
            fig, axes_dict = td.plot.overview(
                var=cluster_var, cluster_ids=int(cluster_ids[0])
            )
            assert fig is not None
            plt.close(fig)


def test_member_id_from_cluster_var():
    from toad.plotting import (
        _input_cluster_legend_label,
        _member_id_from_cluster_var,
        _realisation_from_cluster_var,
    )

    assert _member_id_from_cluster_var("mlotst_r1i1p1f1_dts_cluster") == "r1i1p1f1"
    assert _member_id_from_cluster_var("foo_r2_cluster") == "r2"
    assert _realisation_from_cluster_var("mlotst_r3i1p1f1_dts_cluster") == "r3"
    assert _realisation_from_cluster_var("foo_r2_cluster") == "r2"
    assert (
        _input_cluster_legend_label(
            "mlotst_r3i1p1f1_dts_cluster",
            n_cells=4,
            label_style="member_id",
            include_n_cells=False,
        )
        == "r3i1p1f1"
    )
    assert (
        _input_cluster_legend_label(
            "mlotst_r5i1p1f1_dts_cluster",
            n_cells=12,
            label_style="realisation",
            include_n_cells=True,
        )
        == "(12) r5"
    )
    assert (
        _input_cluster_legend_label(
            "foo_r1_cluster",
            n_cells=2,
            label_style="cluster_var",
            include_n_cells=True,
        )
        == "(2) foo_r1_cluster"
    )


def test_plotting_with_scalar_cluster_ids_attr(td_with_clusters):
    """NetCDF round-trip can store a single cluster id as a scalar attr."""
    from toad.utils import _attrs

    td = td_with_clusters
    cluster_var = td.cluster_vars[0]
    original_ids = td.data[cluster_var].attrs[_attrs.CLUSTER_IDS]
    td.data[cluster_var].attrs[_attrs.CLUSTER_IDS] = np.int64(0)
    try:
        ids = td.get_cluster_ids(cluster_var, exclude_noise=False)
        assert ids.shape == (1,)
        assert int(ids[0]) == 0

        fig, _ = td.plot.overview(cluster_ids=range(3), mode="aggregated")
        assert fig is not None
        plt.close(fig)
    finally:
        td.data[cluster_var].attrs[_attrs.CLUSTER_IDS] = original_ids


def test_repr_html_with_scalar_noise_only_cluster_ids(td_with_clusters):
    """Noise-only exports often round-trip ``cluster_ids=-1`` as a scalar."""
    from toad.utils import _attrs

    td = td_with_clusters
    cluster_var = td.cluster_vars[0]
    original_ids = td.data[cluster_var].attrs[_attrs.CLUSTER_IDS]
    td.data[cluster_var].attrs[_attrs.CLUSTER_IDS] = np.int64(-1)
    try:
        html = td._repr_html_()
        assert isinstance(html, str)
        assert "TOAD" in html or "toad" in html.lower() or cluster_var in html
    finally:
        td.data[cluster_var].attrs[_attrs.CLUSTER_IDS] = original_ids
