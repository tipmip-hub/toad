import numpy as np
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.utils import _attrs


def test_cluster_optimization():
    """Test the cluster optimization."""

    # Setup
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=3, lon=3, boundary="trim").reduce(np.mean)

    # Drop any cluster vars
    td.data = td.data.drop_vars(td.cluster_vars)

    td.compute_clusters(
        optimize=True,
        optimize_params={
            "min_cluster_size": (5, 15),
            "shift_threshold": 0.75,
            "time_weight": (0.5, 2.0),
        },
        method=HDBSCAN,
        shift_selection="local",
        optimize_n_trials=10,
    )

    # Since a short optimization like this won't always converge to the same result, we just check that a new cluster label was added.
    assert len(td.cluster_vars) == 1


def test_fixed_optimize_params_preserved_in_final_clustering():
    """Fixed optimize_params must be applied to the final clustering, not dropped.

    Optuna's study.best_params only includes suggested (ranged) parameters. Fixed
    scalars such as min_cluster_size=10 must still reach the final HDBSCAN call.
    """
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=3, lon=3, boundary="trim").reduce(np.mean)
    td.data = td.data.drop_vars(td.cluster_vars)

    td.compute_clusters(
        optimize=True,
        optimize_params={
            "min_cluster_size": 10,  # fixed
            "shift_threshold": (0.5, 0.95),
            "time_weight": (0.5, 2.0),
        },
        method=HDBSCAN,
        shift_selection="local",
        optimize_n_trials=5,
        optimize_progress_bar=False,
    )

    assert len(td.cluster_vars) == 1
    cluster_var = td.cluster_vars[0]
    attrs = td.data[cluster_var].attrs

    assert attrs["cluster_min_cluster_size"] == "10"
    assert attrs[_attrs.OPT_BEST_PARAMS]["min_cluster_size"] == 10
    assert "shift_threshold" in attrs[_attrs.OPT_BEST_PARAMS]
    assert "time_weight" in attrs[_attrs.OPT_BEST_PARAMS]
