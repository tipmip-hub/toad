import numpy as np
import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.clustering import read_opt_params
from toad.utils import _attrs


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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
    best = read_opt_params(attrs[_attrs.OPT_BEST_PARAMS])
    assert best["min_cluster_size"] == 10
    assert "shift_threshold" in best
    assert "time_weight" in best


# ---------------------------------------------------------------------------
# td.optimize_clusters (audit 2.4, 2.5 and the optimiser refactor)


def _small_global_ds():
    import xarray as xr

    rng = np.random.default_rng(1)
    time = np.arange(20, dtype=float)
    lat = np.arange(-85.0, 90.0, 10.0)
    lon = np.arange(5.0, 360.0, 10.0)
    sh = np.zeros((time.size, lat.size, lon.size))
    sh[5, 10:14, 3:9] = rng.uniform(0.6, 1.0, size=(4, 6))
    sh[12, 4:8, 20:26] = -rng.uniform(0.6, 1.0, size=(4, 6))
    ds = xr.Dataset(
        {"x_dts": (("time", "lat", "lon"), sh)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["x_dts"].attrs[_attrs.BASE_VARIABLE] = "x"
    ds["x_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT
    return ds


def _n_clusters(td, label):
    return float(len(td.get_cluster_ids(label)))


OPT = dict(
    params={"min_cluster_size": (3, 8), "time_weight": (0.5, 1.5)},
    objective=_n_clusters,
    n_trials=4,
    progress_bar=False,
)


def test_deprecated_optimize_warns_and_matches_optimize_clusters(monkeypatch):
    import optuna

    create_study = optuna.create_study

    def seeded_create_study(*args, sampler=None, **kwargs):
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=3)
        return create_study(*args, sampler=sampler, **kwargs)

    monkeypatch.setattr(optuna, "create_study", seeded_create_study)

    td_old = TOAD(_small_global_ds(), log_level="ERROR")
    with pytest.warns(DeprecationWarning, match="td.optimize_clusters"):
        td_old.compute_clusters(
            "x_dts",
            method=HDBSCAN,
            optimize=True,
            optimize_params=OPT["params"],
            optimize_objective=OPT["objective"],
            optimize_n_trials=OPT["n_trials"],
            optimize_progress_bar=False,
        )
    td_new = TOAD(_small_global_ds(), log_level="ERROR")
    td_new.optimize_clusters("x_dts", method=HDBSCAN, **OPT)

    a, b = td_old.data["x_dts_cluster"], td_new.data["x_dts_cluster"]
    np.testing.assert_array_equal(a.values, b.values)
    skip = ("runtime_preprocessing", "runtime_clustering", "runtime_total")
    for k in a.attrs:
        if k not in skip:
            np.testing.assert_equal(a.attrs[k], b.attrs[k])


def test_seed_makes_optimize_clusters_reproducible():
    out = []
    for _ in range(2):
        td = TOAD(_small_global_ds(), log_level="ERROR")
        td.optimize_clusters("x_dts", seed=7, **OPT)
        out.append(td.data["x_dts_cluster"].attrs[_attrs.OPT_BEST_PARAMS])
    assert out[0] == out[1]


def test_optimize_passes_every_argument_to_trials_and_final_fit(monkeypatch):
    import toad.clustering as clustering

    calls = []
    real = clustering._compute_clusters

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(clustering, "_compute_clusters", spy)
    td = TOAD(_small_global_ds(), log_level="ERROR")
    td.optimize_clusters(
        "x_dts",
        disable_regridder=True,
        shift_direction="positive",
        shift_selection="all",
        sort_by_size=False,
        **OPT,
    )
    assert len(calls) == OPT["n_trials"] + 1
    for kw in calls:
        assert kw["disable_regridder"] is True
        assert kw["shift_direction"] == "positive"
        assert kw["shift_selection"] == "all"
        assert kw["sort_by_size"] is False
    attrs = td.data["x_dts_cluster"].attrs
    assert "regridder_name" not in attrs  # not HEALPix-regridded
    assert attrs[_attrs.SHIFT_DIRECTION] == "positive"


def test_optimize_then_save_and_reopen(tmp_path):
    td = TOAD(_small_global_ds(), log_level="ERROR")
    td.optimize_clusters("x_dts", **OPT)
    path = str(tmp_path / "opt.nc")
    td.save(path=path)
    td2 = TOAD(path, log_level="ERROR")
    attrs = td2.data["x_dts_cluster"].attrs
    assert int(attrs[_attrs.OPTIMIZATION]) == 1
    assert read_opt_params(attrs[_attrs.OPT_PARAMS]) == {
        "min_cluster_size": [3, 8],
        "time_weight": [0.5, 1.5],
    }
    best = read_opt_params(attrs[_attrs.OPT_BEST_PARAMS])
    assert set(best) == {"min_cluster_size", "time_weight"}
    np.testing.assert_array_equal(
        td2.data["x_dts_cluster"].values, td.data["x_dts_cluster"].values
    )


def test_read_opt_params_accepts_old_dict_form():
    old = {"min_cluster_size": (3, 8)}
    assert read_opt_params(old) is old
