import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import optuna
import xarray as xr
from sklearn.base import ClusterMixin
from sklearn.cluster import HDBSCAN

import toad.clustering as clustering
from toad.utils import DEFAULT_SHIFT_THRESHOLD, _attrs

if TYPE_CHECKING:
    from toad.core import TOAD
    from toad.regridding import HealPixRegridder

logger = logging.getLogger("TOAD")


def combined_spatial_nonlinearity(td, cluster_variable, weights=[1, 1]) -> float:
    """Compute a weighted combination of spatial autocorrelation and nonlinearity scores for the largest 10 clusters.

    Args:
        td: TOAD object containing the data
        cluster_variable: Name of the cluster variable to analyze
        weights: List of two weights for spatial and nonlinearity scores. Defaults to [1, 1].

    Returns:
        float: Weighted sum of spatial autocorrelation and nonlinearity scores
    """

    cluster_ids = td.get_cluster_ids(cluster_variable)
    score1 = td.stats(cluster_variable).general.aggregate_cluster_scores(
        cluster_ids=cluster_ids[:10],
        score_method="score_spatial_autocorrelation",
        aggregation="median",
    )

    score2 = td.stats(cluster_variable).general.aggregate_cluster_scores(
        cluster_ids=cluster_ids[:10],
        score_method="score_nonlinearity",
        aggregation="median",
    )

    return float(weights[0] * score1 + weights[1] * score2)


default_opt_params = dict(
    {
        "min_cluster_size": (10, 25),
        "time_weight": (0.5, 1.5),
    }
)

"""
Cluster Scoring Methods
    Heaviside Score (score_heaviside):
    - Residual of a straight-line fit to the cluster's aggregated time series, relative to its spread
    - 0 = linear trend, 0.25 = step in the middle of the series, about 1 = noise
    - A high value does not by itself indicate a step

    Consistency Score (score_consistency):
    - Measures internal coherence using hierarchical clustering of time series correlations
    - Converts R² correlations to distances, performs Ward linkage
    - Higher scores indicate more consistent temporal behavior within clusters

    Spatial Autocorrelation Score (score_spatial_autocorrelation):
    - Computes average pairwise R² correlations between all time series in a cluster
    - Measures spatial coherence and synchronization of cluster behavior

    Nonlinearity Score (score_nonlinearity):
    - Measures RMSE deviation from linear fit
    - Optional normalization against unclustered data to identify clusters that stand out
    - Supports various spatial aggregation methods (mean, median, percentile, etc.)

    You can also define your own objective function by passing a function which takes (td, cluster_variable) as arguments.
    Then you just compute your score and return it. See `combined_spatial_nonlinearity()` above for an example. 
"""

"""
Little note about parameters:
- We have parameters that go directly into the clustering method. These must be very flexible because the clustering method can be anything.
- We also have parameters that modify how TOAD applies the clustering method. These are specific.
- The user however passes both in the same dictionary of optimize_params.
- So we need to be careful when extracting and leading the params to the right places. 
"""


def optimize_clusters(
    td: "TOAD",
    var: str,
    method: ClusterMixin | type = HDBSCAN,
    shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    shift_direction: Literal["both", "positive", "negative"] | str = "both",
    shift_selection: Literal["local", "global", "all"] | str = "local",
    time_weight: float = 1,
    regridder: "HealPixRegridder | None" = None,
    disable_regridder: bool = False,
    output_label_suffix: str = "",
    output_label: str | None = None,
    overwrite: bool = False,
    sort_by_size: bool = True,
    params: dict = default_opt_params,
    objective: Callable
    | Literal[
        "median_heaviside",
        "mean_heaviside",
        "mean_consistency",
        "mean_spatial_autocorrelation",
        "mean_nonlinearity",
        "combined_spatial_nonlinearity",
    ]
    | str = "combined_spatial_nonlinearity",
    n_trials: int = 50,
    direction: str = "maximize",
    log_level: int = optuna.logging.WARNING,
    progress_bar: bool = True,
    seed: int | None = None,
) -> xr.Dataset:
    """Search clustering parameters with Optuna, then cluster with the best ones (experimental).

    Every trial is an ordinary ``compute_clusters`` call with the arguments given
    here, plus the parameters sampled from ``params``. The final clustering uses
    the best parameters and the same arguments, and is written to one cluster
    variable with the optimisation details in its attributes.

    Args:
        td: TOAD object containing the data to cluster.
        var, method, shift_threshold, shift_direction, shift_selection, time_weight,
            regridder, disable_regridder, output_label_suffix, output_label,
            overwrite, sort_by_size: As in ``compute_clusters``. Parameters of a
            method instance (e.g. ``HDBSCAN(min_cluster_size=10)``) are ignored;
            pass them through ``params`` instead.
        params: Parameters to optimise. A ``(min, max)`` tuple is sampled (float
            if the first bound is a float, else int); a single value is fixed.
            Keys are arguments of the clustering method, or ``shift_threshold``
            and ``time_weight``. Defaults to ``default_opt_params``.
        objective: The score to optimise. Defaults to "combined_spatial_nonlinearity". Can be one of:
            - callable: Custom objective function taking (td, output_label) as arguments
            - "median_heaviside": Median heaviside score across clusters
            - "mean_heaviside": Mean heaviside score across clusters
            - "mean_consistency": Mean consistency score across clusters
            - "mean_spatial_autocorrelation": Mean spatial autocorrelation score
            - "mean_nonlinearity": Mean nonlinearity score across clusters
            - "combined_spatial_nonlinearity": See ``combined_spatial_nonlinearity``
        n_trials: Number of trials. Defaults to 50.
        direction: "maximize" or "minimize". Defaults to "maximize".
        log_level: Optuna log level. Defaults to optuna.logging.WARNING.
        progress_bar: Whether to show Optuna's progress bar. Defaults to True.
        seed: Seed for Optuna's sampler, for reproducible trials. Defaults to None.

    Returns:
        xr.Dataset: The data with the cluster variable of the best parameters.
    """
    # Resolve the shifts variable and the output label once, as compute_clusters
    # would; every trial and the final fit then write to that label.
    shifts_variable = clustering._resolve_shifts_variable(td, var)
    clustering._check_args(shift_threshold, regridder)
    output_label = clustering._resolve_output_label(
        td, shifts_variable, output_label, output_label_suffix, overwrite
    )

    # Arguments shared by every trial and the final fit.
    kwargs = dict(
        shift_direction=shift_direction,
        shift_selection=shift_selection,
        regridder=regridder,
        disable_regridder=disable_regridder,
        output_label=output_label,
        overwrite=True,
        sort_by_size=sort_by_size,
    )
    opt_params = params
    show_progress_bar = progress_bar

    # ==================== VALIDATE INPUT ====================

    # Validate/correct method
    if not isinstance(method, type):
        # If it's an instance, get its class: i.e. convert HDBCSAN() -> HDBSCAN
        # This is ok if user didn't change default params
        method_class = method.__class__

        # Try to check if params have been changed
        warn_user_about_params = False
        try:
            warn_user_about_params = method.__dict__ != method.__class__().__dict__
        except TypeError:
            # Crashes if method has required parameters, so we can't see if params have been changed
            warn_user_about_params = True
        finally:
            if warn_user_about_params:
                logger.warning(
                    "When optimizing, params passed through the clustering method (e.g. HDBSCAN(min_cluster_size=10)) will be ignored."
                    "\nPlease pass params through `params` instead."
                    "\nExample: params={'min_cluster_size': 10} for a fixed min_cluster_size of 10."
                )
    else:
        method_class = method

    if not issubclass(method_class, ClusterMixin):
        raise ValueError(
            "Method must be a clustering algorithm, extending ClusterMixin."
        )

    if not isinstance(opt_params, dict):
        raise TypeError("params must be a dict")
    if len(opt_params) == 0:
        raise ValueError(
            "params cannot be empty. Example: params={'min_cluster_size': (5, 15)}"
        )

    # Print optimization params
    logger.info(f"optimizing {n_trials} trials with params: {opt_params}")

    score_computation_time = 0.0

    def objective_fn(trial) -> float:
        nonlocal score_computation_time

        # Sample optimization params: these may contains both TOAD and Clustering params (see note at the top).
        cluster_params = _sample_params(trial, opt_params)

        # Get time_weight if present, if not use the one from the kwargs
        sample_time_weight = cluster_params.pop("time_weight", time_weight)

        # Get shift_threshold if present, if not use the one from the kwargs
        sample_shift_threshold = cluster_params.pop("shift_threshold", shift_threshold)

        # Compute clusters
        td.data = clustering._compute_clusters(
            td,
            shifts_variable,
            method=method_class(**cluster_params),
            shift_threshold=sample_shift_threshold,
            time_weight=sample_time_weight,
            **kwargs,
        )

        # If no clusters found, return -inf
        cluster_ids = td.get_cluster_ids(output_label)
        if len(cluster_ids) == 0:
            return -np.inf if direction == "maximize" else np.inf

        # Compute score
        time_start = time.time()
        if callable(objective):
            score = float(objective(td, output_label))  # type: ignore
        else:
            # fmt: off
            # Map objective names to their corresponding scoring functions
            # Note: all the score functions are really slow and take ~90% of the optimization time
            # We pass output_label (cluster variable) to stats() - the smart inference in
            # get_cluster_timeseries will extract the base variable for actual data values.
            score_funcs = {
                "median_heaviside":           lambda: np.median([td.stats(output_label).general.score_heaviside(cid, aggregation="median") for cid in cluster_ids[:10]]),
                "mean_heaviside":               lambda: np.mean([td.stats(output_label).general.score_heaviside(cid, aggregation="mean") for cid in cluster_ids[:10]]),
                "mean_consistency":             lambda: np.mean([td.stats(output_label).general.score_consistency(cid) for cid in cluster_ids[:10]]),
                "mean_spatial_autocorrelation": lambda: np.mean([td.stats(output_label).general.score_spatial_autocorrelation(cid) for cid in cluster_ids[:10]]),
                "mean_nonlinearity":            lambda: np.mean([td.stats(output_label).general.score_nonlinearity(cid, aggregation="mean") for cid in cluster_ids[:10]]),
                "combined_spatial_nonlinearity": lambda: combined_spatial_nonlinearity(td, output_label)
            }
            # fmt: on

            if objective not in score_funcs:
                raise ValueError("Invalid objective")
            score = float(score_funcs[objective]())
        score_computation_time += time.time() - time_start
        return score

    optuna.logging.set_verbosity(log_level)
    toad_log_level = td.logger.level  # this gives an int
    td.set_log_level("WARNING")  # this fetches int of flag warning sets that
    try:
        t0 = time.time()
        sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            show_progress_bar=show_progress_bar,
        )
        t1 = time.time()
    finally:
        # restore toad log level in finally block, in case of error.
        td.logger.setLevel(toad_log_level)  # this sets int

    # Optuna only tracks suggested (ranged) params; re-attach fixed ones from
    # optimize_params so the final clustering matches what was used in trials.
    best_params = {**_fixed_opt_params(opt_params), **study.best_params}

    # print final score and best params
    logger.info(
        f"Ran {n_trials} trials in {t1 - t0:.2f} seconds. "
        f"Best (#{study.best_trial.number}): score {study.best_value:.4f}, "
        f"params {best_params}. "
        # f"Score computation time: {score_computation_time:.2f} seconds." # score computation is slow...
    )

    # pop shift_threshold and time_weight if present, if not use the one from the kwargs
    best_params_for_attrs = best_params.copy()
    best_shift_threshold = best_params.pop("shift_threshold", shift_threshold)
    best_time_weight = best_params.pop("time_weight", time_weight)
    new_data = clustering._compute_clusters(
        td,
        shifts_variable,
        method=method_class(**best_params),
        shift_threshold=best_shift_threshold,
        time_weight=best_time_weight,
        **kwargs,
    )

    # Add optimization details to attributes, in types netCDF can store: the
    # flag as int, the parameter dicts as JSON strings (see read_opt_params).
    new_data[output_label].attrs.update(
        {
            _attrs.OPTIMIZATION: 1,
            _attrs.OPT_OBJECTIVE: getattr(objective, "__name__", repr(objective))
            if callable(objective)
            else objective,
            _attrs.OPT_BEST_SCORE: study.best_value,
            _attrs.OPT_DIRECTION: direction,
            _attrs.OPT_PARAMS: json.dumps(opt_params, default=str),
            _attrs.OPT_BEST_PARAMS: json.dumps(best_params_for_attrs, default=str),
            _attrs.OPT_N_TRIALS: n_trials,
        }
    )

    return new_data


def read_opt_params(value) -> dict:
    """Decode an ``opt_params`` or ``opt_best_params`` attribute to a dict.

    Accepts the JSON string written now and the dict written by older versions.
    Ranges come back as lists.
    """
    if isinstance(value, dict):
        return value
    return json.loads(value)


def _fixed_opt_params(param_ranges: dict) -> dict:
    """Return parameters that are fixed (not ranges) in optimize_params.

    Optuna's ``study.best_params`` only includes values registered via
    ``trial.suggest_*``. Fixed scalars must be merged back in separately.
    """
    return {k: v for k, v in param_ranges.items() if not isinstance(v, (list, tuple))}


def _sample_params(trial, param_ranges):
    """Sample parameters for optimization trial based on provided ranges.

    Args:
        trial: An Optuna trial object used to sample parameter values.
        param_ranges: Dictionary mapping parameter names to either single values or
            (min, max) tuples. For tuples, the type of the first value determines
            whether to sample integers or floats.

    Returns:
        dict: Dictionary mapping parameter names to their sampled values. Single
            values are used directly, while ranges are sampled as either floats or
            integers based on the type of the range bounds.
    """
    params = {}
    for k, v in param_ranges.items():
        if not isinstance(v, (list, tuple)):
            # If v is a single value, use it directly
            params[k] = v
        else:
            # If v is a range, suggest float or int based on type
            params[k] = (
                trial.suggest_float(k, *v)
                if isinstance(v[0], float)
                else trial.suggest_int(k, *v)
            )
    return params
