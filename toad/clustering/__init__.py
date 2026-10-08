"""
Clustering module for TOAD (Temporal Offset Analysis and Detection).

This module provides functionality for clustering temporal shifts in climate data. The main function
`compute_clusters` takes temporal shift patterns and groups them into clusters using sklearn-compatible
clustering algorithms. The clustering is performed in both space and time dimensions, allowing
identification of regions with similar temporal shift behaviors.

The module supports various clustering methods from scikit-learn (e.g., HDBSCAN, DBSCAN, etc.) and
includes utilities for:
- Preprocessing data with different scaling methods
- Handling geographic coordinates and projections
- Converting between geodetic and cartesian coordinates
- Sorting clusters by size
- Preserving metadata and attributes in xarray objects
- Filtering shifts based on thresholds and directions
- Selecting between local and global shift patterns

The clustering results are returned as xarray objects with appropriate metadata and can be
visualized using TOAD's plotting utilities.
"""

import logging
from collections.abc import Callable
from time import time as time_now
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import optuna
import xarray as xr
from sklearn.base import ClusterMixin

from toad._version import __version__
from toad.clustering.methods.space_time_dbscan import SpaceTimeDBSCAN
from toad.clustering.optimizing import (
    _optimize_clusters,
    combined_spatial_nonlinearity,
    default_opt_params,
)
from toad.regridding import HealPixRegridder
from toad.regridding.base import BaseRegridder
from toad.utils import (
    DEFAULT_SHIFT_THRESHOLD,
    _attrs,
    _reorder_space_dims,
    get_latlon_info,
    get_unique_variable_name,
)
from toad.utils.shift_selection_utils import (
    _compute_dts_peak_sign_mask,
    _compute_episode_pass_mask,
)

logger = logging.getLogger("TOAD")

__all__ = [
    "compute_clusters",
    "default_opt_params",
    "combined_spatial_nonlinearity",
    "sorted_cluster_labels",
    "SpaceTimeDBSCAN",
]

# to avoid circular import we use TYPE_CHECKING for importing TOAD obj
if TYPE_CHECKING:
    from toad.core import TOAD


def compute_clusters(
    td: "TOAD",
    var: str,
    method: ClusterMixin | type,
    shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    shift_direction: Literal["both", "positive", "negative"] | str = "both",
    shift_selection: Literal["local", "global", "all"] | str = "local",
    min_event_magnitude: float | None = None,
    min_event_magnitude_L: int = 25,
    min_event_magnitude_G: int = 10,
    time_weight: float = 1,
    regridder: BaseRegridder | None = None,
    disable_regridder: bool = False,
    output_label_suffix: str = "",
    output_label: str | None = None,
    overwrite: bool = False,
    sort_by_size: bool = True,
    export_for_mma: str | None = None,
    mma_grid: Literal["healpix", "native"] = "healpix",
    # optimization params
    optimize: bool = False,
    optimize_params: dict = default_opt_params,
    optimize_objective: Callable
    | Literal[
        "median_heaviside",
        "mean_heaviside",
        "mean_consistency",
        "mean_spatial_autocorrelation",
        "mean_nonlinearity",
        "combined_spatial_nonlinearity",
    ]
    | str = "combined_spatial_nonlinearity",
    optimize_n_trials: int = 50,
    optimize_direction: str = "maximize",
    optimize_log_level: int = optuna.logging.WARNING,
    optimize_progress_bar: bool = True,
) -> xr.Dataset:
    """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm.

    Args:
        td: TOAD object containing the data to cluster
        var: Name of the base variable or shifts variable to compute clusters for. If multiple shifts variables exist for the base variable, a ValueError is thrown, in which case you should specify the shifts variable name.
        method: The clustering method to use. Choose methods from `sklearn.cluster` or create your own by inheriting from `sklearn.base.ClusterMixin`.
        shift_threshold: The minimum magnitude a shift must reach to be included in clustering. Raising this threshold filters out less significant shifts and helps focus clustering on the most meaningful events, while reducing it will include more subtle (and potentially noisier) shifts. Default is 0.5, which effectively excludes most noise when using ASDETECT.
        shift_direction: The direction of the shift. Options are "both", "positive", "negative".
            When "both", positive and negative shifts are clustered separately and merged
            into one output variable so that no cluster contains mixed signs. Defaults to "both".
        shift_selection: How shift values are selected for clustering. All options respect shift_threshold and shift_direction:
            - "local": Finds peaks within individual shift episodes. Cluster only local maxima within each contiguous segment where abs(shift) > shift_threshold.
            - "global": Finds the overall strongest shift per grid cell. Cluster only the single maximum shift value per grid cell where abs(shift) > shift_threshold.
            - "all": Cluster all shift values that meet the threshold and direction criteria. Includes all data points above threshold, not just peaks.
            Defaults to "local".
        min_event_magnitude: Minimum absolute change in the base variable per detected dts
            episode (physical units). Magnitude is the peak-anchored plateau difference
            with outer half-width ``min_event_magnitude_L`` and guard
            ``min_event_magnitude_G``. When set, only episodes meeting this threshold are
            clustered. Defaults to None (no magnitude filter).
        min_event_magnitude_L: Outer half-width (time steps) around the dts peak for
            plateau means when ``min_event_magnitude`` is set. Defaults to 25.
        min_event_magnitude_G: Guard half-width (time steps) excluded around the peak.
            Requires ``L > G >= 0``. Defaults to 10.
        time_weight: Controls the relative influence of time in clustering. By default, time values are automatically scaled to match the standard deviation of the spatial coordinates. Increasing time_weight gives more emphasis to the temporal dimension, resulting in clusters that are tighter in time (shorter delays between abrupt events). Decreasing it emphasizes the spatial dimensions, allowing clusters to span a wider range of shift times. Defaults to 1.
        regridder: The regridding method to use from `toad.clustering.regridding`. Defaults to None. If None and coordinates are lat/lon, a HealPixRegridder will be created automatically.
        disable_regridder: Whether to disable the regridder. Defaults to False.
        output_label_suffix: A suffix to add to the output label. Defaults to "".
        overwrite: If True, overwrite existing variable of same name. If False, same name is used with an added number. Defaults to False.
        sort_by_size: Whether to reorder clusters by size. Defaults to True.
        export_for_mma: If set to a file path, exports cluster labels for MMA (multi-model
            aggregation). When regridding is used, HealPix labels are extracted from df_healpix.
        mma_grid: Grid format for MMA export: "healpix" or "native". Defaults to "healpix".
        optimize: Whether to optimize the clustering parameters. Defaults to False.
        optimize_params: Parameters for the optimization. Defaults to default_opt_params.
        optimize_objective: The objective function to optimize. Defaults to combined_spatial_nonlinearity. Can be one of:
            - callable: Custom objective function taking (td, output_label) as arguments
            - "median_heaviside": Median heaviside score across clusters
            - "mean_heaviside": Mean heaviside score across clusters
            - "mean_consistency": Mean consistency score across clusters
            - "mean_spatial_autocorrelation": Mean spatial autocorrelation score
            - "mean_nonlinearity": Mean nonlinearity score across clusters
        optimize_n_trials: Number of trials to run for optimization. Defaults to 50.
        optimize_direction: The direction of the optimization. Defaults to "maximize".
        optimize_log_level: The log level for the optimization. Defaults to optuna.logging.WARNING.
        optimize_progress_bar: Whether to show the progress bar for the optimization. Defaults to True.

    Returns:
        An `xarray.Dataset` containing the original data and the clustering results.

    Notes:
        For global datasets, use `toad.clustering.regridding.HealpyRegridder` to ensure equal spacing between data points and prevent biased clustering at high latitudes.
    """

    """
    Overview of the clustering process:
    1. Input Validation
        - Verify shifts variable exists in dataset (either directly or via base variable)
        - Check data has required 3 dimensions
        - Validate shift_threshold is positive
    2. Preprocessing
        - Generate output label with optional suffix
        - Check for existing results and handle overwrite based on parameters
        - Compute peak/sign mask based on shift_selection ("local"/"global")
        - Filter points based on shift_direction ("both"/"positive"/"negative");
          when "both", cluster each sign separately and merge labels
        - Extract spatial and temporal coordinates
        - Apply optional regridding to standardize coordinates
        - Scale coordinates using sklearn preprocessing
        - Scale time values by time_weight
        - Calculate shift magnitudes at selected points (passed as ``y``; see TODO below)
    3. Clustering
        - Store clustering parameters as metadata
        - Fit clustering model to coordinates (built-in clusterers ignore ``y`` today)
        - Generate cluster labels for each point
    4. Postprocessing
        - Sort clusters by size if requested
        - Scatter labels back to xarray coordinates
        - Add clustering parameters as attributes
        - Merge results with input dataset and return Dataset
    """

    start_time = time_now()

    # ==================== VARIABLE CHECKING ====================
    # if supplied variable is a shift variable, use that
    if td._is_shift_variable(var):
        shifts_variable = var
    else:
        if td._is_cluster_variable(var):
            raise ValueError(
                f"{var} is a cluster variable. Please pass a base or shift variable."
            )

        # if supplied variable is a base variable, check if multiple shifts variables exist
        shift_vars = td.shift_vars_for_var(var)
        if len(shift_vars) > 1:
            raise ValueError(
                f"Multiple shifts variables exist for {var}: {shift_vars}. Please specify which one to use"
            )
        elif len(shift_vars) == 0:
            raise ValueError(
                f"No shifts found for base variable {var}. Please run compute_shifts() for var={var} first."
            )
        # use the first/only shift variable
        shifts_variable = shift_vars[0]

    if td.data[shifts_variable].ndim != 3:
        raise ValueError(
            "Shifts variable must be 3-dimensional: time/forcing x space x space"
        )

    # we add neg sign manually to detect negative shift
    if shift_threshold < 0:
        raise ValueError(f"shift_threshold must be positive, got {shift_threshold}")

    # ==================== LABEL MAKING ====================
    # Set output label (name of shifts_variable + _cluster + output_label_suffix) and check if already in data
    new_output_label = (
        output_label
        if output_label
        else f"{shifts_variable}_cluster{output_label_suffix}"
    )
    if not overwrite:
        new_output_label = get_unique_variable_name(new_output_label, td.data, logger)
    elif overwrite and new_output_label in td.data:
        td.data = td.data.drop_vars(new_output_label)

    # ==================== optimization ====================
    # if optimize is True, optimize the parameters for clustering.
    if optimize:
        if export_for_mma:
            raise ValueError(
                "Optimization is not yet supported when exporting for MMA."
            )
        return _optimize_clusters(
            td=td,
            var=var,
            method=method,
            shift_threshold=shift_threshold,
            shift_direction=shift_direction,
            shift_selection=shift_selection,
            min_event_magnitude=min_event_magnitude,
            min_event_magnitude_L=min_event_magnitude_L,
            min_event_magnitude_G=min_event_magnitude_G,
            time_weight=time_weight,
            regridder=regridder,
            output_label=new_output_label,
            overwrite=True,
            sort_by_size=sort_by_size,
            optimize=False,
            optimize_params=optimize_params,
            optimize_objective=optimize_objective,
            optimize_n_trials=optimize_n_trials,
            optimize_direction=optimize_direction,
            optimize_log_level=optimize_log_level,
            optimize_progress_bar=optimize_progress_bar,
        )

    # ==================== SHIFT SELECTION ====================
    sh = td.data[shifts_variable]

    base_variable_name = sh.attrs.get(_attrs.BASE_VARIABLE)
    if min_event_magnitude is not None:
        if not base_variable_name or base_variable_name not in td.data:
            raise ValueError(
                f"min_event_magnitude requires base variable {base_variable_name!r} in dataset"
            )
        base_da = td.data[base_variable_name]
    else:
        base_da = None

    # Create mask to exclude grid cells with all NaN values
    # Grid cells that are all NaN across time should not be included in clustering
    has_valid_data = ~sh.isnull().all(dim=td.time_dim)

    if shift_selection in ("local", "global"):
        mask_da = _compute_dts_peak_sign_mask(
            sh,
            td.time_dim,
            shift_threshold,
            shift_selection=shift_selection,
            base=base_da,
            min_event_magnitude=min_event_magnitude,
            min_event_magnitude_L=min_event_magnitude_L,
            min_event_magnitude_G=min_event_magnitude_G,
        )
        if shift_direction == "both":
            cond = (mask_da != 0) & has_valid_data
        elif shift_direction == "positive":
            cond = (mask_da > 0) & has_valid_data
        else:  # "negative"
            cond = (mask_da < 0) & has_valid_data
    else:
        # shift_selection == "all": filter original magnitudes
        if shift_direction == "both":
            cond = (np.abs(sh) > shift_threshold) & has_valid_data
        elif shift_direction == "positive":
            cond = (sh > shift_threshold) & has_valid_data
        else:
            cond = (sh < -shift_threshold) & has_valid_data
        if min_event_magnitude is not None:
            episode_mask = _compute_episode_pass_mask(
                sh,
                base_da,
                td.time_dim,
                shift_threshold,
                min_event_magnitude,
                min_event_magnitude_L=min_event_magnitude_L,
                min_event_magnitude_G=min_event_magnitude_G,
            )
            cond = cond & episode_mask

    # boolean → indices per axis (axis order matches sh.dims, not necessarily time-first)
    cond_vals = np.asarray(cond.data)
    idx = np.nonzero(cond_vals)
    n_pts = idx[0].size

    if n_pts == 0:
        logger.warning(
            f'No gridcells left after applying shift_threshold={shift_threshold} and shift_direction="{shift_direction}"'
        )
        # create cluster variable with all NaN (no points were clustered)
        clusters = xr.full_like(sh, np.nan).rename(new_output_label)
        preprocessing_time = 0.0
        clustering_time = 0.0
        cluster_labels = np.array([-1], dtype=int)
        coords = np.empty((0, 3))
        method_params = {}
        regridder_params = {}
        cluster_sign = None
    else:
        # Handle dimensions
        space_dims = td.space_dims
        space_dims = _reorder_space_dims(space_dims)

        # Map flat indices from np.nonzero to dimension names (matches sh.dims axis order)
        dims_sh = list(sh.dims)
        idx_by_dim = {dims_sh[i]: idx[i] for i in range(len(dims_sh))}
        if td.time_dim not in idx_by_dim:
            raise ValueError(
                f"time dimension {td.time_dim!r} not found in shifts dims {dims_sh}"
            )

        # Determine latitude/longitude names and grid type from dataset
        lat_name, lon_name, has_latlon, is_latlon_dims = get_latlon_info(
            td.data, space_dims
        )

        # Build coordinates array (NumPy only, no DataFrame merges)
        # time coordinate
        time_numeric = td.numeric_time_values[idx_by_dim[td.time_dim]]

        # ==================== COORDINATES ====================
        if is_latlon_dims:
            # lat/lon are 1D dims: index directly
            lat_vals = td.data[lat_name].values[idx_by_dim[lat_name]]
            lon_vals = td.data[lon_name].values[idx_by_dim[lon_name]]
            coords = np.column_stack((time_numeric, lat_vals, lon_vals))
        elif has_latlon:
            # Irregular i/j grids: take 2D lat/lon variables aligned with space_dims
            lat_grid = td.data[lat_name].transpose(*space_dims).values
            lon_grid = td.data[lon_name].transpose(*space_dims).values
            space_index_tuple = tuple(idx_by_dim[d] for d in space_dims)
            lat_vals = lat_grid[space_index_tuple]
            lon_vals = lon_grid[space_index_tuple]
            coords = np.column_stack((time_numeric, lat_vals, lon_vals))
        else:
            # No lat/lon (as dims or coords or variables) → Fall back to using raw index dimensions (e.g., x/y or i/j)
            cols = [time_numeric]
            for d in space_dims:
                vals_d = td.data[shifts_variable].coords[d].values
                cols.append(vals_d[idx_by_dim[d]])
            coords = np.column_stack(cols)

        # TODO(clustering-weights): Shift magnitudes are passed as ``y`` but built-in
        # clusterers (HDBSCAN, sklearn DBSCAN, SpaceTimeDBSCAN) do not use them.
        # Magnitudes only affect point selection upstream (shift_threshold / shift_selection).
        # HealPix regridding averages magnitudes when binning duplicate pixels.
        # Decide later whether to:
        #   - stop passing ``y`` by default and document ``y`` as a custom-clusterer hook only;
        #   - implement a weighted clusterer; or
        #   - wire ``sample_weight`` where sklearn supports it.
        # See tutorials/custom_clustering.ipynb for the intended custom ``y`` API.
        # take absolute value of shifts as weights (at selected points)
        vals_sh = np.asarray(sh.data)[idx]
        weights = np.abs(vals_sh)

        # Create HealPixRegridder only for regular 1D lat/lon grids
        if regridder is None and is_latlon_dims and not disable_regridder:
            regridder = HealPixRegridder()

        method = method() if isinstance(method, type) else method
        preprocessing_time = time_now() - start_time
        space_dims_size = (
            td.data.sizes[td.space_dims[0]],
            td.data.sizes[td.space_dims[1]],
        )

        signs = np.sign(vals_sh)
        split_by_sign = shift_direction == "both"
        sign_masks = (
            [signs > 0, signs < 0] if split_by_sign else [np.ones(n_pts, dtype=bool)]
        )

        cluster_labels = np.full(n_pts, -1, dtype=int)
        label_offset = 0
        regridders_used: list[BaseRegridder] = []
        clustering_time = 0.0

        for mask in sign_masks:
            if not np.any(mask):
                continue
            n_sub = int(mask.sum())
            logger.debug(
                f"Applying clusterer {method.__class__.__name__} to {shifts_variable} "
                f"with {n_sub} points" + (" (sign split)" if split_by_sign else "")
            )
            cluster_start = time_now()
            sub_labels, used_regridder = _cluster_coords_subset(
                coords[mask],
                weights[mask],
                method,
                has_latlon=has_latlon,
                regridder=_clone_regridder(regridder),
                disable_regridder=disable_regridder,
                space_dims_size=space_dims_size,
                time_weight=time_weight,
                signs=np.sign(vals_sh[mask]),
            )
            clustering_time += time_now() - cluster_start

            sub_labels = np.asarray(sub_labels, dtype=int)
            valid = sub_labels >= 0
            if label_offset > 0 and np.any(valid):
                sub_labels = sub_labels.copy()
                sub_labels[valid] += label_offset
            if np.any(valid):
                label_offset = int(sub_labels[valid].max()) + 1
            cluster_labels[mask] = sub_labels
            if used_regridder is not None:
                regridders_used.append(used_regridder)

        if regridders_used:
            regridder = regridders_used[-1]
            if len(regridders_used) > 1:
                import pandas as pd

                regridder.df_healpix = pd.concat(
                    [rg.df_healpix for rg in regridders_used],
                    ignore_index=True,
                )

        cluster_labels = (
            sorted_cluster_labels(cluster_labels) if sort_by_size else cluster_labels
        )

        # Scatter labels back into xarray without DataFrame
        # Start with NaN (points not included in clustering remain NaN)
        clusters = xr.full_like(sh, np.nan).rename(new_output_label)
        # Assign cluster labels (including -1 for noise) only to points that were clustered
        clusters.data[idx] = np.asarray(cluster_labels, dtype=np.float64)

        sign_output_label = f"{new_output_label}_sign"
        cluster_sign = xr.full_like(sh, np.nan).rename(sign_output_label)
        sign_at_points = np.sign(vals_sh)
        cluster_sign.data[idx] = np.where(cluster_labels >= 0, sign_at_points, np.nan)

        # Transpose if dimensions don't match (shouldn't be needed but keep)
        if clusters.dims != td.data[shifts_variable].dims:
            clusters = clusters.transpose(*td.data[shifts_variable].dims)
        if cluster_sign.dims != td.data[shifts_variable].dims:
            cluster_sign = cluster_sign.transpose(*td.data[shifts_variable].dims)

        # end of if n_pts > 0

    # Get base variable from shifts attrs
    base_variable = base_variable_name if base_variable_name else "Unknown"

    # Save method params (specifically after clustering, to get all final parameters)
    method_params = {
        f"cluster_{param}": str(value)
        for param, value in dict(sorted(vars(method).items())).items()
        if value is not None and not param.startswith("_")
    }

    # Save regridder params
    regridder_params = {}
    if regridder:
        regridder_params["regridder_name"] = regridder.__class__.__name__
        regridder_params.update(
            {
                f"regridder_{param}": str(value)
                for param, value in dict(sorted(vars(regridder).items())).items()
                if value is not None and isinstance(value, (int, float, str))
            }
        )

    # Save details as attributes (single update block)
    cluster_attrs: dict[str, Any] = {
        _attrs.CLUSTER_IDS: np.unique(cluster_labels).astype(int),
        _attrs.SHIFT_THRESHOLD: shift_threshold,
        _attrs.SHIFT_SELECTION: shift_selection,
        _attrs.SHIFT_DIRECTION: shift_direction,
        _attrs.TIME_WEIGHT: time_weight,
        _attrs.N_DATA_POINTS: n_pts,
        _attrs.METHOD_NAME: method.__class__.__name__,
        _attrs.RUNTIME_PREPROCESSING: float(preprocessing_time),
        _attrs.RUNTIME_CLUSTERING: float(clustering_time),
        _attrs.RUNTIME_TOTAL: float(preprocessing_time + clustering_time),
        _attrs.TOAD_VERSION: __version__,
        _attrs.BASE_VARIABLE: base_variable,
        _attrs.SHIFTS_VARIABLE: shifts_variable,
        _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
        **method_params,
        **regridder_params,
    }
    if min_event_magnitude is not None:
        cluster_attrs[_attrs.MIN_EVENT_MAGNITUDE] = min_event_magnitude
        cluster_attrs[_attrs.MIN_EVENT_MAGNITUDE_L] = min_event_magnitude_L
        cluster_attrs[_attrs.MIN_EVENT_MAGNITUDE_G] = min_event_magnitude_G
    clusters.attrs.update(cluster_attrs)
    if cluster_sign is not None:
        cluster_sign.attrs.update(
            {
                "description": "Per-voxel shift sign at clustered points (+1/-1)",
                _attrs.BASE_VARIABLE: base_variable,
                _attrs.SHIFTS_VARIABLE: shifts_variable,
                _attrs.CLUSTER_LABELS_VAR: new_output_label,
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER_SIGN,
                _attrs.TOAD_VERSION: __version__,
            }
        )

    logger.info(_format_cluster_summary(new_output_label, cluster_labels, n_pts))

    if export_for_mma:
        _export_mma_cluster_labels(
            path=export_for_mma,
            mma_grid=mma_grid,
            td=td,
            clusters=clusters,
            cluster_sign=cluster_sign,
            regridder=regridder,
            source_variable=new_output_label,
        )

    merge_vars = [clusters]
    if cluster_sign is not None:
        merge_vars.append(cluster_sign)
    return xr.merge([td.data, *merge_vars], combine_attrs="override", compat="override")


def _clone_regridder(regridder: BaseRegridder | None) -> BaseRegridder | None:
    if regridder is None:
        return None
    nside = getattr(regridder, "nside", None)
    return regridder.__class__(nside=nside)


def _cluster_coords_subset(
    coords: np.ndarray,
    weights: np.ndarray,
    method: ClusterMixin,
    *,
    has_latlon: bool,
    regridder: BaseRegridder | None,
    disable_regridder: bool,
    space_dims_size: tuple[int, int],
    time_weight: float,
    signs: np.ndarray | None = None,
) -> tuple[np.ndarray, BaseRegridder | None]:
    """Regrid, scale, and cluster one subset of (time, lat, lon) points."""
    used_regridder = regridder
    if used_regridder and not disable_regridder:
        coords, weights = used_regridder.regrid(
            coords, weights, space_dims_size, signs=signs
        )

    if has_latlon:
        coords = geodetic_to_cartesian(
            time=coords[:, 0], lat=coords[:, 1], lon=coords[:, 2]
        )

    space_coords = coords[:, 1:]
    space_std = np.mean(np.std(space_coords, axis=0))
    time_values = coords[:, 0]
    time_mean = np.mean(time_values)
    time_std = np.std(time_values)
    skip_time_scaling = getattr(method, "skip_time_scaling", False)

    if not skip_time_scaling:
        coords = coords.copy()
        if time_std > 0:
            coords[:, 0] = (time_values - time_mean) / time_std * space_std
        if time_weight != 1:
            coords[:, 0] = coords[:, 0] * time_weight

    try:
        labels = np.asarray(method.fit_predict(X=coords, y=weights), dtype=int)
    except ValueError as e:
        msg = str(e)
        insufficient_samples = ("min_samples" in msg and "must be at most" in msg) or (
            "requires more than one sample" in msg
        )
        if insufficient_samples:
            logger.warning(
                "No events available for clustering (%d point(s)); returning no clusters. %s",
                len(coords),
                e,
            )
            labels = np.full(len(coords), -1, dtype=int)
        else:
            raise

    if used_regridder and not disable_regridder:
        labels = used_regridder.regrid_clusters_back(labels)

    return labels, used_regridder if (
        used_regridder and not disable_regridder
    ) else None


def _export_mma_cluster_labels(
    path: str,
    mma_grid: Literal["healpix", "native"],
    td: "TOAD",
    clusters: xr.DataArray,
    cluster_sign: xr.DataArray | None,
    regridder: BaseRegridder | None,
    source_variable: str,
) -> None:
    """Export cluster labels for MMA (multi-model aggregation)."""
    spatial_dims = td.space_dims
    time_dim = td.time_dim

    if mma_grid == "healpix":
        if not (
            isinstance(regridder, HealPixRegridder)
            and regridder.df_healpix is not None
            and len(regridder.df_healpix) > 0
            and "cluster" in regridder.df_healpix.columns
        ):
            raise ValueError(
                "mma_grid='healpix' requires HealPix regridding. Pass regridder=HealPixRegridder(nside=...) "
                "to compute_clusters and ensure the grid has lat/lon coordinates."
            )
        df = regridder.df_healpix
        nside = regridder.nside if regridder.nside is not None else 32
        npix = 12 * nside**2
        time_vals = td.data[time_dim].values
        n_time = len(time_vals)
        time_to_idx = {t: i for i, t in enumerate(time_vals)}
        cluster_healpix = np.full((n_time, npix), np.nan, dtype=np.float32)
        cluster_sign_healpix = np.full((n_time, npix), np.nan, dtype=np.float32)
        for _, row in df.iterrows():
            t_idx = time_to_idx.get(row["time"])
            if t_idx is None:
                continue
            hp_idx = int(row["hp_pix"])
            if 0 <= hp_idx < npix:
                cluster_healpix[t_idx, hp_idx] = np.float32(row["cluster"])
                if (
                    "sign" in row
                    and np.isfinite(row["cluster"])
                    and row["cluster"] >= 0
                ):
                    cluster_sign_healpix[t_idx, hp_idx] = np.float32(
                        np.sign(row["sign"])
                    )

        cluster_attrs = {
            "description": "Cluster variable (time, hp_pixel). For consensus and shift time extraction.",
            "source_variable": source_variable,
            "format": "healpix",
            "nside": nside,
        }
        if _attrs.CLUSTER_IDS in clusters.attrs:
            cluster_attrs[_attrs.CLUSTER_IDS] = clusters.attrs[_attrs.CLUSTER_IDS]

        data_vars = {
            "cluster": (
                (time_dim, "hp_pixel"),
                cluster_healpix,
                cluster_attrs,
            ),
            "cluster_sign": (
                (time_dim, "hp_pixel"),
                cluster_sign_healpix,
                {
                    "description": "Per-voxel shift sign (+1/-1) aligned with cluster labels",
                    "source_variable": source_variable,
                    "format": "healpix",
                    "nside": nside,
                    _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER_SIGN,
                    _attrs.CLUSTER_LABELS_VAR: "cluster",
                },
            ),
        }
        our_dims = {time_dim, *spatial_dims}
        all_coords = {
            k: v for k, v in td.data.coords.items() if set(v.dims).issubset(our_dims)
        }
        latlon_candidates = {
            "lat",
            "latitude",
            "lon",
            "longitude",
            "nav_lat",
            "nav_lon",
        }
        for k in latlon_candidates:
            if (
                k in td.data
                and k not in all_coords
                and set(td.data[k].dims).issubset(our_dims)
            ):
                all_coords[k] = td.data[k]
        out = xr.Dataset(
            data_vars,
            coords={**all_coords, "hp_pixel": np.arange(npix)},
            attrs={
                "format": "healpix",
                "nside": nside,
                "Conventions": "TOAD_cluster_labels_v2",
            },
        )
    else:
        our_dims = {time_dim, *spatial_dims}
        all_coords = {
            k: v for k, v in td.data.coords.items() if set(v.dims).issubset(our_dims)
        }
        latlon_candidates = {
            "lat",
            "latitude",
            "lon",
            "longitude",
            "nav_lat",
            "nav_lon",
        }
        for k in latlon_candidates:
            if (
                k in td.data
                and k not in all_coords
                and set(td.data[k].dims).issubset(our_dims)
            ):
                all_coords[k] = td.data[k]
        cluster_attrs = {
            "description": "Cluster variable in original dims for shift time extraction",
            "source_variable": source_variable,
        }
        if _attrs.CLUSTER_IDS in clusters.attrs:
            cluster_attrs[_attrs.CLUSTER_IDS] = clusters.attrs[_attrs.CLUSTER_IDS]
        native_vars = {
            "cluster": (
                clusters.dims,
                clusters.values.astype(np.float32),
                cluster_attrs,
            ),
        }
        if cluster_sign is not None:
            native_vars["cluster_sign"] = (
                cluster_sign.dims,
                cluster_sign.values.astype(np.float32),
                {
                    "description": "Per-voxel shift sign (+1/-1) aligned with cluster labels",
                    "source_variable": source_variable,
                    _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER_SIGN,
                    _attrs.CLUSTER_LABELS_VAR: "cluster",
                },
            )
        out = xr.Dataset(
            native_vars,
            coords=all_coords,
            attrs={"format": "native", "Conventions": "TOAD_cluster_labels_v2"},
        )
    out.to_netcdf(path)
    logger.info(f"Exported cluster labels for MMA to {path} (mma_grid={mma_grid})")


def _format_cluster_summary(
    output_label: str, cluster_labels: np.ndarray, n_points_used: int
) -> str:
    """
    Produce a concise summary:
      - name of the new variable (output_label)
      - number of identified clusters (excluding -1)
      - number of data points used (after filtering)
      - percentage of points labeled as noise (-1)
    """
    n = int(n_points_used)
    if n == 0:
        return f"{output_label}: Identified 0 CLUSTERS in 0 points"

    labels = np.asarray(cluster_labels)
    noise = int(np.count_nonzero(labels == -1))
    pct_noise = 100.0 * noise / n
    n_clusters = int(np.unique(labels[labels != -1]).size)

    # nice, compact, and informative
    clusters_text = f"{n_clusters} {'cluster' if n_clusters == 1 else 'clusters'}"
    return (
        f"New cluster variable \033[1m{output_label}\033[0m: Identified \033[1m{clusters_text}\033[0m in {n:,} pts; "
        f"Left {pct_noise:.1f}% as noise"
        f" ({noise:,} pts)."
    )


def sorted_cluster_labels(cluster_labels: np.ndarray) -> np.ndarray:
    """Sort clusters by size (largest cluster -> 0, second largest -> 1, etc., keeping -1 and NaN).

    Non-finite values (e.g. NaN meaning “no shift” in a label field) are left unchanged
    and excluded from the size-based renumbering.
    """
    original_shape = cluster_labels.shape
    flat = np.ravel(np.asarray(cluster_labels))
    valid = np.isfinite(flat) & (flat != -1)
    if not np.any(valid):
        return np.asarray(cluster_labels).copy()

    unique_labels, inverse = np.unique(
        flat[valid].astype(np.int64, copy=False),
        return_inverse=True,
    )
    counts = np.bincount(inverse)
    order = np.argsort(counts)[::-1]
    new_ids = np.empty(unique_labels.size, dtype=np.int64)
    new_ids[order] = np.arange(unique_labels.size, dtype=np.int64)

    out = np.asarray(flat, dtype=np.float64).copy()
    out[valid] = new_ids[inverse]
    out = out.reshape(original_shape)
    if np.issubdtype(cluster_labels.dtype, np.integer) and np.all(np.isfinite(flat)):
        return out.astype(np.int64)
    return out


def geodetic_to_cartesian(time, lat, lon, height=0) -> np.ndarray:
    """Converts geodetic coordinates to Cartesian coordinates.

    Transforms geodetic coordinates (time, latitude, longitude, optional height) into
    Cartesian coordinates (time, x, y, z) using the WGS84 ellipsoid model.

    Args:
        time: Array of timestamps.
        lat: Array of latitudes in degrees.
        lon: Array of longitudes in degrees.
        height: Optional array of heights above ellipsoid in km. Defaults to 0.

    Returns:
        np.ndarray: Array of shape (n, 4) containing [time, x, y, z] coordinates,
            where x, y, z are in km from the Earth's center.
    """
    # WGS84 parameters
    a = 6378.137  # semi-major axis (km)
    b = 6356.752  # semi-minor axis (km)
    e2 = 1 - (b**2 / a**2)  # eccentricity squared

    # Convert latitude and longitude to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Cartesian coordinates
    x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (b**2 / a**2 * N + height) * np.sin(lat_rad)

    return np.column_stack((time, x, y, z))  # Shape: (n, 4)
