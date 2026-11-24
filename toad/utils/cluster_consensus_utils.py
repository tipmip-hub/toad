from typing import Any, Literal, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from toad.utils import detect_latlon_names

StitchMeridianSetting = bool | Literal["auto"]
FULL_LONGITUDE_COVERAGE_DEG = 350.0

# ---------------------------------------------------------------------------
# Meridian seam detection (for stitch_meridian="auto")
# ---------------------------------------------------------------------------


def _mean_abs_circular_step(lon_vals: np.ndarray, axis: int) -> float:
    """Mean absolute longitude step (degrees, wrapped to [0, 180]) along ``axis``."""
    d = np.abs(np.diff(np.asarray(lon_vals, dtype=np.float64), axis=axis))
    d = np.minimum(np.mod(d, 360.0), 360.0 - np.mod(d, 360.0))
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else 0.0


def longitude_last_spatial_dims(
    dataset: xr.Dataset,
    spatial_dims: Tuple[str, str],
) -> Tuple[str, str]:
    """Return ``spatial_dims`` ordered so that the longitude dimension comes last.

    Seam stitching (``stitch_meridian``) wraps the *last* spatial axis. This helper
    finds the axis along which geographic longitude varies and puts it last:

    * a 1D longitude coordinate on one of ``spatial_dims`` → that dimension;
    * a 2D longitude coordinate over both ``spatial_dims`` (curvilinear grid) → the
      first dimension only if longitude changes more than twice as much per step
      along it as along the second.

    If no longitude coordinate is found (e.g. projected ``x``/``y`` grids), the order
    is returned unchanged.
    """
    sd0, sd1 = spatial_dims
    _, lon_name = detect_latlon_names(dataset)
    if lon_name is None:
        return (sd0, sd1)
    lon_coord = dataset.coords.get(lon_name)
    if lon_coord is None:
        lon_coord = dataset.get(lon_name)
    if lon_coord is None:
        return (sd0, sd1)

    if lon_coord.ndim == 1:
        if lon_coord.dims[0] == sd0:
            return (sd1, sd0)
        return (sd0, sd1)

    if lon_coord.ndim == 2 and set(lon_coord.dims) == {sd0, sd1}:
        lon_vals = np.asarray(lon_coord.transpose(sd0, sd1).values, dtype=np.float64)
        step_along_0 = _mean_abs_circular_step(lon_vals, axis=0)
        step_along_1 = _mean_abs_circular_step(lon_vals, axis=1)
        # Switch only when clearly dominant, so ambiguous (e.g. polar) grids keep
        # their order.
        if step_along_0 > 2.0 * step_along_1:
            return (sd1, sd0)
    return (sd0, sd1)


def _longitude_vector_along_x_dim(
    dataset: xr.Dataset,
    spatial_dims: Tuple[str, str],
    lon_name: str,
) -> np.ndarray | None:
    """Return longitude values along the last native spatial dimension."""
    x_dim = spatial_dims[1]
    lon_coord = dataset.coords.get(lon_name)
    if lon_coord is None:
        lon_coord = dataset.get(lon_name)
    if lon_coord is None:
        return None

    lon_vals = np.asarray(lon_coord.values, dtype=np.float64)
    if lon_vals.ndim == 1:
        if x_dim in lon_coord.dims and len(lon_coord.dims) == 1:
            return lon_vals
        return None

    if lon_vals.ndim == 2 and set(lon_coord.dims) == set(spatial_dims):
        y_dim = spatial_dims[0]
        if lon_coord.dims[0] == y_dim:
            mid = lon_vals.shape[0] // 2
            return lon_vals[mid, :]
        mid = lon_vals.shape[1] // 2
        return lon_vals[:, mid]
    return None


def _longitude_coverage_degrees(lon_vec: np.ndarray) -> float:
    """Angular span covered by ``lon_vec`` on a 0–360° circle."""
    lon = np.mod(np.asarray(lon_vec, dtype=np.float64), 360.0)
    lon = lon[np.isfinite(lon)]
    if lon.size < 2:
        return 0.0
    sorted_lon = np.sort(lon)
    gaps = np.diff(np.concatenate([sorted_lon, sorted_lon[:1] + 360.0]))
    return float(360.0 - gaps.max())


def _longitude_seam_gap_degrees(lon_vec: np.ndarray) -> float:
    """Shortest angular distance between the first and last grid columns."""
    lon = np.mod(np.asarray(lon_vec, dtype=np.float64), 360.0)
    lon0, lon1 = float(lon[0]), float(lon[-1])
    return min(abs(lon1 - lon0), 360.0 - abs(lon1 - lon0))


def infer_stitch_meridian(
    dataset: xr.Dataset,
    spatial_dims: Tuple[str, str],
) -> bool:
    """Return True when the native grid spans nearly all longitudes with a wrapped seam.

    The longitude dimension may be either of ``spatial_dims``; callers that stitch
    must order their grid with :func:`longitude_last_spatial_dims` first.
    """
    _, lon_name = detect_latlon_names(dataset)
    if lon_name is None:
        return False
    spatial_dims = longitude_last_spatial_dims(dataset, spatial_dims)

    lon_vec = _longitude_vector_along_x_dim(dataset, spatial_dims, lon_name)
    if lon_vec is None or lon_vec.size < 2:
        return False

    coverage = _longitude_coverage_degrees(lon_vec)
    if coverage < FULL_LONGITUDE_COVERAGE_DEG:
        return False

    seam_gap = _longitude_seam_gap_degrees(lon_vec)
    mean_spacing = coverage / max(lon_vec.size - 1, 1)
    return seam_gap <= max(2.0 * mean_spacing, 5.0)


def resolve_stitch_meridian(
    setting: StitchMeridianSetting,
    *,
    dataset: xr.Dataset,
    spatial_dims: Tuple[str, str],
) -> bool:
    """Resolve ``stitch_meridian`` from ``False``, ``True``, or ``\"auto\"``."""
    if setting is True:
        return True
    if setting is False:
        return False
    if setting == "auto":
        return infer_stitch_meridian(dataset, spatial_dims)
    raise ValueError(
        f"`stitch_meridian` must be False, True, or 'auto', got {setting!r}."
    )


def _empty_transition_time_df() -> pd.DataFrame:
    """Return an empty long-form transition-time dataframe with stable dtypes."""
    return pd.DataFrame(
        {
            "consensus_cluster_id": pd.Series(dtype=np.int64),
            "cluster_var": pd.Series(dtype=str),
            "transition_time": pd.Series(dtype=np.float64),
        }
    )


def _empty_label_field_shift_time_df() -> pd.DataFrame:
    """Empty table for :func:`label_field_shift_time_samples` (per-label-field events)."""
    return pd.DataFrame(
        {
            "cluster_id": pd.Series(dtype=np.int64),
            "transition_time": pd.Series(dtype=np.float64),
        }
    )


def _consensus_cluster_vars(td: Any, da_clusters: xr.DataArray) -> list[str]:
    """Resolve which input cluster variables contributed to a consensus result."""
    cluster_vars_attr = da_clusters.attrs.get("cluster_vars")
    if cluster_vars_attr is None:
        return list(td.cluster_vars)
    return list(cluster_vars_attr)


def _consensus_input_support_mask(
    td: Any,
    da_clusters: xr.DataArray,
    cluster_var: str,
    *,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
) -> xr.DataArray:
    """Boolean mask of non-noise labels in one input clustering.

    Returns where ``cluster_var`` has a cluster assignment (label ``>= 0``), with the
    same dimensionality as ``da_clusters`` when ``time_dim`` is a dimension; otherwise
    a 2D mask with any cluster activity collapsed over time.

    This does **not** intersect with ``da_clusters`` — callers combine it with consensus
    labels themselves. For example, :func:`consensus_shift_time_distribution` requires
    consensus ``>= 0`` **and** this mask at the **same** ``(time, y, x)``; see
    :meth:`Aggregation.consensus_extraction_mask_2d` for the looser rule used when
    extracting full time series over a shared spatial footprint.
    """
    if spatial_dims is None:
        spatial_dims = tuple(td.space_dims)
    if time_dim is None and td.time_dim in da_clusters.dims:
        time_dim = td.time_dim

    labels = td.data[cluster_var].transpose(
        td.time_dim, spatial_dims[0], spatial_dims[1]
    )
    support_3d = labels.notnull() & (labels >= 0)
    if time_dim is not None and time_dim in da_clusters.dims:
        return support_3d.transpose(time_dim, spatial_dims[0], spatial_dims[1])
    return support_3d.any(dim=td.time_dim)


def _build_consensus_summary_df_spacetime(
    td: Any,
    labels3d: xr.DataArray,
    rate3d: xr.DataArray,
    spatial_dims: Tuple[str, str],
    time_dim: str,
) -> pd.DataFrame:
    """Build summary statistics over all ``(time × space)`` consensus labels.

    One row is returned per cluster id that appears anywhere in the spacetime field.

    * ``area`` — number of unique **spatial** cells in the cluster footprint (any time).
    * ``mean_consensus_rate`` — mean member-support fraction over **spacetime** voxels
      with that id (a cell appearing at many timesteps contributes multiple values).
    * Transition-time columns — from :func:`consensus_shift_time_distribution`: event
      times only where consensus and the input both have a non-noise label at the
      **same** ``(time, y, x)``. ``median_median_*`` columns summarise across inputs;
      ``pooled_*`` columns pool every event row (voxel-weighted, not one vote per input).

    For full base-variable time series over a shared region (times need not match),
    use :meth:`Aggregation.consensus_cluster_timeseries` instead — it uses a looser
    2D footprint via :meth:`Aggregation.consensus_extraction_mask_2d`.

    Args:
        td: TOAD object containing clustering results.
        labels3d: 3D DataArray of consensus cluster labels (``-1`` = shift but not
            in consensus; ``NaN`` = no abrupt shift in any input, matching ``compute_clusters``).
        rate3d: 3D consensus rate scores, same dims as ``labels3d``.
        spatial_dims: Tuple of spatial dimension names.
        time_dim: Name of the time dimension (must be a dim of ``labels3d``).

    Returns:
        DataFrame with one row per consensus cluster id present in ``labels3d``.
    """
    sd0, sd1 = spatial_dims
    dim = labels3d.name if labels3d.name else "cluster"
    # Exclude noise (-1) and no-shift (NaN); ``x == x`` is False for NaN.
    cluster_map = labels3d.where((labels3d >= 0) & (labels3d == labels3d))

    empty_cols = [
        "cluster_id",
        "mean_consensus_rate",
        "area",
        f"mean_{sd0}",
        f"mean_{sd1}",
        "median_median_shift_time",
        "std_median_shift_time",
        "median_std_shift_time",
        "std_std_shift_time",
        "pooled_median_shift_time",
        "pooled_std_shift_time",
    ]
    v = np.asarray(labels3d.values, dtype=np.float64)
    if not np.any(np.isfinite(v) & (v >= 0)):
        return pd.DataFrame({c: [] for c in empty_cols})

    mean_rate = rate3d.groupby(cluster_map).mean(skipna=True)
    cluster_ids = mean_rate[dim].values.astype(int)
    area_vals: list[int] = []
    mean_sd0_vals: list[float] = []
    mean_sd1_vals: list[float] = []
    coord0 = td.data[sd0]
    coord1 = td.data[sd1]
    # --- per-cluster spatial footprint and centroid (time collapsed) ---
    for cid in cluster_ids:
        footprint = (labels3d == cid).any(dim=time_dim)
        area_vals.append(int(footprint.sum(skipna=True).item()))
        mean_sd0_vals.append(float(coord0.where(footprint).mean(skipna=True).item()))
        mean_sd1_vals.append(float(coord1.where(footprint).mean(skipna=True).item()))

    df = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "mean_consensus_rate": mean_rate.values.astype(np.float32),
            "area": np.asarray(area_vals, dtype=np.int32),
            f"mean_{sd0}": np.asarray(mean_sd0_vals, dtype=np.float32),
            f"mean_{sd1}": np.asarray(mean_sd1_vals, dtype=np.float32),
        }
    )

    # --- strict same-(t,y,x) event times → summary shift columns ---
    dist_ds, df_cell = consensus_shift_time_distribution(
        td,
        labels3d,
        spatial_dims=spatial_dims,
        time_dim=time_dim,
    )
    if len(dist_ds.data_vars) == 0:
        df_transitions = pd.DataFrame(
            {
                "cluster_id": df["cluster_id"].values.astype(int),
                "median_median_shift_time": np.nan,
                "std_median_shift_time": np.nan,
                "median_std_shift_time": np.nan,
                "std_std_shift_time": np.nan,
            }
        )
    else:
        # Per-input-map spatial median / std over voxels; outer medians are over cluster_var.
        per_cluster_per_model_median = dist_ds["spatial_median_transition_time"]
        per_cluster_per_model_std = dist_ds["spatial_std_transition_time"]

        finite_median = per_cluster_per_model_median.where(
            np.isfinite(per_cluster_per_model_median)
        )
        median_median_shift_time = finite_median.median(dim="cluster_var", skipna=True)

        std_median_shift_time_by = finite_median.std(
            dim="cluster_var", skipna=True
        ).fillna(0.0)
        finite_std = per_cluster_per_model_std.where(
            np.isfinite(per_cluster_per_model_std)
        )
        median_std_shift_time = finite_std.median(
            dim="cluster_var", skipna=True
        ).fillna(0.0)
        std_std_shift_time = finite_std.std(dim="cluster_var", skipna=True).fillna(0.0)

        group_dim = "consensus_cluster_id"
        df_transitions = pd.DataFrame(
            {
                "cluster_id": median_median_shift_time[group_dim].values.astype(int),
                "median_median_shift_time": median_median_shift_time.values.astype(
                    np.float32
                ),
                "std_median_shift_time": std_median_shift_time_by.values.astype(
                    np.float32
                ),
                "median_std_shift_time": median_std_shift_time.values.astype(
                    np.float32
                ),
                "std_std_shift_time": std_std_shift_time.values.astype(np.float32),
            }
        )

    df = df.merge(df_transitions, on="cluster_id", how="left")
    pooled_m, pooled_s = _pooled_median_std_from_df_cell(
        df["cluster_id"].to_numpy(np.int64), df_cell
    )
    df["pooled_median_shift_time"] = pooled_m
    df["pooled_std_shift_time"] = pooled_s
    return df


def _pooled_median_std_from_df_cell(
    cluster_ids: np.ndarray, df_cell: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Median and sample std of pooled transition_time samples per consensus cluster id.

    Pooled = all voxels in the long table from :func:`consensus_shift_time_distribution`,
    with duplicate ``(cluster_var, …)`` rows kept (each input can contribute many events).
    """
    n = int(cluster_ids.size)
    med = np.full(n, np.nan, dtype=np.float32)
    std = np.full(n, np.nan, dtype=np.float32)
    if (
        df_cell is None
        or df_cell.empty
        or "consensus_cluster_id" not in df_cell.columns
    ):
        return med, std
    for i, raw in enumerate(np.asarray(cluster_ids, dtype=np.int64).ravel()):
        cid = int(raw)
        sub = df_cell.loc[df_cell["consensus_cluster_id"] == cid, "transition_time"]
        v = sub.to_numpy(dtype=np.float64, copy=False)
        v = v[np.isfinite(v)]
        if v.size:
            med[i] = np.float32(np.median(v))
            std[i] = np.float32(np.std(v, ddof=1)) if v.size > 1 else np.float32(np.nan)
    return med, std


def label_field_shift_time_samples(
    td: Any,
    da_labels: xr.DataArray,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
) -> pd.DataFrame:
    """Event-time of each spacetime cell with a non-noise label (one clustering, any kind).

    For each point where ``da_labels >= 0``, records the numeric :attr:`~toad.TOAD.numeric_time_value`
    at that timestep. Same time-coordinate convention as
    :func:`consensus_shift_time_distribution` for the consensus case (event times at labelled
    voxels), but **without** intersecting with other cluster maps—use a single
    3D label field (e.g. a normal ``*cluster`` variable on ``td`` or consensus labels
    with one logical map if passed manually).

    Args:
        td: TOAD instance (time coordinate, ``space_dims``).
        da_labels: 3D cluster labels, ``(time, y, x)`` (or with ``time_dim`` as below).
        spatial_dims: Defaults to ``tuple(td.space_dims)``.
        time_dim: Defaults to ``td.time_dim`` when that dimension is present on ``da_labels``.

    Returns:
        Long-form ``DataFrame`` with columns ``cluster_id``, ``transition_time``.
    """
    if spatial_dims is None:
        spatial_dims = tuple(td.space_dims)

    if time_dim is None and td.time_dim in da_labels.dims:
        time_dim = td.time_dim
    if time_dim is None or time_dim not in da_labels.dims:
        raise ValueError(
            "`label_field_shift_time_samples` requires 3D time-resolved label field "
            f"(time_dim={time_dim!r} not in dims={tuple(da_labels.dims)})."
        )

    if not np.any(np.isfinite(np.asarray(da_labels.values, dtype=np.float64))):
        return _empty_label_field_shift_time_df()

    time_values = np.asarray(td.numeric_time_values, dtype=np.float64)
    t_len = int(da_labels.sizes[time_dim])
    if time_values.size != t_len:
        raise ValueError(
            f"TOAD time coordinate length {time_values.size} != label field "
            f"length {t_len!r} along {time_dim!r}."
        )

    lab = np.asarray(da_labels.values, order="C")
    tt_b = np.broadcast_to(
        time_values.reshape((-1, 1, 1)),
        lab.shape,
    )
    m = lab >= 0
    cid = lab[m].astype(np.int64, copy=False)
    ttv = tt_b[m]
    fin = np.isfinite(ttv)
    if not np.any(fin):
        return _empty_label_field_shift_time_df()
    return pd.DataFrame(
        {
            "cluster_id": cid[fin].astype(np.int64, copy=False),
            "transition_time": ttv[fin].astype(np.float64, copy=False),
        }
    )


def label_field_shift_time_distributions(
    td: Any,
    da_labels: xr.DataArray,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
) -> dict[int, np.ndarray]:
    """Plotting-friendly transition-time samples grouped by cluster id in one label field.

    Pooled the same way as :func:`consensus_shift_time_distributions` (one array per id).
    In spacetime, the same cell can appear at multiple times if the cluster footprint
    spans several timesteps.
    """
    df = label_field_shift_time_samples(
        td, da_labels, spatial_dims=spatial_dims, time_dim=time_dim
    )
    if df.empty:
        return {}
    out: dict[int, np.ndarray] = {}
    for cid, grp in df.groupby("cluster_id", sort=True):
        vals = grp["transition_time"].to_numpy(dtype=np.float64, copy=True)
        out[int(np.asarray(cid).item())] = vals[np.isfinite(vals)]
    return out


def consensus_shift_time_distribution(
    td: Any,
    da_clusters: xr.DataArray,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Per-consensus-cluster event-time samples used to build summary shift columns.

    Each sample is the numeric time coordinate at a spacetime voxel where **both**
    the consensus label and the input ``cluster_var`` are non-noise (``>= 0``) at
    the **same** ``(time, y, x)``. Dilated-only support or events at different
    timesteps on the same grid cell do not contribute.

    This is stricter than :meth:`Aggregation.consensus_extraction_mask_2d`, which
    collapses over time so that consensus at one timestep and an input cluster at
    another can still define the same spatial cell — that looser rule is for
    extracting full time series, not for timing statistics here.

    **Dataset** (when clusterings exist):

    - ``spatial_mean_transition_time`` / ``spatial_median_transition_time`` /
      ``spatial_std_transition_time`` — per ``(consensus_cluster_id, cluster_var)`` over
      the matching voxels above.

    **Summary table mapping:**

    - ``median_median_shift_time`` — median over ``cluster_var`` of ``spatial_median``.
    - ``std_median_shift_time`` — std over ``cluster_var`` of ``spatial_median``.
    - ``median_std_shift_time`` / ``std_std_shift_time`` — same for ``spatial_std``.
    - ``pooled_median_shift_time`` / ``pooled_std_shift_time`` — median and sample std
      of all ``transition_time`` rows for that id in the long dataframe.

    **Long DataFrame:** columns ``consensus_cluster_id``, ``cluster_var``,
    ``transition_time`` (one row per matching voxel; the same ``(y, x)`` may appear
    multiple times at different timesteps).

    Args:
        td: TOAD instance with ``cluster_vars`` and shifts.
        da_clusters: Time-resolved consensus ``clusters`` from
            :meth:`Aggregation.compute_consensus`.
        spatial_dims: Grid dimension names; default ``tuple(td.space_dims)``.
        time_dim: Time dimension of ``da_clusters`` if 3D; default ``td.time_dim`` when
            that dimension is present.

    Returns:
        ``(dataset, dataframe)``. If there are no cluster variables or no positive
        consensus labels, returns an empty Dataset and an empty DataFrame with the
        expected columns.
    """
    if spatial_dims is None:
        spatial_dims = tuple(td.space_dims)
    sd0, sd1 = spatial_dims

    if time_dim is None and td.time_dim in da_clusters.dims:
        time_dim = td.time_dim
    if time_dim is None or time_dim not in da_clusters.dims:
        raise ValueError(
            "`consensus_shift_time_distribution` requires a time-resolved consensus field."
        )

    labels = da_clusters
    lv = np.asarray(labels.values, dtype=np.float64)
    if not np.any(np.isfinite(lv) & (lv >= 0)):
        empty_ds = xr.Dataset(attrs={"note": "no consensus clusters (all noise)"})
        return empty_ds, _empty_transition_time_df()

    cluster_vars = _consensus_cluster_vars(td, da_clusters)
    if not cluster_vars:
        empty_ds = xr.Dataset(attrs={"note": "no cluster_vars on td"})
        return empty_ds, _empty_transition_time_df()

    parts_pc: list[pd.DataFrame] = []
    time_values = np.asarray(td.numeric_time_values, dtype=np.float64)
    for cvar in cluster_vars:
        support_mask = _consensus_input_support_mask(
            td,
            da_clusters,
            cvar,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )
        # Keep consensus label only where input also has a cluster at the same voxel
        support_labels = labels.where(support_mask, other=-1)
        lab = np.asarray(support_labels.values)
        tt_b = np.broadcast_to(
            time_values.reshape((-1, 1, 1)),
            lab.shape,
        )
        m = lab >= 0
        cid = lab[m]
        ttv = tt_b[m]
        fin = np.isfinite(ttv)
        if not np.any(fin):
            continue
        parts_pc.append(
            pd.DataFrame(
                {
                    "consensus_cluster_id": cid[fin].astype(np.int64, copy=False),
                    "cluster_var": cvar,
                    "transition_time": ttv[fin].astype(np.float64, copy=False),
                }
            )
        )
    if not parts_pc:
        empty_ds = xr.Dataset(attrs={"note": "no supported transition-time samples"})
        return empty_ds, _empty_transition_time_df()

    df_cell = pd.concat(parts_pc, ignore_index=True)

    cluster_ids = np.sort(df_cell["consensus_cluster_id"].unique().astype(np.int64))
    mean_arr = np.full((cluster_ids.size, len(cluster_vars)), np.nan, dtype=np.float32)
    median_arr = np.full(
        (cluster_ids.size, len(cluster_vars)), np.nan, dtype=np.float32
    )
    std_arr = np.full((cluster_ids.size, len(cluster_vars)), np.nan, dtype=np.float32)
    cluster_id_to_idx = {int(cid): i for i, cid in enumerate(cluster_ids.tolist())}
    cluster_var_to_idx = {cvar: i for i, cvar in enumerate(cluster_vars)}
    grouped = (
        df_cell.groupby(["consensus_cluster_id", "cluster_var"])["transition_time"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    for cid, cvar, mean_val, std_val, median_val in grouped.itertuples(
        index=False, name=None
    ):
        i = cluster_id_to_idx[int(cid)]
        j = cluster_var_to_idx[str(cvar)]
        mean_arr[i, j] = np.float32(mean_val)
        std_arr[i, j] = np.float32(0.0 if pd.isna(std_val) else std_val)
        median_arr[i, j] = np.float32(median_val)

    ds_out = xr.Dataset(
        {
            "spatial_mean_transition_time": xr.DataArray(
                mean_arr,
                dims=["consensus_cluster_id", "cluster_var"],
                coords={
                    "consensus_cluster_id": cluster_ids.astype(np.int64),
                    "cluster_var": pd.Index(cluster_vars, name="cluster_var"),
                },
            ),
            "spatial_median_transition_time": xr.DataArray(
                median_arr,
                dims=["consensus_cluster_id", "cluster_var"],
                coords={
                    "consensus_cluster_id": cluster_ids.astype(np.int64),
                    "cluster_var": pd.Index(cluster_vars, name="cluster_var"),
                },
            ),
            "spatial_std_transition_time": xr.DataArray(
                std_arr,
                dims=["consensus_cluster_id", "cluster_var"],
                coords={
                    "consensus_cluster_id": cluster_ids.astype(np.int64),
                    "cluster_var": pd.Index(cluster_vars, name="cluster_var"),
                },
            ),
        },
        attrs={
            "spatial_dims": f"{sd0}, {sd1}",
            "support_rule": (
                "consensus and input both non-noise at the same (time, y, x) voxel"
            ),
        },
    )

    return ds_out, df_cell


def consensus_shift_time_distributions(
    td: Any,
    da_clusters: xr.DataArray,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
    *,
    distribution_result: tuple[xr.Dataset, pd.DataFrame] | None = None,
    source_input_cluster_var: str | None = None,
) -> dict[int, np.ndarray]:
    """Plotting-friendly transition-time samples grouped by consensus cluster id.

    This is a convenience wrapper around :func:`consensus_shift_time_distribution`.
    It aggregates the returned long-form dataframe across all input ``cluster_var``
    values and returns one 1D ``numpy`` array per consensus cluster, suitable for
    violin plots or histograms.

    The samples match the values underlying the summary shift columns. In spacetime
    mode this means a spatial cell can contribute multiple times if it appears in the
    same consensus component at multiple timesteps.

    Args:
        distribution_result: If provided, must be a value already returned from
            :func:`consensus_shift_time_distribution` for the same ``td``, ``da_clusters``,
            and options; the inner call is skipped (avoids duplicate work when both the
            dataset and grouped arrays are needed).
        source_input_cluster_var: If set, keep only rows whose ``cluster_var`` column
            equals this name (one input clustering’s events for each consensus id).
    """
    if distribution_result is not None:
        _, df_cell = distribution_result
    else:
        _, df_cell = consensus_shift_time_distribution(
            td,
            da_clusters,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )
    if source_input_cluster_var is not None:
        df_cell = df_cell[df_cell["cluster_var"] == source_input_cluster_var]
    if df_cell.empty:
        return {}

    out: dict[int, np.ndarray] = {}
    for cid, grp in df_cell.groupby("consensus_cluster_id", sort=True):
        vals = np.asarray(grp["transition_time"], dtype=np.float64).copy()
        out[int(np.asarray(cid).item())] = vals[np.isfinite(vals)]
    return out


# Native grid edges for meridian-stitch connectivity; re-export for tests.
from toad.postprocessing.member_support_consensus import (  # noqa: E402, F401
    _native_edges_from_mask,
)
