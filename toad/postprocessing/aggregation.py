import logging
from typing import Any, List, Literal

import numpy as np
import pandas as pd
import xarray as xr

from toad._version import __version__
from toad.postprocessing.member_support_consensus import (
    SpatialTolerance,
    TemporalTolerance,
    _accumulate_member_support,
    _build_grid_context,
    _build_member_support_dataset,
    _empty_result,
    area_rank_id_mapping,
    cluster_spatial_areas,
    consensus_sign_var_name,
    min_consensus_members,
    relabel_by_id_mapping,
    spatial_tolerance_attr_value,
    temporal_tolerance_attr_value,
    validate_consensus_tolerances,
)
from toad.utils import _attrs, get_unique_variable_name
from toad.utils.cluster_consensus_utils import (
    StitchMeridianSetting,
    _build_consensus_summary_df_spacetime,
    _consensus_input_support_mask,
    consensus_shift_time_distribution,
    consensus_shift_time_distributions,
    label_field_shift_time_distributions,
    label_field_shift_time_samples,
    resolve_stitch_meridian,
)

logger = logging.getLogger("TOAD")


def _format_consensus_summary(output_label: str, labels: np.ndarray) -> str:
    """One-line summary for logging after :meth:`Aggregation.compute_consensus`.

    Mirrors :func:`toad.clustering._format_cluster_summary` in tone and ANSI styling.
    """
    flat = np.asarray(labels, dtype=np.float64).ravel()
    n = int(flat.size)
    if n == 0:
        return f"{output_label}: empty consensus grid"

    n_no_shift = int(np.count_nonzero(~np.isfinite(flat)))
    noise = int(np.count_nonzero(np.isfinite(flat) & (flat == -1)))
    # Two decimals: with .1f, sub-percent shares (e.g. 0.05%) round to "0.0%" and
    # look inconsistent with a nonzero (noise:,) count.
    pct_noise = 100.0 * noise / n
    pct_no_shift = 100.0 * n_no_shift / n
    pos = flat[np.isfinite(flat) & (flat >= 0)]
    n_clusters = int(np.unique(pos).size) if pos.size else 0

    clusters_text = f"{n_clusters} {'consensus cluster' if n_clusters == 1 else 'consensus clusters'}"
    return (
        f"New consensus variable \033[1m{output_label}\033[0m: Identified \033[1m{clusters_text}\033[0m "
        f"over {n:,} spacetime cells; {pct_noise:.2f}% shift noise / not in consensus ({noise:,} cells); "
        f"{pct_no_shift:.2f}% no abrupt shift ({n_no_shift:,} cells NaN)."
    )


def _filter_consensus_labels_min_size(
    da_labels: xr.DataArray,
    da_rate: xr.DataArray,
    min_cluster_area: int,
    *,
    time_dim: str,
    da_sign: xr.DataArray | None = None,
) -> (
    tuple[xr.DataArray, xr.DataArray] | tuple[xr.DataArray, xr.DataArray, xr.DataArray]
):
    """Remove consensus clusters whose spatial footprint is too small.

    For each cluster id, ``area`` is the number of spatial cells where that id appears at
    any time along ``time_dim`` (same definition as the ``area`` column in
    :func:`toad.utils.cluster_consensus_utils._build_consensus_summary_df_spacetime`).
    Clusters with ``area < min_cluster_area`` are set to noise ``-1``. Remaining clusters
    are renumbered by descending area (largest id 0, etc.), i.e. on the same quantity
    this filter thresholds. The companion rate field is left unchanged (member-support
    fractions are independent of the consensus threshold and cluster-size filter).
    """
    if min_cluster_area <= 0:
        if da_sign is None:
            return da_labels, da_rate
        return da_labels, da_rate, da_sign
    if time_dim not in da_labels.dims:
        raise ValueError(
            f"`time_dim` {time_dim!r} must be a dimension of `da_labels`, "
            f"got dims={tuple(da_labels.dims)}."
        )
    lab = np.asarray(da_labels.data, dtype=np.float64)
    time_axis = da_labels.get_axis_num(time_dim)
    lab_ts = np.moveaxis(lab, time_axis, 0).reshape(lab.shape[time_axis], -1)
    areas_by_id = cluster_spatial_areas(lab_ts)
    if not areas_by_id:
        if da_sign is None:
            return da_labels, da_rate
        return da_labels, da_rate, da_sign

    remove = [cid for cid, area in areas_by_id.items() if area < int(min_cluster_area)]
    if not remove:
        if da_sign is None:
            return da_labels, da_rate
        return da_labels, da_rate, da_sign

    # Demote small clusters to noise (-1) and re-rank survivors by area
    flat = lab.ravel().copy()
    fin = np.isfinite(flat)
    flat[fin & np.isin(flat, np.array(remove, dtype=np.float64))] = -1.0
    survivors = {
        cid: area for cid, area in areas_by_id.items() if cid not in set(remove)
    }
    flat = relabel_by_id_mapping(flat, area_rank_id_mapping(survivors))
    lab_out = flat.reshape(lab.shape)
    da_l = xr.DataArray(
        lab_out,
        coords=da_labels.coords,
        dims=da_labels.dims,
        attrs=da_labels.attrs,
        name=da_labels.name,
    )
    if da_sign is None:
        return da_l, da_rate

    sign_flat = np.asarray(da_sign.data, dtype=np.float32).ravel().copy()
    sign_flat[~np.isfinite(lab_out.ravel()) | (lab_out.ravel() < 0)] = np.nan
    da_s = xr.DataArray(
        sign_flat.reshape(lab_out.shape),
        coords=da_sign.coords,
        dims=da_sign.dims,
        attrs=da_sign.attrs,
        name=da_sign.name,
    )
    return da_l, da_rate, da_s


def _finalize_consensus_variables(
    td: Any,
    *,
    ds_out: xr.Dataset,
    new_output_label: str,
    cluster_vars: List[str],
    min_consensus: float,
    temporal_tolerance: TemporalTolerance,
    spatial_tolerance: SpatialTolerance,
    stitch_meridian: StitchMeridianSetting,
    stitch_meridian_resolved: bool,
    min_cluster_area: int | None,
    time_dim: str,
) -> np.ndarray:
    """Rename solver outputs, post-filter, attach TOAD attrs, and merge into ``td.data``."""
    # --- rename interim solver variables to user-facing names ---
    da_labels = ds_out["clusters"].rename(new_output_label)
    da_rate = ds_out["rate"].rename(f"{new_output_label}{_attrs.CONSENSUS_RATE_SUFFIX}")
    da_sign = None
    if "signs" in ds_out:
        da_sign = ds_out["signs"].rename(consensus_sign_var_name(new_output_label))

    for stale_key in (
        "_interim_sign_by_id",
        "consensus_cluster_signs",
        "cluster_signs",
    ):
        da_labels.attrs.pop(stale_key, None)
        da_rate.attrs.pop(stale_key, None)
        if da_sign is not None:
            da_sign.attrs.pop(stale_key, None)

    # --- optional post-filter on spatial footprint (see _filter_consensus_labels_min_size) ---
    if min_cluster_area is not None and min_cluster_area > 0:
        if da_sign is not None:
            da_labels, da_rate, da_sign = _filter_consensus_labels_min_size(
                da_labels,
                da_rate,
                min_cluster_area,
                time_dim=time_dim,
                da_sign=da_sign,
            )
        else:
            da_labels, da_rate = _filter_consensus_labels_min_size(
                da_labels,
                da_rate,
                min_cluster_area,
                time_dim=time_dim,
            )
        da_labels.attrs["min_cluster_area"] = int(min_cluster_area)
        da_rate.attrs["min_cluster_area"] = int(min_cluster_area)
        if da_sign is not None:
            da_sign.attrs["min_cluster_area"] = int(min_cluster_area)

    # --- TOAD metadata (method params, variable_type, cluster_vars, version) ---
    lab = np.asarray(da_labels.data, dtype=np.float64)
    min_votes = min_consensus_members(len(cluster_vars), min_consensus)
    consensus_param_attrs: dict[str, object] = {
        "consensus_method": "member_support",
        "min_consensus": min_consensus,
        "min_consensus_members": min_votes,
        "temporal_tolerance": temporal_tolerance_attr_value(temporal_tolerance),
        "spatial_tolerance": spatial_tolerance_attr_value(spatial_tolerance),
        "stitch_meridian": (
            int(stitch_meridian)
            if isinstance(stitch_meridian, bool)
            else stitch_meridian
        ),
        "stitch_meridian_applied": int(stitch_meridian_resolved),
    }
    da_labels.attrs.update(consensus_param_attrs)
    da_rate.attrs.update(consensus_param_attrs)
    da_labels.attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_CONSENSUS_CLUSTER
    u = np.unique(lab[np.isfinite(lab) & (lab >= 0)])
    da_labels.attrs[_attrs.CLUSTER_IDS] = (
        u.astype(int) if u.size else np.array([], dtype=int)
    )
    da_labels.attrs[_attrs.CLUSTER_VARS] = list(cluster_vars)
    da_labels.attrs[_attrs.TOAD_VERSION] = __version__

    da_rate.attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_CONSENSUS_RATE
    da_rate.attrs[_attrs.CONSENSUS_LABELS_VAR] = new_output_label
    da_rate.attrs[_attrs.CLUSTER_VARS] = list(cluster_vars)
    da_rate.attrs[_attrs.TOAD_VERSION] = __version__

    merge_vars = [da_labels, da_rate]
    if da_sign is not None:
        da_sign.attrs.update(consensus_param_attrs)
        da_sign.attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_CONSENSUS_SIGN
        da_sign.attrs[_attrs.CONSENSUS_LABELS_VAR] = new_output_label
        da_sign.attrs[_attrs.CLUSTER_VARS] = list(cluster_vars)
        da_sign.attrs[_attrs.TOAD_VERSION] = __version__
        merge_vars.append(da_sign)

    # --- merge label + rate (+ sign) into td.data ---
    td.data = xr.merge(
        [td.data, *merge_vars],
        combine_attrs="override",
        compat="override",
    )
    return lab


class Aggregation:
    """
    Aggregation methods for TOAD objects.
    """

    def __init__(self, toad):
        self.td = toad

    def cluster_occurrence_rate(
        self,
        cluster_vars: list[str] | None = None,
    ) -> xr.DataArray:
        """Share of input clusterings in which a grid cell was ever assigned a cluster.

        For each spatial cell and each label field, TOAD checks whether any timestep
        has a non-noise label (``>= 0``); multiple cluster events at different times
        in the same run still count as one ``yes``. The result is the mean over inputs,
        in ``[0, 1]`` (``1`` = every included clustering ever labelled that cell).

        This is a time-collapsed hotspot diagnostic, not spacetime consensus: it does
        not require agreement on timing or cluster id across inputs.

        Args:
            cluster_vars: Label variables to include. If None, uses all
                :attr:`TOAD.cluster_vars`. Each must use ``-1`` for noise and ``>= 0`` for
                cluster membership.

        Returns:
            2D DataArray named ``cluster_occurrence_rate`` (or a uniquified name).
        """
        cluster_vars = cluster_vars if cluster_vars else self.td.cluster_vars
        if not cluster_vars:
            raise ValueError(
                "cluster_vars is empty; add cluster label variables or pass a non-empty list."
            )

        ever_clustered = xr.concat(
            [
                (self.td.data[cvar].max(dim=self.td.time_dim) > -1).astype(np.float32)
                for cvar in cluster_vars
            ],
            dim="_cluster_input",
        )
        cluster_normalized = ever_clustered.mean(dim="_cluster_input")

        output_label = get_unique_variable_name(
            "cluster_occurrence_rate", self.td.data, self.td.logger
        )
        cluster_normalized = cluster_normalized.rename(output_label)
        cluster_normalized.attrs.update(
            {
                "cluster_vars": list(cluster_vars),
                "description": (
                    "Fraction of input clusterings that ever assigned a non-noise "
                    "label at this cell (time collapsed; 1 = all inputs)"
                ),
            }
        )
        return cluster_normalized

    def compute_consensus(
        self,
        cluster_vars: List[str] | None = None,
        *,
        min_consensus: float,
        temporal_tolerance: TemporalTolerance,
        spatial_tolerance: SpatialTolerance,
        stitch_meridian: StitchMeridianSetting = "auto",
        show_progress: bool = True,
        output_label_suffix: str = "",
        output_label: str | None = None,
        overwrite: bool = False,
        min_cluster_area: int | None = 2,
    ) -> None:
        """Merge several input cluster maps into one spacetime consensus field on ``self.td.data``.

        Use this when you have multiple cluster label fields on the same time × space grid
        (different models, parameters, or variables) and want regions where enough inputs
        agree that an abrupt shift occurred, within chosen time and space windows.

        **Algorithm**

        1. For each input, build a mask of **native event voxels** (non-noise cluster labels,
           ``>= 0``).
        2. Dilate each mask in ``(time, y, x)`` by ``temporal_tolerance`` and
           ``spatial_tolerance`` for **support counting only**.
        3. At each native event voxel, count how many distinct inputs have dilated support
           covering that cell.
        4. Retain the voxel if the count reaches ``max(1, ceil(min_consensus * n_inputs))``.
           When inputs carry cluster signs, votes are counted separately for ``+`` and
           ``−``, and a voxel is retained for a sign only if a **native event of that
           sign** is present (opposite-sign support cannot promote it).
        5. Group retained voxels into consensus cluster ids using the same tolerances for
           spacetime connectivity (``max(1, tolerance)`` along each axis). Ids are numbered
           by descending **spatial footprint**, so cluster ``0`` is always the largest on a
           map — the same quantity ``min_cluster_area`` thresholds.
        6. Optionally drop clusters whose spatial footprint is below ``min_cluster_area``;
           remaining ids are re-ranked; the companion sign grid is cleared at removed voxels.

        Dilation never writes extra cells to the output: only voxels that were **detected** in
        at least one input can appear in the consensus mask. Spatial tolerance is in native
        grid indices, not kilometres. With ``stitch_meridian``, the first and last longitude
        column can be treated as neighbours during dilation and labelling on global grids.

        **Writes**

        Two variables are merged into ``self.td.data`` (default names ``cluster_consensus`` and
        ``cluster_consensus_rate``):

        * **Labels** (``variable_type=consensus_cluster``): ``NaN`` if every input has no
          abrupt shift at that cell; ``-1`` if at least one input had a defined label but the
          cell is not in consensus (or was filtered out); non-negative integers are consensus
          cluster ids.
        * **Rate** (``variable_type=consensus_rate``): supporting inputs divided
          by total inputs at each native event voxel, **including** voxels below the consensus
          threshold; ``0`` where no input assigned a cluster; ``NaN`` where the label is
          ``NaN``. With signed inputs the support is counted for the sign of the native
          event at that voxel (the stronger sign where a voxel has both), so the rate never
          credits a voxel with opposite-sign support that the keep rule ignored.

        Both arrays store ``cluster_vars``, ``min_consensus``, ``min_consensus_members``,
        tolerance settings, and ``stitch_meridian`` / ``stitch_meridian_applied``. For a
        per-cluster table after the fact, call :meth:`consensus_summary`.

        Args:
            cluster_vars: Input clustering variables to include. If None, uses all
                ``self.td.cluster_vars``.
            min_consensus: Fraction in ``[0, 1]`` of inputs required per retained voxel after
                dilation. Required; no default.
            temporal_tolerance: Time-step radius for support dilation and cluster connectivity.
                Required; ``0`` means exact-time support only. ``np.inf`` means unlimited
                temporal reach (spatial agreement only). Cannot be combined with
                ``spatial_tolerance=np.inf``.
            spatial_tolerance: Grid-cell radius in ``y/x`` for support dilation and cluster
                connectivity. Required; ``0`` means exact spatial support only;
                ``np.inf`` means unlimited spatial reach (temporal agreement only).
            stitch_meridian: ``False``, ``True``, or ``\"auto\"`` (default). ``\"auto\"`` enables
                seam stitching when the grid spans nearly all longitudes.
            show_progress: Show a progress bar while processing inputs. Default: True.
            output_label_suffix: Suffix for the default label name ``cluster_consensus``.
            output_label: Explicit consensus labels variable name. If None, uses
                ``\"cluster_consensus\" + output_label_suffix``.
            overwrite: Replace existing variables with the same names. If False, uniquify
                names like :func:`toad.clustering.compute_clusters`.
            min_cluster_area: Drop consensus clusters whose spatial footprint (cells at any
                time) is strictly below this value; ids are re-sorted afterward. Default ``2``.
                ``None`` disables the filter.

        See Also:
            :doc:`consensus_clustering` for a longer guide and :doc:`Consensus tutorial
            <tutorials/consensus>` for a worked example.

        Example:
            >>> td.compute_consensus(
            ...     cluster_vars=['var_dts_cluster', 'var_dts_cluster_1'],
            ...     min_consensus=0.7,
            ...     temporal_tolerance=5,
            ...     spatial_tolerance=1,
            ...     min_cluster_area=10,
            ... )
            >>> td.plot.consensus_overview()
            >>> td.aggregate.consensus_summary().head()

        Raises:
            ValueError: If no cluster variables are found, a tolerance is negative,
            both tolerances are ``np.inf``, ``min_consensus`` is outside ``[0, 1]``,
            or ``min_cluster_area`` is invalid.

        """
        # --- inputs and parameters ---
        if cluster_vars is None:
            cluster_vars = list(self.td.cluster_vars)
        if len(cluster_vars) == 0:
            raise ValueError("No cluster variables provided/found.")

        temporal_tolerance, spatial_tolerance = validate_consensus_tolerances(
            temporal_tolerance, spatial_tolerance
        )
        if not (0.0 <= min_consensus <= 1.0):
            raise ValueError(f"`min_consensus` must be in [0, 1], got {min_consensus}.")
        if min_cluster_area is not None and min_cluster_area < 0:
            raise ValueError(
                f"`min_cluster_area` must be >= 0 or None, got {min_cluster_area}."
            )

        # --- grid layout (meridian stitching for labelling / dilation on global lon grids) ---
        spatial_dims = tuple(self.td.space_dims)
        stitch_meridian_resolved = resolve_stitch_meridian(
            stitch_meridian,
            dataset=self.td.data,
            spatial_dims=spatial_dims,
        )
        if stitch_meridian == "auto" and stitch_meridian_resolved:
            self.td.logger.info(
                "Meridian seam stitching enabled automatically (global longitude grid)."
            )

        # --- output variable names ---
        new_output_label = (
            output_label if output_label else f"cluster_consensus{output_label_suffix}"
        )
        if not overwrite:
            new_output_label = get_unique_variable_name(
                new_output_label, self.td.data, logger
            )
        else:
            if new_output_label in self.td.data:
                self.td.data = self.td.data.drop_vars(new_output_label)
            rate_drop = f"{new_output_label}{_attrs.CONSENSUS_RATE_SUFFIX}"
            if rate_drop in self.td.data:
                self.td.data = self.td.data.drop_vars(rate_drop)
            sign_drop = consensus_sign_var_name(new_output_label)
            if sign_drop in self.td.data:
                self.td.data = self.td.data.drop_vars(sign_drop)

        # --- member-support solver: dilated votes → threshold → connected components ---
        sample = self.td.data[cluster_vars[0]]
        context = _build_grid_context(
            sample,
            spatial_dims=spatial_dims,
            time_dim=self.td.time_dim,
            stitch_meridian=stitch_meridian_resolved,
        )
        support = _accumulate_member_support(
            self.td,
            cluster_vars=cluster_vars,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            show_progress=show_progress,
            context=context,
        )
        if not np.any(support.native_union):
            ds_out = _empty_result(self.td, cluster_vars, context)
        else:
            ds_out = _build_member_support_dataset(
                self.td,
                cluster_vars=cluster_vars,
                min_consensus=min_consensus,
                temporal_tolerance=temporal_tolerance,
                spatial_tolerance=spatial_tolerance,
                context=context,
                support=support,
            )

        # --- optional size filter, TOAD attrs, merge into td.data ---
        lab = _finalize_consensus_variables(
            self.td,
            ds_out=ds_out,
            new_output_label=new_output_label,
            cluster_vars=cluster_vars,
            min_consensus=min_consensus,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            stitch_meridian=stitch_meridian,
            stitch_meridian_resolved=stitch_meridian_resolved,
            min_cluster_area=min_cluster_area,
            time_dim=self.td.time_dim,
        )

        logger.info(_format_consensus_summary(new_output_label, lab))

    def consensus_summary(self, consensus_var: str | None = None) -> pd.DataFrame:
        """Rebuild the per-cluster summary table from stored consensus label and rate arrays.

        Shift-time columns use strict same-``(time, y, x)`` agreement between consensus
        and each input (see :func:`toad.utils.cluster_consensus_utils.consensus_shift_time_distribution`).
        To plot full base-variable trajectories over a shared spatial region with a looser
        time rule, use :meth:`consensus_cluster_timeseries` instead.

        Args:
            consensus_var: Name of the consensus labels variable (``variable_type=consensus_cluster``).
                If None, infers the variable when exactly one consensus label exists on the dataset
                (same resolution rules as :meth:`toad.core.TOAD._resolve_consensus_var`).

        Returns:
            DataFrame with one row per consensus cluster. Includes ``mean_consensus_rate``,
            spatial ``area`` and centroid columns, ``median_median_shift_time`` (median of
            per-input spatial medians), related between-input std columns, and
            ``pooled_median_shift_time`` / ``pooled_std_shift_time`` over all pooled event-time
            samples; see :func:`toad.utils.cluster_consensus_utils._build_consensus_summary_df_spacetime`.
        """
        consensus_var = self.td._resolve_consensus_var(consensus_var)
        rate_var = self.td._resolve_consensus_rate_var(consensus_var)
        labels = self.td.data[consensus_var]
        rate = self.td.data[rate_var]
        spatial_dims = tuple(self.td.space_dims)
        time_dim = self.td.time_dim
        return _build_consensus_summary_df_spacetime(
            self.td, labels, rate, spatial_dims, time_dim
        )

    def consensus_shift_time_distribution(
        self,
        da_clusters: xr.DataArray,
        *,
        spatial_dims: tuple[str, str] | None = None,
        time_dim: str | None = None,
    ) -> tuple[xr.Dataset, pd.DataFrame]:
        """Export event-time samples behind the consensus summary shift columns.

        Wraps :func:`toad.utils.cluster_consensus_utils.consensus_shift_time_distribution`.
        Returns an ``xr.Dataset`` (per-cluster × ``cluster_var`` means/stds) and a long
        ``DataFrame`` of per-cell transition times for histograms.

        Args:
            da_clusters: Consensus labels from :meth:`compute_consensus` / :meth:`TOAD.compute_consensus`.
            spatial_dims: Defaults to ``self.td.space_dims``.
            time_dim: Inferred from ``self.td.time_dim`` when present on ``da_clusters``.

        Returns:
            ``(xr.Dataset, pandas.DataFrame)`` — see the utility docstring.
        """
        return consensus_shift_time_distribution(
            self.td,
            da_clusters,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )

    def consensus_shift_time_distributions(
        self,
        da_clusters: xr.DataArray,
        *,
        spatial_dims: tuple[str, str] | None = None,
        time_dim: str | None = None,
        distribution_result: tuple[xr.Dataset, pd.DataFrame] | None = None,
        source_input_cluster_var: str | None = None,
    ) -> dict[int, np.ndarray]:
        """Return transition-time samples grouped by consensus cluster id.

        This is a plotting-friendly wrapper around
        :meth:`consensus_shift_time_distribution`. It aggregates across all input
        ``cluster_var`` values and returns ``{cluster_id: shift_times}``, which is
        convenient for violin plots of consensus-cluster timing distributions.

        Args:
            da_clusters: Consensus labels from :meth:`compute_consensus` / :meth:`TOAD.compute_consensus`.
            spatial_dims: Defaults to ``self.td.space_dims``.
            time_dim: Inferred from ``self.td.time_dim`` when present on ``da_clusters``.
            distribution_result: If provided, a prior ``(dataset, dataframe)`` tuple from
                :meth:`consensus_shift_time_distribution` for the same inputs; recomputation
                of the long table is skipped.
            source_input_cluster_var: If set, use only long-form rows from this input
                clustering (see :func:`toad.utils.cluster_consensus_utils.consensus_shift_time_distributions`).

        Returns:
            Mapping from consensus cluster id to a 1D array of transition times.
            In spacetime mode the samples match the summary pipeline exactly, so the
            same spatial cell may appear multiple times if it belongs to the same
            consensus component at multiple timesteps.
        """
        return consensus_shift_time_distributions(
            self.td,
            da_clusters,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
            distribution_result=distribution_result,
            source_input_cluster_var=source_input_cluster_var,
        )

    def label_shift_time_samples(
        self,
        label_data: str | xr.DataArray,
        *,
        spatial_dims: tuple[str, str] | None = None,
        time_dim: str | None = None,
    ) -> pd.DataFrame:
        """Per-cell event times for a single 3D label field (non-consensus or any labels).

        Wraps :func:`toad.utils.cluster_consensus_utils.label_field_shift_time_samples`.
        Pass a variable name in ``self.td.data`` or a :class:`xarray.DataArray` of labels.
        """
        da = self.td.data[label_data] if isinstance(label_data, str) else label_data
        return label_field_shift_time_samples(
            self.td,
            da,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )

    def label_shift_time_distributions(
        self,
        label_data: str | xr.DataArray,
        *,
        spatial_dims: tuple[str, str] | None = None,
        time_dim: str | None = None,
    ) -> dict[int, np.ndarray]:
        """Transition-time samples per cluster id in one time-resolved label field.

        For violin-style plots of a **normal** clustering, pass that cluster map’s
        name. Same convention as the consensus long table (event time at each
        labelled spacetime cell).

        See :func:`toad.utils.cluster_consensus_utils.label_field_shift_time_distributions`.
        """
        da = self.td.data[label_data] if isinstance(label_data, str) else label_data
        return label_field_shift_time_distributions(
            self.td,
            da,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )

    def consensus_extraction_mask_2d(
        self,
        da_clusters: xr.DataArray,
        consensus_cluster_id: int,
        cluster_var: str,
    ) -> xr.DataArray:
        """2D boolean mask of grid cells used by :meth:`consensus_cluster_timeseries`.

        A cell is True iff, at **some** timestep, the consensus label equals
        ``consensus_cluster_id`` **and** the input ``cluster_var`` has a non-noise label
        (``>= 0``). Those conditions need not hold at the **same** time — the mask is
        collapsed with OR over time, then used to extract **full** time series on the
        resulting spatial footprint.

        This is looser than the support rule in :meth:`consensus_summary` /
        :func:`~toad.utils.cluster_consensus_utils.consensus_shift_time_distribution`,
        which require consensus and input to agree at the same ``(time, y, x)`` when
        computing shift-time statistics.

        Args:
            da_clusters: Consensus label field.
            consensus_cluster_id: Target consensus cluster id.
            cluster_var: One input clustering variable name.

        Returns:
            DataArray of bool with dimensions ``self.td.space_dims``.
        """
        cluster_mask = da_clusters == consensus_cluster_id
        time_dim = self.td.time_dim
        has_time_dim = time_dim in da_clusters.dims
        support_mask = _consensus_input_support_mask(
            self.td,
            da_clusters,
            cluster_var,
            spatial_dims=tuple(self.td.space_dims),
            time_dim=time_dim if has_time_dim else None,
        )
        # OR over time: consensus and input need not agree at the same timestep
        if has_time_dim:
            return (cluster_mask & support_mask).any(dim=time_dim)
        return cluster_mask & support_mask

    def consensus_cluster_timeseries(
        self,
        da_clusters: xr.DataArray,
        consensus_cluster_id: int,
        *,
        var: str | None = None,
        cluster_vars: list[str] | None = None,
        aggregation: Literal[
            "raw", "mean", "sum", "std", "median", "percentile", "max", "min"
        ]
        | str = "raw",
        percentile: float | None = None,
        normalize: Literal["max", "max_each"] | str | None = None,
        keep_full_timeseries: bool = True,
    ) -> dict[str, xr.DataArray]:
        """Extract per-input timeseries for one consensus cluster.

        Uses :meth:`consensus_extraction_mask_2d` — a **spatial** footprint where consensus
        and each input were active at least once (not necessarily at the same time), then
        aggregates the base variable over those cells for the full time axis (see
        ``keep_full_timeseries``). Convenient for overlaying trajectories from the same
        region across inputs.

        For *when* shifts occurred, use :meth:`consensus_summary` or
        :meth:`consensus_shift_time_distribution` instead; those require consensus and
        input labels at the same ``(time, y, x)`` voxel.

        Args:
            da_clusters: Consensus labels from :meth:`compute_consensus` / :meth:`TOAD.compute_consensus`.
            consensus_cluster_id: Consensus cluster id to extract.
            var: Optional base variable override used for all returned timeseries.
                If omitted, each ``cluster_var`` is mapped to its own base variable.
            cluster_vars: Optional subset of input clustering variables. Defaults to
                ``da_clusters.attrs["cluster_vars"]`` when available, else ``self.td.cluster_vars``.
            aggregation: Same as :meth:`toad.core.TOAD.get_timeseries`.
            percentile: Required when ``aggregation="percentile"``.
            normalize: Same as :meth:`toad.core.TOAD.get_timeseries`.
            keep_full_timeseries: If ``False`` and ``da_clusters`` is time-resolved,
                values outside the consensus cluster's overall start/end window are
                masked out.

        Returns:
            Mapping ``{cluster_var: xr.DataArray}`` for input clusterings with a
            non-empty footprint. Each series matches :meth:`toad.core.TOAD.get_timeseries`
            for the chosen ``aggregation``.
        """
        if cluster_vars is None:
            cluster_vars_attr = da_clusters.attrs.get("cluster_vars")
            if cluster_vars_attr is not None:
                cluster_vars = list(cluster_vars_attr)
            else:
                cluster_vars = list(self.td.cluster_vars)
        if len(cluster_vars) == 0:
            raise ValueError("No cluster variables available for consensus timeseries.")

        cluster_mask = da_clusters == consensus_cluster_id
        if not bool(cluster_mask.any().item()):
            present_ids = np.unique(da_clusters.values)
            present_ids = present_ids[np.isfinite(present_ids) & (present_ids >= 0)]
            raise ValueError(
                f"Consensus cluster id {consensus_cluster_id} not found. "
                f"Available ids: {present_ids.astype(int).tolist()}"
            )

        time_dim = self.td.time_dim
        has_time_dim = time_dim in da_clusters.dims
        time_window_mask: xr.DataArray | None = None
        if has_time_dim and not keep_full_timeseries:
            cluster_present_over_time = cluster_mask.any(dim=self.td.space_dims)
            active_idx = np.flatnonzero(cluster_present_over_time.values)
            if active_idx.size > 0:
                in_window = np.zeros(da_clusters.sizes[time_dim], dtype=bool)
                in_window[active_idx[0] : active_idx[-1] + 1] = True
                time_window_mask = xr.DataArray(
                    in_window,
                    dims=[time_dim],
                    coords={time_dim: da_clusters[time_dim]},
                    name="consensus_time_window",
                )

        out: dict[str, xr.DataArray] = {}
        for cluster_var in cluster_vars:
            mask_for_data = self.consensus_extraction_mask_2d(
                da_clusters, consensus_cluster_id, cluster_var
            )
            if not bool(mask_for_data.any().item()):
                continue

            base_var = var
            if base_var is None:
                base_var = self.td.data[cluster_var].attrs.get(_attrs.BASE_VARIABLE)
                if base_var is None:
                    raise ValueError(
                        f"Cluster variable '{cluster_var}' has no BASE_VARIABLE attribute. "
                        "Pass `var=...` explicitly."
                    )

            data = self.td.data[base_var].where(mask_for_data)
            if time_window_mask is not None:
                data = data.where(time_window_mask)

            series = self.td._finalize_timeseries(
                data, aggregation, percentile, normalize
            )
            if not np.isfinite(np.asarray(series.values, dtype=float)).any():
                continue

            series = series.copy()
            series.name = f"{cluster_var}_consensus_cluster_{consensus_cluster_id}"
            series.attrs.update(
                {
                    "consensus_cluster_id": int(consensus_cluster_id),
                    "cluster_var": cluster_var,
                    "base_var": base_var,
                    "aggregation": aggregation,
                    "keep_full_timeseries": keep_full_timeseries,
                }
            )
            out[cluster_var] = series

        if not out:
            raise ValueError(
                f"No input clusterings support consensus cluster id {consensus_cluster_id} "
                "under the current extraction settings."
            )
        return out
