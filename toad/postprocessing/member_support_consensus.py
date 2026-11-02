"""Per-voxel member-support spacetime consensus.

Pipeline (read top to bottom in this module):

1. :func:`_accumulate_member_support` — dilated vote counts per native event voxel
2. :func:`_build_member_support_dataset` — threshold, label, assemble xarray output
3. Connectivity helpers — group retained voxels into consensus cluster ids

:meth:`~toad.postprocessing.aggregation.Aggregation.compute_consensus` and
:meth:`~toad.MMA.run_consensus` orchestrate these steps (grid context, empty result,
finalize attrs on ``td.data``).

Native **8-neighbour** grid edges (:func:`native_spatial_edges`) are **not** left over
from the old edge-vote (EAC) algorithm. They are used only when ``stitch_meridian=True``
to build spatial adjacency for connected-component labelling across the longitude seam.
Support counting uses box dilation (:func:`_dilate_boolean_support_mask`); non-stitch
labelling uses ``scipy.ndimage.label`` or a KD-tree Chebyshev graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from tqdm import tqdm

from toad.clustering import sorted_cluster_labels
from toad.utils import _attrs

# ``spatial_tolerance=np.inf``: unlimited spatial reach (temporal-only support /
# connectivity). Finite values are hop / grid-cell radii; ``0`` = exact cell.
SpatialTolerance = int | float


def is_infinite_spatial_tolerance(spatial_tolerance: SpatialTolerance | str) -> bool:
    """True when ``spatial_tolerance`` requests unlimited spatial reach."""
    if isinstance(spatial_tolerance, str):
        return spatial_tolerance.strip().lower() in {"inf", "+inf", "infinity"}
    try:
        return bool(np.isinf(spatial_tolerance)) and float(spatial_tolerance) > 0
    except (TypeError, ValueError):
        return False


def validate_spatial_tolerance(
    spatial_tolerance: SpatialTolerance | str,
) -> SpatialTolerance:
    """Normalise ``spatial_tolerance``; allow non-negative ints or ``+inf``.

    Returns:
        ``np.inf`` for unlimited spatial reach, otherwise a non-negative ``int``.

    Raises:
        ValueError: If the value is negative or not a number.
    """
    if is_infinite_spatial_tolerance(spatial_tolerance):
        return np.inf
    try:
        # Reject plain bools (subclass of int) and non-numeric input.
        if isinstance(spatial_tolerance, bool):
            raise TypeError
        value = int(spatial_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`spatial_tolerance` must be a non-negative int or np.inf, "
            f"got {spatial_tolerance!r}."
        ) from exc
    if value < 0:
        raise ValueError(
            f"`spatial_tolerance` must be >= 0 or np.inf, got {spatial_tolerance}."
        )
    return value


def spatial_tolerance_attr_value(spatial_tolerance: SpatialTolerance) -> float | int:
    """Value stored on consensus variable attrs (NetCDF-friendly)."""
    if is_infinite_spatial_tolerance(spatial_tolerance):
        return float("inf")
    return int(spatial_tolerance)


def _temporal_only_component_edges(
    *,
    keep: np.ndarray,
    T: int,
    n_space: int,
    temporal_tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Connect kept voxels by time only (unlimited spatial reach).

    Within each timestep all kept voxels form one component (star edges). Timesteps
    within ``max(1, temporal_tolerance)`` of each other are bridged via their
    representatives — enough for undirected connected components.
    """
    del T  # layout size unused; kept_t already bounds-checked by construction
    kept_nodes = np.flatnonzero(keep).astype(np.int64, copy=False)
    empty = np.array([], dtype=np.int64)
    if kept_nodes.size == 0:
        return empty, empty, empty

    connect_t = max(1, int(temporal_tolerance))
    kept_t = kept_nodes // n_space
    first_pos: dict[int, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for i, t in enumerate(kept_t.tolist()):
        t_i = int(t)
        if t_i not in first_pos:
            first_pos[t_i] = i
        else:
            rows.append(first_pos[t_i])
            cols.append(i)

    times_present = sorted(first_pos)
    for i, t in enumerate(times_present):
        for t2 in times_present[i + 1 :]:
            if t2 - t > connect_t:
                break
            a = first_pos[t]
            b = first_pos[t2]
            rows.append(min(a, b))
            cols.append(max(a, b))

    if not rows:
        return kept_nodes, empty, empty
    return (
        kept_nodes,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
    )


def sign_var_for_cluster_var(cluster_var: str) -> str:
    """Companion per-voxel sign field for a cluster labels variable."""
    if cluster_var == "cluster":
        return "cluster_sign"
    if cluster_var.endswith("_cluster"):
        return f"{cluster_var}_sign"
    return f"{cluster_var}_sign"


def consensus_sign_var_name(consensus_var: str) -> str:
    """Companion sign grid name for a consensus labels variable."""
    return f"{consensus_var}{_attrs.CONSENSUS_SIGN_SUFFIX}"


def cluster_signs_flat_for_var(td: Any, cluster_var: str) -> np.ndarray | None:
    """Return flattened per-voxel signs for ``cluster_var``, if present."""
    sign_var = sign_var_for_cluster_var(cluster_var)
    if sign_var not in td.data:
        return None
    return np.asarray(td.data[sign_var].values, dtype=np.float64).ravel()


def has_sign_aware_inputs(td: Any, cluster_vars: list[str]) -> bool:
    """True when every cluster input has a companion sign grid."""
    if not cluster_vars:
        return False
    return all(sign_var_for_cluster_var(cvar) in td.data for cvar in cluster_vars)


def cluster_label_id_mapping(
    labels_before: np.ndarray,
    labels_after: np.ndarray,
) -> dict[int, int]:
    """Build ``{old_id: new_id}`` for a relabelling that only renumbers or drops ids.

    Ids absent from the result (e.g. demoted to noise by a size filter) are omitted.

    Raises:
        ValueError: if the two arrays differ in size, or if one old id maps to more
            than one new id (i.e. the relabelling split a cluster).
    """
    before = np.asarray(labels_before, dtype=np.float64).ravel()
    after = np.asarray(labels_after, dtype=np.float64).ravel()
    if before.size != after.size:
        raise ValueError(
            f"Label arrays must have the same size, got {before.size} and {after.size}."
        )
    paired = np.isfinite(before) & (before >= 0) & np.isfinite(after) & (after >= 0)
    if not np.any(paired):
        return {}
    pairs = np.unique(
        np.column_stack(
            (before[paired].astype(np.int64), after[paired].astype(np.int64))
        ),
        axis=0,
    )
    mapping: dict[int, int] = {}
    for old, new in pairs:
        old_id, new_id = int(old), int(new)
        prev = mapping.get(old_id)
        if prev is not None and prev != new_id:
            raise ValueError(
                f"Cluster {old_id} maps to both {prev} and {new_id}; "
                "expected a relabelling that only renumbers or drops ids."
            )
        mapping[old_id] = new_id
    return mapping


def cluster_spatial_areas(labels_ts: np.ndarray) -> dict[int, int]:
    """``{cluster_id: area}`` where area is distinct spatial cells occupied at any time.

    ``labels_ts`` must be ``(time, space)``. This is the same quantity that
    ``min_cluster_area`` thresholds and that the ``area`` column of
    :func:`toad.utils.cluster_consensus_utils._build_consensus_summary_df_spacetime`
    reports, so filtering and ranking stay on one definition.
    """
    valid = np.isfinite(labels_ts) & (labels_ts >= 0)
    if not np.any(valid):
        return {}
    ids = labels_ts[valid].astype(np.int64, copy=False)
    spatial = np.broadcast_to(
        np.arange(labels_ts.shape[1], dtype=np.int64).reshape(1, -1),
        labels_ts.shape,
    )[valid]
    pairs = np.unique(np.column_stack((ids, spatial)), axis=0)
    unique_ids, areas = np.unique(pairs[:, 0], return_counts=True)
    return {int(cid): int(area) for cid, area in zip(unique_ids, areas)}


def area_rank_id_mapping(areas_by_id: dict[int, int]) -> dict[int, int]:
    """``{old_id: new_id}`` ranking ids by descending area; ties broken by old id."""
    order = sorted(areas_by_id, key=lambda cid: (-areas_by_id[cid], cid))
    return {old_id: new_id for new_id, old_id in enumerate(order)}


def relabel_by_id_mapping(
    labels: np.ndarray,
    id_mapping: dict[int, int],
) -> np.ndarray:
    """Apply ``{old_id: new_id}``, leaving noise (``-1``) and non-finite values as-is."""
    out = np.array(labels, copy=True)
    if not id_mapping:
        return out
    lut = np.arange(max(id_mapping) + 1, dtype=np.int64)
    for old_id, new_id in id_mapping.items():
        lut[old_id] = new_id
    flat = out.reshape(-1)
    valid = np.isfinite(flat) & (flat >= 0)
    if np.any(valid):
        flat[valid] = lut[flat[valid].astype(np.int64, copy=False)]
    return flat.reshape(out.shape)


def sorted_consensus_labels_by_area(
    labels_flat: np.ndarray,
    *,
    n_space: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Renumber consensus ids by descending spatial footprint (largest is ``0``).

    ``labels_flat`` is a time-major flat ``(time, space)`` label array. Ranking by area
    rather than spacetime voxel count means id ``0`` is the largest cluster *on a map*,
    consistent with the ``min_cluster_area`` parameter. A long-lived cluster covering
    few cells therefore no longer outranks a short-lived but widespread one.

    Returns the relabelled array and the ``{old_id: new_id}`` mapping.
    """
    if n_space <= 0:
        raise ValueError(f"`n_space` must be >= 1, got {n_space}.")
    flat = np.asarray(labels_flat)
    if flat.size % n_space:
        raise ValueError(
            f"Label array of size {flat.size} is not a whole number of "
            f"{n_space}-cell timesteps."
        )
    mapping = area_rank_id_mapping(cluster_spatial_areas(flat.reshape(-1, n_space)))
    return relabel_by_id_mapping(flat, mapping), mapping


def _label_retained_voxels_with_offset(
    keep: np.ndarray,
    *,
    label_fn,
    label_offset: int = 0,
) -> tuple[np.ndarray, int]:
    """Label one sign-specific retained mask."""
    n_st = keep.size
    labels_flat = np.full(n_st, -1, dtype=np.int64)
    if not np.any(keep):
        return labels_flat, label_offset

    labeled = label_fn(keep)
    labeled = sorted_cluster_labels(labeled)
    assigned = labeled >= 0
    if not np.any(assigned):
        return labels_flat, label_offset

    if label_offset > 0:
        labeled = labeled.copy()
        labeled[assigned] += label_offset
    labels_flat[assigned] = labeled[assigned]
    next_offset = int(labeled[assigned].max()) + 1
    return labels_flat, next_offset


@dataclass(frozen=True)
class MemberSupport:
    """Native event voxels plus dilated support votes, accumulated over members.

    ``native_pos`` / ``native_neg`` are the sign-split native event masks and are set
    only when ``sign_aware``. They must stay separate from ``native_union``: retaining a
    voxel as positive consensus requires a native *positive* event there, so a negative
    event that happens to sit under positive support is not a positive consensus voxel.
    """

    native_union: np.ndarray
    votes_primary: np.ndarray
    votes_secondary: np.ndarray | None
    sign_aware: bool
    native_pos: np.ndarray | None = None
    native_neg: np.ndarray | None = None


def build_sign_aware_consensus_labels(
    *,
    native_union: np.ndarray,
    votes_pos: np.ndarray,
    votes_neg: np.ndarray,
    n_members: int,
    min_consensus: float,
    n_st: int,
    n_space: int,
    label_fn,
    sign_aware: bool,
    votes_any: np.ndarray | None = None,
    native_pos: np.ndarray | None = None,
    native_neg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Threshold sign-split votes, label components, and build member-support rate.

    ``native_pos`` / ``native_neg`` are the sign-split native event masks and are
    required when ``sign_aware``; ``native_union`` alone cannot decide which sign a
    voxel is eligible for. ``native_union`` still sets where the member-support rate is
    reported, namely at every native event voxel.

    Returned ids are numbered by descending spatial footprint (largest is ``0``) across
    both signs. The third return value is a per-voxel sign grid (``±1`` where labelled,
    NaN elsewhere).
    """
    if sign_aware and (native_pos is None or native_neg is None):
        raise ValueError(
            "Sign-aware consensus requires `native_pos` and `native_neg`; "
            "thresholding against `native_union` would retain voxels under "
            "opposite-sign support."
        )
    min_votes = min_consensus_members(n_members, min_consensus)
    labels_flat = np.full(n_st, -1, dtype=np.int64)
    rate_flat = np.zeros(n_st, dtype=np.float32)
    signs_flat = np.full(n_st, np.nan, dtype=np.float32)

    if not sign_aware:
        vote_arr = votes_any if votes_any is not None else votes_pos
        keep = native_union & (vote_arr >= min_votes)
        if np.any(native_union):
            rate_flat[native_union] = (
                vote_arr[native_union].astype(np.float32) / n_members
            )
        if np.any(keep):
            labels_flat = sorted_cluster_labels(label_fn(keep))
            labels_flat, _ = sorted_consensus_labels_by_area(
                labels_flat, n_space=n_space
            )
        return labels_flat, rate_flat, signs_flat

    native_pos = cast(np.ndarray, native_pos)
    native_neg = cast(np.ndarray, native_neg)
    keep_pos = native_pos & (votes_pos >= min_votes)
    keep_neg = native_neg & (votes_neg >= min_votes)

    # Credit each native voxel with the votes for the sign(s) it actually has natively.
    # The keep rule requires a native event of the matching sign, so reporting
    # opposite-sign votes here would advertise support the consensus test never used --
    # e.g. a native decrease drawn at 5/8 on a rate map while (correctly) absent from
    # the cluster map. Only voxels with both signs natively take the stronger of the two.
    if np.any(native_union):
        votes_native = np.where(native_pos, votes_pos, votes_neg)
        both_native = native_pos & native_neg
        if np.any(both_native):
            votes_native = np.where(
                both_native, np.maximum(votes_pos, votes_neg), votes_native
            )
        rate_flat[native_union] = (
            votes_native[native_union].astype(np.float32) / n_members
        )

    both = keep_pos & keep_neg
    keep_pos_only = keep_pos & ~(both & (votes_neg > votes_pos))
    keep_neg_only = keep_neg & ~(both & (votes_pos >= votes_neg))

    offset = 0
    pos_labels, offset = _label_retained_voxels_with_offset(
        keep_pos_only,
        label_fn=label_fn,
        label_offset=0,
    )
    assigned = pos_labels >= 0
    labels_flat[assigned] = pos_labels[assigned]
    signs_flat[assigned] = 1.0

    neg_labels, _ = _label_retained_voxels_with_offset(
        keep_neg_only,
        label_fn=label_fn,
        label_offset=offset,
    )
    assigned_neg = neg_labels >= 0
    labels_flat[assigned_neg] = neg_labels[assigned_neg]
    signs_flat[assigned_neg] = -1.0

    # Each sign block is ranked on its own and positives are labelled first, so rank
    # across both blocks to keep the guarantee that id 0 is the largest cluster.
    # Must not wait for the optional min_cluster_area filter, which may not run.
    if np.any(labels_flat >= 0):
        labels_flat, _ = sorted_consensus_labels_by_area(labels_flat, n_space=n_space)
        labels_flat = np.asarray(labels_flat, dtype=np.int64)

    # Retained voxels report the votes for their own cluster's sign, which differs from
    # the native-sign rate above only where a voxel has both signs natively.
    labelled = labels_flat >= 0
    if np.any(labelled):
        chosen = np.where(
            signs_flat[labelled] > 0, votes_pos[labelled], votes_neg[labelled]
        )
        rate_flat[labelled] = chosen.astype(np.float32) / n_members

    return labels_flat, rate_flat, signs_flat


StitchMeridianSetting = bool | Literal["auto"]


def min_consensus_members(n_inputs: int, min_consensus: float) -> int:
    """Minimum distinct inputs required to retain a native event voxel."""
    if n_inputs < 1:
        raise ValueError(f"`n_inputs` must be >= 1, got {n_inputs}.")
    if not (0.0 <= min_consensus <= 1.0):
        raise ValueError(f"`min_consensus` must be in [0, 1], got {min_consensus}.")
    return max(1, int(np.ceil(float(min_consensus) * n_inputs)))


@dataclass(frozen=True)
class SpacetimeGridContext:
    """Fixed grid layout for one member-support consensus run."""

    spatial_dims: tuple[str, str]
    time_dim: str
    T: int
    y_len: int
    x_len: int
    coords_spatial: dict[str, Any]
    time_coord: xr.DataArray
    n_space: int
    spatial_er: np.ndarray
    spatial_ec: np.ndarray
    stitch_meridian: bool


# ---------------------------------------------------------------------------
# Native spatial adjacency (8-neighbour; meridian seam when stitching)
#
# Used only for connected-component *labelling* when stitch_meridian=True.
# Support *counting* uses box dilation instead (see _dilate_boolean_support_mask).
# Each spatial cell is a flat index s = y * x_len + x; edges are undirected pairs (s_i, s_j).
# ---------------------------------------------------------------------------


def _add_adjacent_true_pairs(
    mask2d: np.ndarray,
    edge_set: set[tuple[int, int]],
    flat_idx_2d: np.ndarray,
) -> None:
    """Add undirected 8-neighbour edges between True cells in a 2D mask."""
    # East–west neighbours (same row, adjacent columns)
    common = mask2d[:, :-1] & mask2d[:, 1:]
    if common.any():
        a = flat_idx_2d[:, :-1][common].ravel()
        b = flat_idx_2d[:, 1:][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    # North–south neighbours (same column, adjacent rows)
    common = mask2d[:-1, :] & mask2d[1:, :]
    if common.any():
        a = flat_idx_2d[:-1, :][common].ravel()
        b = flat_idx_2d[1:, :][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    # SW–NE diagonal
    common = mask2d[:-1, :-1] & mask2d[1:, 1:]
    if common.any():
        a = flat_idx_2d[:-1, :-1][common].ravel()
        b = flat_idx_2d[1:, 1:][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    # SE–NW diagonal
    common = mask2d[:-1, 1:] & mask2d[1:, :-1]
    if common.any():
        a = flat_idx_2d[:-1, 1:][common].ravel()
        b = flat_idx_2d[1:, :-1][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))


def _add_wrapped_longitude_pairs(
    mask2d: np.ndarray,
    edge_set: set[tuple[int, int]],
    flat_idx_2d: np.ndarray,
) -> None:
    """Add seam edges between the first and last native-grid columns."""
    if mask2d.shape[1] < 2:
        return
    # Same row across the dateline (col 0 ↔ col -1)
    common = mask2d[:, 0] & mask2d[:, -1]
    if common.any():
        a = flat_idx_2d[:, 0][common].ravel()
        b = flat_idx_2d[:, -1][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    # Diagonal seam neighbours (8-connectivity at the wrap)
    common = mask2d[:-1, 0] & mask2d[1:, -1]
    if common.any():
        a = flat_idx_2d[:-1, 0][common].ravel()
        b = flat_idx_2d[1:, -1][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    common = mask2d[1:, 0] & mask2d[:-1, -1]
    if common.any():
        a = flat_idx_2d[1:, 0][common].ravel()
        b = flat_idx_2d[:-1, -1][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))


def native_spatial_edges(
    mask2d: np.ndarray,
    flat_idx_2d: np.ndarray,
    *,
    stitch_longitude: bool = False,
) -> tuple[list[int], list[int]]:
    """Undirected native 8-neighbour edges for cells where ``mask2d`` is True."""
    edges: set[tuple[int, int]] = set()
    _add_adjacent_true_pairs(mask2d, edges, flat_idx_2d)
    if stitch_longitude:
        _add_wrapped_longitude_pairs(mask2d, edges, flat_idx_2d)
    if not edges:
        return [], []
    # Split undirected pairs into CSR-style row/col arrays for graph algorithms
    rows, cols = zip(*edges)
    return list(rows), list(cols)


# ---------------------------------------------------------------------------
# Support dilation and vote counting
# ---------------------------------------------------------------------------


def _labels_tyx(
    td: Any,
    cvar_name: str,
    time_dim: str,
    spatial_dims: tuple[str, str],
) -> np.ndarray:
    """Cluster labels as a (time, y, x) numpy array."""
    da = td.data[cvar_name]
    return da.transpose(time_dim, spatial_dims[0], spatial_dims[1]).values


def _dilate_boolean_support_mask(
    mask_tyx: np.ndarray,
    *,
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
    stitch_meridian: bool,
) -> np.ndarray:
    """Box-dilate a native-event mask for *support counting* (does not create output voxels).

    ``spatial_tolerance=np.inf`` collapses space first: a timestep with any detection
    supports every cell at that time (then a temporal OR window is applied).
    """
    k_t = max(0, int(temporal_tolerance))
    mask = np.asarray(mask_tyx, dtype=bool)
    if is_infinite_spatial_tolerance(spatial_tolerance):
        any_t = mask.any(axis=(1, 2))
        if k_t > 0:
            any_t = ndimage.maximum_filter1d(
                any_t.astype(np.uint8),
                size=2 * k_t + 1,
                mode="constant",
                cval=0,
            ).astype(bool, copy=False)
        return np.broadcast_to(any_t[:, None, None], mask.shape).copy()

    k_s = max(0, int(spatial_tolerance))
    if k_t == 0 and k_s == 0:
        return mask.copy()
    # maximum_filter = logical OR over a (2*k_t+1) × (2*k_s+1) × (2*k_s+1) box
    size = (2 * k_t + 1, 2 * k_s + 1, 2 * k_s + 1)
    # Wrap longitude axis so dilation sees col 0 and col -1 as neighbours
    mode: Any = ["constant", "constant", "wrap"] if stitch_meridian else "constant"
    return ndimage.maximum_filter(
        mask,
        size=size,
        mode=mode,
        cval=False,
    ).astype(bool, copy=False)


def _accumulate_member_support(
    td: Any,
    *,
    cluster_vars: list[str],
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
    show_progress: bool,
    context: SpacetimeGridContext,
) -> MemberSupport:
    """Accumulate native event masks and dilated support votes over members.

    When sign-aware inputs are present, votes and native masks are split by sign;
    otherwise ``votes_primary`` is the combined vote count and ``votes_secondary`` is
    ``None``.
    """
    n_st = context.T * context.n_space
    native_union = np.zeros(n_st, dtype=bool)
    sign_aware = has_sign_aware_inputs(td, cluster_vars)
    native_by_sign = {
        1: np.zeros(n_st, dtype=bool),
        -1: np.zeros(n_st, dtype=bool),
    }
    votes_pos = np.zeros(n_st, dtype=np.int16)
    votes_neg = np.zeros(n_st, dtype=np.int16)
    votes_any = np.zeros(n_st, dtype=np.int16)

    # --- per input: native event mask → dilated support → increment vote count ---
    for cvar in tqdm(
        cluster_vars,
        total=len(cluster_vars),
        disable=not show_progress,
        desc="member-support consensus",
    ):
        labels_orig = _labels_tyx(td, cvar, context.time_dim, context.spatial_dims)
        orig_flat = np.asarray(labels_orig).reshape(-1)
        valid = np.isfinite(orig_flat) & (orig_flat >= 0)
        if not np.any(valid):
            continue

        if sign_aware:
            signs_flat = cluster_signs_flat_for_var(td, cvar)
            if signs_flat is None:
                continue
            for sign_value, vote_arr in ((1.0, votes_pos), (-1.0, votes_neg)):
                native_mask = valid & (signs_flat == sign_value)
                if not np.any(native_mask):
                    continue
                native_union |= native_mask
                native_by_sign[int(sign_value)] |= native_mask
                dilated = _dilate_boolean_support_mask(
                    native_mask.reshape(context.T, context.y_len, context.x_len),
                    temporal_tolerance=temporal_tolerance,
                    spatial_tolerance=spatial_tolerance,
                    stitch_meridian=context.stitch_meridian,
                )
                vote_arr[dilated.reshape(-1)] += 1
        else:
            native_union |= valid
            dilated = _dilate_boolean_support_mask(
                valid.reshape(context.T, context.y_len, context.x_len),
                temporal_tolerance=temporal_tolerance,
                spatial_tolerance=spatial_tolerance,
                stitch_meridian=context.stitch_meridian,
            )
            votes_any[dilated.reshape(-1)] += 1

    if sign_aware:
        return MemberSupport(
            native_union=native_union,
            votes_primary=votes_pos,
            votes_secondary=votes_neg,
            sign_aware=True,
            native_pos=native_by_sign[1],
            native_neg=native_by_sign[-1],
        )
    return MemberSupport(
        native_union=native_union,
        votes_primary=votes_any,
        votes_secondary=None,
        sign_aware=False,
    )


# ---------------------------------------------------------------------------
# Connected-component labelling of retained voxels
#
# After thresholding, group kept voxels into consensus cluster ids. Two voxels belong
# to the same cluster if they lie within temporal_tolerance (time steps) and
# spatial_tolerance (grid cells) of each other — same tolerances as support dilation,
# but applied to connectivity rather than vote counting.
# ---------------------------------------------------------------------------


def _spatial_neighbourhoods_for_tolerance(
    *,
    spatial_indices: np.ndarray,
    n_space: int,
    spatial_rows: np.ndarray,
    spatial_cols: np.ndarray,
    spatial_tolerance: int,
) -> dict[int, np.ndarray]:
    """All native spatial cells within ``spatial_tolerance`` hops on the meridian graph."""
    unique_s = np.unique(np.asarray(spatial_indices, dtype=np.int64))
    adjacency: list[list[int]] = [[] for _ in range(n_space)]
    for u, v in zip(spatial_rows.tolist(), spatial_cols.tolist()):
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    # BFS from each kept cell: expand ``spatial_tolerance`` times along native 8-neighbours
    neighbourhoods: dict[int, np.ndarray] = {}
    for s in unique_s.tolist():
        root = int(s)
        seen = {root}
        frontier = {root}
        for _ in range(spatial_tolerance):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier -= seen
            if not next_frontier:
                break
            seen.update(next_frontier)
            frontier = next_frontier
        neighbourhoods[root] = np.asarray(sorted(seen), dtype=np.int64)
    return neighbourhoods


def _component_graph_edges_for_kept_voxels(
    *,
    keep: np.ndarray,
    context: SpacetimeGridContext,
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse-graph edges for tolerance-aware spacetime connectivity."""
    if is_infinite_spatial_tolerance(spatial_tolerance):
        return _temporal_only_component_edges(
            keep=keep,
            T=context.T,
            n_space=context.n_space,
            temporal_tolerance=temporal_tolerance,
        )

    kept_nodes = np.flatnonzero(keep).astype(np.int64, copy=False)
    if kept_nodes.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    # Tolerance 0 still connects immediate neighbours (Chebyshev distance 1)
    connect_t = max(1, int(temporal_tolerance))
    connect_s = max(1, int(spatial_tolerance))

    # --- non-stitch: Chebyshev graph in normalised (t, y, x) tolerance space ---
    if not context.stitch_meridian:
        kept_t = kept_nodes // context.n_space
        kept_s = kept_nodes % context.n_space
        kept_y = kept_s // context.x_len
        kept_x = kept_s % context.x_len
        # Scale each axis by its tolerance so KDTree L∞ radius 1 == within the box
        coords = np.column_stack(
            (
                kept_t.astype(np.float32) / float(connect_t),
                kept_y.astype(np.float32) / float(connect_s),
                kept_x.astype(np.float32) / float(connect_s),
            )
        )
        pairs = KDTree(coords).query_pairs(r=1.0, p=np.inf, output_type="ndarray")
        if pairs.size == 0:
            empty = np.array([], dtype=np.int64)
            return kept_nodes, empty, empty
        return (
            kept_nodes,
            pairs[:, 0].astype(np.int64),
            pairs[:, 1].astype(np.int64),
        )

    # --- stitch: BFS spatial neighbourhoods on native grid + time window ---
    # KDTree/scipy.label cannot wrap longitude; use precomputed meridian edges instead.
    kept_t = kept_nodes // context.n_space
    kept_s = kept_nodes % context.n_space
    node_to_pos = {int(node): i for i, node in enumerate(kept_nodes.tolist())}
    neighbourhoods = _spatial_neighbourhoods_for_tolerance(
        spatial_indices=kept_s,
        n_space=context.n_space,
        spatial_rows=context.spatial_er,
        spatial_cols=context.spatial_ec,
        spatial_tolerance=connect_s,
    )
    keep_ts = keep.reshape(context.T, context.n_space)
    rows: list[int] = []
    cols: list[int] = []
    # Connect each kept voxel to others within ±connect_t and spatial neighbourhood
    for i, (t, s) in enumerate(zip(kept_t.tolist(), kept_s.tolist())):
        lo_t = max(0, int(t) - connect_t)
        hi_t = min(context.T - 1, int(t) + connect_t)
        for tt in range(lo_t, hi_t + 1):
            for ss in neighbourhoods[int(s)].tolist():
                if not keep_ts[tt, ss]:
                    continue
                cand = int(tt) * context.n_space + int(ss)
                j = node_to_pos.get(cand)
                if j is None or j <= i:
                    continue  # skip missing nodes and duplicate undirected edges
                rows.append(i)
                cols.append(j)

    return (
        kept_nodes,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
    )


def _label_retained_voxels(
    keep: np.ndarray,
    *,
    context: SpacetimeGridContext,
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
) -> np.ndarray:
    """Assign consensus cluster ids to voxels that passed the vote threshold."""
    n_st = keep.size
    if is_infinite_spatial_tolerance(spatial_tolerance):
        kept_nodes, rows, cols = _component_graph_edges_for_kept_voxels(
            keep=keep,
            context=context,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
        )
        if kept_nodes.size == 0:
            return np.full(n_st, -1, dtype=np.int64)
        graph = coo_matrix(
            (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
            shape=(kept_nodes.size, kept_nodes.size),
        )
        _, labels_kept = connected_components(graph, directed=False, return_labels=True)
        labels_flat = np.full(n_st, -1, dtype=np.int64)
        labels_flat[kept_nodes] = labels_kept.astype(np.int64, copy=False)
        return labels_flat

    connect_t = max(1, int(temporal_tolerance))
    connect_s = max(1, int(spatial_tolerance))

    # --- fast path: unit tolerances, no meridian seam → 3D scipy.label ---
    if connect_t == 1 and connect_s == 1 and not context.stitch_meridian:
        keep_tyx = keep.reshape(context.T, context.y_len, context.x_len)
        structure = np.ones((3, 3, 3), dtype=bool)
        # SciPy stubs mis-type ndimage.label as int | tuple; runtime always returns (labels, n).
        label_out: Any = ndimage.label(keep_tyx, structure=structure)
        labels_tyx, _ = label_out
        flat_labels = labels_tyx.reshape(-1)
        labels_flat = np.full(n_st, -1, dtype=np.int64)
        labelled = flat_labels > 0
        labels_flat[labelled] = flat_labels[labelled] - 1
        return labels_flat

    # --- general path: sparse graph on kept voxels → connected_components ---
    kept_nodes, rows, cols = _component_graph_edges_for_kept_voxels(
        keep=keep,
        context=context,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=spatial_tolerance,
    )
    if kept_nodes.size == 0:
        return np.full(n_st, -1, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
        shape=(kept_nodes.size, kept_nodes.size),
    )
    _, labels_kept = connected_components(graph, directed=False, return_labels=True)
    labels_flat = np.full(n_st, -1, dtype=np.int64)
    labels_flat[kept_nodes] = labels_kept.astype(np.int64, copy=False)
    return labels_flat


# ---------------------------------------------------------------------------
# xarray assembly and encoding
#
# Solver internals use flat numpy arrays; these helpers build the shared grid
# layout, then map results back to xarray with TOAD's three-way label semantics:
#   NaN  — no input detected an abrupt shift at this cell
#   -1   — at least one input had a shift here, but no consensus (or filtered out)
#   >=0  — consensus cluster id
# ---------------------------------------------------------------------------


def _build_grid_context(
    sample: xr.DataArray,
    *,
    spatial_dims: tuple[str, str],
    time_dim: str,
    stitch_meridian: bool,
) -> SpacetimeGridContext:
    """Fixed grid metadata reused by vote accumulation, labelling, and xarray output."""
    # --- grid shape and spatial coordinates ---
    T = int(sample.sizes[time_dim])
    y_len = int(sample.sizes[spatial_dims[0]])
    x_len = int(sample.sizes[spatial_dims[1]])
    n_space = y_len * x_len

    # Keep lat/lon (and any auxiliary spatial coords) for the output DataArrays
    coords_spatial = {
        name: coord
        for name, coord in sample.coords.items()
        if (len(coord.dims) > 0) and set(coord.dims).issubset(spatial_dims)
    }
    for d in spatial_dims:
        coords_spatial.setdefault(d, sample[d])

    # --- precompute native 8-neighbour edges when labelling across the lon seam ---
    # Full grid mask: edges depend only on topology, not on which voxels are kept.
    if stitch_meridian:
        flat_idx_2d = np.arange(n_space, dtype=np.int64).reshape(y_len, x_len)
        er, ec = native_spatial_edges(
            np.ones((y_len, x_len), dtype=bool),
            flat_idx_2d,
            stitch_longitude=True,
        )
        spatial_er = np.asarray(er, dtype=np.int64)
        spatial_ec = np.asarray(ec, dtype=np.int64)
    else:
        spatial_er = np.empty(0, dtype=np.int64)
        spatial_ec = np.empty(0, dtype=np.int64)

    return SpacetimeGridContext(
        spatial_dims=spatial_dims,
        time_dim=time_dim,
        T=T,
        y_len=y_len,
        x_len=x_len,
        coords_spatial=cast(dict[str, Any], coords_spatial),
        time_coord=sample[time_dim],
        n_space=n_space,
        spatial_er=spatial_er,
        spatial_ec=spatial_ec,
        stitch_meridian=stitch_meridian,
    )


def _all_inputs_no_shift_mask_flat(
    td: Any,
    cluster_vars: list[str],
    context: SpacetimeGridContext,
) -> np.ndarray:
    """Flat mask: True where every input label is NaN (no abrupt shift anywhere)."""
    T, y_len, x_len = context.T, context.y_len, context.x_len
    all_nan = np.ones((T, y_len, x_len), dtype=bool)
    for cvar in cluster_vars:
        lab_tyx = _labels_tyx(td, cvar, context.time_dim, context.spatial_dims)
        if lab_tyx.shape != (T, y_len, x_len):
            raise ValueError(
                f"Label field {cvar!r} has shape {lab_tyx.shape}, expected "
                f"({T}, {y_len}, {x_len})."
            )
        all_nan &= np.isnan(np.asarray(lab_tyx, dtype=np.float64))
    return all_nan.ravel()


def _mark_no_shift_nan(
    ds: xr.Dataset,
    td: Any,
    cluster_vars: list[str],
    context: SpacetimeGridContext,
) -> xr.Dataset:
    """Promote solver noise (-1) to NaN where no input ever had a shift."""
    all_none = _all_inputs_no_shift_mask_flat(td, cluster_vars, context)
    da_c = ds["clusters"]
    T, y_len, x_len = context.T, context.y_len, context.x_len
    # --- cells where every input is NaN stay NaN (no cluster assigned) ---
    flat = np.asarray(da_c.data, dtype=np.float64).ravel()
    flat = flat.copy()
    flat[(flat == -1) & all_none] = np.nan
    new_lab = flat.reshape(T, y_len, x_len)
    # Rate is undefined wherever the label is NaN
    rate = np.asarray(ds["rate"].data, dtype=np.float32).copy().reshape(-1)
    rate[~np.isfinite(new_lab.ravel())] = np.nan
    rate = rate.reshape(T, y_len, x_len)
    out_vars: dict[str, xr.DataArray] = {
        "clusters": xr.DataArray(
            new_lab,
            coords=da_c.coords,
            dims=da_c.dims,
            attrs=da_c.attrs,
            name=da_c.name,
        ),
        "rate": xr.DataArray(
            rate,
            coords=ds["rate"].coords,
            dims=ds["rate"].dims,
            attrs=ds["rate"].attrs,
            name=ds["rate"].name,
        ),
    }
    if "signs" in ds:
        sign = np.asarray(ds["signs"].data, dtype=np.float32).copy().reshape(-1)
        sign[~np.isfinite(new_lab.ravel())] = np.nan
        sign = sign.reshape(T, y_len, x_len)
        da_s = ds["signs"]
        out_vars["signs"] = xr.DataArray(
            sign,
            coords=da_s.coords,
            dims=da_s.dims,
            attrs=da_s.attrs,
            name=da_s.name,
        )
    return xr.Dataset(out_vars)


def _empty_result(
    td: Any,
    cluster_vars: list[str],
    context: SpacetimeGridContext,
) -> xr.Dataset:
    """No input had a native event anywhere — return all-noise grid before NaN encoding."""
    sd0, sd1 = context.spatial_dims
    T, y_len, x_len = context.T, context.y_len, context.x_len
    # Solver uses -1 for "not in consensus"; _mark_no_shift_nan promotes to NaN where apt
    da_clusters = xr.DataArray(
        np.full((T, y_len, x_len), -1, dtype=np.int32),
        coords={context.time_dim: context.time_coord, **context.coords_spatial},
        dims=[context.time_dim, sd0, sd1],
        name="clusters",
    )
    da_rate = xr.DataArray(
        np.zeros((T, y_len, x_len), dtype=np.float32),
        coords={context.time_dim: context.time_coord, **context.coords_spatial},
        dims=[context.time_dim, sd0, sd1],
        name="rate",
    )
    da_signs = xr.DataArray(
        np.full((T, y_len, x_len), np.nan, dtype=np.float32),
        coords={context.time_dim: context.time_coord, **context.coords_spatial},
        dims=[context.time_dim, sd0, sd1],
        name="signs",
    )
    return _mark_no_shift_nan(
        xr.Dataset({"clusters": da_clusters, "rate": da_rate, "signs": da_signs}),
        td,
        cluster_vars,
        context,
    )


def _build_member_support_dataset(
    td: Any,
    *,
    cluster_vars: list[str],
    min_consensus: float,
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
    context: SpacetimeGridContext,
    support: MemberSupport,
) -> xr.Dataset:
    """Threshold votes, label retained voxels, and return interim clusters + rate + signs."""
    n_members = len(cluster_vars)
    n_st = context.T * context.n_space

    def label_fn(keep: np.ndarray) -> np.ndarray:
        return _label_retained_voxels(
            keep,
            context=context,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
        )

    labels_flat, rate_flat, signs_flat = build_sign_aware_consensus_labels(
        native_union=support.native_union,
        votes_pos=support.votes_primary,
        votes_neg=(
            support.votes_secondary
            if support.votes_secondary is not None
            else np.zeros(n_st, dtype=np.int16)
        ),
        n_members=n_members,
        min_consensus=min_consensus,
        n_st=n_st,
        n_space=context.n_space,
        label_fn=label_fn,
        sign_aware=support.sign_aware,
        votes_any=support.votes_primary if not support.sign_aware else None,
        native_pos=support.native_pos,
        native_neg=support.native_neg,
    )

    # --- reshape to xarray; mark all-NaN-input cells as NaN ---
    clusters_out = np.asarray(labels_flat, dtype=np.int32).reshape(
        context.T, context.y_len, context.x_len
    )
    rate_out = np.asarray(rate_flat, dtype=np.float32).reshape(
        context.T, context.y_len, context.x_len
    )
    da_clusters = xr.DataArray(
        clusters_out,
        coords={context.time_dim: context.time_coord, **context.coords_spatial},
        dims=[context.time_dim, context.spatial_dims[0], context.spatial_dims[1]],
        name="clusters",
    )
    da_rate = xr.DataArray(
        rate_out,
        coords={context.time_dim: context.time_coord, **context.coords_spatial},
        dims=[context.time_dim, context.spatial_dims[0], context.spatial_dims[1]],
        name="rate",
    )
    da_signs = xr.DataArray(
        np.asarray(signs_flat, dtype=np.float32).reshape(
            context.T, context.y_len, context.x_len
        ),
        coords={context.time_dim: context.time_coord, **context.coords_spatial},
        dims=[context.time_dim, context.spatial_dims[0], context.spatial_dims[1]],
        name="signs",
    )
    ds = _mark_no_shift_nan(
        xr.Dataset({"clusters": da_clusters, "rate": da_rate, "signs": da_signs}),
        td,
        cluster_vars,
        context,
    )
    # Interim attrs only; TOAD-facing metadata is attached in Aggregation.compute_consensus.
    ds["clusters"].attrs["description"] = (
        "Tolerance-aware per-voxel member-support consensus. Input clusters are dilated "
        "in space and time for support counting; native event voxels are retained only "
        "when covered by at least min_consensus_members distinct input variables. "
        "Consensus ids are tolerance-aware connected components of retained voxels."
    )
    return ds


def _mark_votes_no_shift_nan(
    da_votes: xr.DataArray,
    td: Any,
    cluster_vars: list[str],
    context: SpacetimeGridContext,
) -> xr.DataArray:
    """Promote zero vote counts to NaN where no input ever had a shift."""
    all_none = _all_inputs_no_shift_mask_flat(td, cluster_vars, context)
    flat = np.asarray(da_votes.data, dtype=np.float64).ravel().copy()
    flat[(flat == 0) & all_none] = np.nan
    new_votes = flat.reshape(context.T, context.y_len, context.x_len)
    return xr.DataArray(
        new_votes,
        coords=da_votes.coords,
        dims=da_votes.dims,
        attrs=da_votes.attrs,
        name=da_votes.name,
    )


def build_consensus_votes_dataarray(
    td: Any,
    *,
    cluster_vars: list[str],
    native_union: np.ndarray,
    member_vote_count: np.ndarray,
    context: SpacetimeGridContext,
) -> xr.DataArray:
    """Assemble a spacetime consensus vote-count field from accumulated support."""
    n_st = context.T * context.n_space
    votes_flat = np.zeros(n_st, dtype=np.float32)
    if np.any(native_union):
        votes_flat[native_union] = member_vote_count[native_union].astype(
            np.float32, copy=False
        )
    da_votes = xr.DataArray(
        votes_flat.reshape(context.T, context.y_len, context.x_len),
        coords={context.time_dim: context.time_coord, **context.coords_spatial},
        dims=[context.time_dim, context.spatial_dims[0], context.spatial_dims[1]],
        name="votes",
    )
    return _mark_votes_no_shift_nan(da_votes, td, cluster_vars, context)


def grid_context_from_votes(
    da_votes: xr.DataArray,
    *,
    spatial_dims: tuple[str, str],
    time_dim: str,
    stitch_meridian_resolved: bool,
) -> SpacetimeGridContext:
    """Rebuild labelling grid context from a stored consensus votes field."""
    return _build_grid_context(
        da_votes,
        spatial_dims=spatial_dims,
        time_dim=time_dim,
        stitch_meridian=stitch_meridian_resolved,
    )


def consensus_clusters_from_votes(
    td: Any,
    da_votes: xr.DataArray,
    *,
    min_consensus: float,
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
    context: SpacetimeGridContext,
    cluster_vars: list[str],
) -> xr.DataArray:
    """Threshold stored vote counts and label tolerance-aware consensus clusters."""
    n_members = len(cluster_vars)
    min_votes = min_consensus_members(n_members, min_consensus)
    n_st = context.T * context.n_space

    votes_flat = np.nan_to_num(
        np.asarray(da_votes.data, dtype=np.float32).ravel(), nan=0.0
    )
    member_vote_count = votes_flat.astype(np.int16, copy=False)
    native_union = member_vote_count > 0

    if not np.any(native_union):
        da_clusters = xr.DataArray(
            np.full((context.T, context.y_len, context.x_len), -1, dtype=np.int32),
            coords={context.time_dim: context.time_coord, **context.coords_spatial},
            dims=[context.time_dim, context.spatial_dims[0], context.spatial_dims[1]],
            name="clusters",
        )
        ds = _mark_no_shift_nan(
            xr.Dataset(
                {
                    "clusters": da_clusters,
                    "rate": xr.zeros_like(da_clusters, dtype=np.float32),
                }
            ),
            td,
            cluster_vars,
            context,
        )
        return ds["clusters"]

    keep = native_union & (member_vote_count >= min_votes)
    if np.any(keep):
        labels_flat = _label_retained_voxels(
            keep,
            context=context,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
        )
        labels_flat, _ = sorted_consensus_labels_by_area(
            sorted_cluster_labels(labels_flat), n_space=context.n_space
        )
    else:
        labels_flat = np.full(n_st, -1, dtype=np.int64)

    clusters_out = np.asarray(labels_flat, dtype=np.int32).reshape(
        context.T, context.y_len, context.x_len
    )
    da_clusters = xr.DataArray(
        clusters_out,
        coords={context.time_dim: context.time_coord, **context.coords_spatial},
        dims=[context.time_dim, context.spatial_dims[0], context.spatial_dims[1]],
        name="clusters",
    )
    ds = _mark_no_shift_nan(
        xr.Dataset(
            {
                "clusters": da_clusters,
                "rate": xr.zeros_like(da_clusters, dtype=np.float32),
            }
        ),
        td,
        cluster_vars,
        context,
    )
    da_clusters = ds["clusters"]
    return da_clusters


# Backward-compatible alias (tests, external imports)
_native_edges_from_mask = native_spatial_edges
