"""Per-voxel member-support spacetime consensus.

Pipeline (read top to bottom in this module):

1. :func:`_accumulate_member_support` — dilated vote counts per native event voxel
2. :func:`_build_member_support_dataset` — threshold, label, assemble xarray output
3. Connectivity helpers — group retained voxels into consensus cluster ids

:meth:`~toad.postprocessing.aggregation.Aggregation.compute_consensus` orchestrates
these steps (grid context, empty result, finalize attrs on ``td.data``).

Native **8-neighbour** grid edges (:func:`native_spatial_edges`) are **not** left over
from the old edge-vote (EAC) algorithm. They are used only when ``stitch_meridian=True``
to build spatial adjacency for connected-component labelling across the longitude seam.
Support counting uses box dilation (:func:`_dilate_boolean_support_mask`); non-stitch
labelling uses ``scipy.ndimage.label`` or a KD-tree Chebyshev graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from tqdm import tqdm

from toad.clustering import sorted_cluster_labels


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
    spatial_tolerance: int,
    stitch_meridian: bool,
) -> np.ndarray:
    """Box-dilate a native-event mask for *support counting* (does not create output voxels)."""
    k_t = max(0, int(temporal_tolerance))
    k_s = max(0, int(spatial_tolerance))
    mask = np.asarray(mask_tyx, dtype=bool)
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
    spatial_tolerance: int,
    show_progress: bool,
    context: SpacetimeGridContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(native_union, member_vote_count)`` flat spacetime arrays.

    Flat index layout: ``node = t * n_space + s`` with ``s = y * x_len + x``.
    ``native_union`` marks voxels where at least one input had a real cluster event;
    ``member_vote_count`` counts how many inputs' *dilated* support covers each voxel.
    """
    n_st = context.T * context.n_space
    native_union = np.zeros(n_st, dtype=bool)
    member_vote_count = np.zeros(n_st, dtype=np.int16)

    # --- per input: native event mask → dilated support → increment vote count ---
    for cvar in tqdm(
        cluster_vars,
        total=len(cluster_vars),
        disable=not show_progress,
        desc="member-support consensus",
    ):
        labels_orig = _labels_tyx(td, cvar, context.time_dim, context.spatial_dims)
        orig_flat = np.asarray(labels_orig).reshape(-1)
        native_mask = np.isfinite(orig_flat) & (orig_flat >= 0)
        if not np.any(native_mask):
            continue
        native_union |= native_mask
        dilated = _dilate_boolean_support_mask(
            native_mask.reshape(context.T, context.y_len, context.x_len),
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            stitch_meridian=context.stitch_meridian,
        )
        member_vote_count[dilated.reshape(-1)] += 1

    return native_union, member_vote_count


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
    spatial_tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse-graph edges for tolerance-aware spacetime connectivity."""
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
    spatial_tolerance: int,
) -> np.ndarray:
    """Assign consensus cluster ids to voxels that passed the vote threshold."""
    connect_t = max(1, int(temporal_tolerance))
    connect_s = max(1, int(spatial_tolerance))
    n_st = keep.size

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
    return xr.Dataset(
        {
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
    )


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
    return _mark_no_shift_nan(
        xr.Dataset({"clusters": da_clusters, "rate": da_rate}),
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
    spatial_tolerance: int,
    context: SpacetimeGridContext,
    native_union: np.ndarray,
    member_vote_count: np.ndarray,
) -> xr.Dataset:
    """Threshold votes, label retained voxels, and return interim clusters + rate."""
    n_members = len(cluster_vars)
    min_votes = min_consensus_members(n_members, min_consensus)
    n_st = context.T * context.n_space

    # --- retain native event voxels with enough distinct input votes ---
    keep = native_union & (member_vote_count >= min_votes)

    # --- member support fraction at every native event voxel (incl. sub-threshold) ---
    rate_flat = np.zeros(n_st, dtype=np.float32)
    if np.any(native_union):
        rate_flat[native_union] = (
            member_vote_count[native_union].astype(np.float32) / n_members
        )

    # --- tolerance-aware connected components on retained voxels ---
    if np.any(keep):
        labels_flat = _label_retained_voxels(
            keep,
            context=context,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
        )
        labels_flat = sorted_cluster_labels(labels_flat)
    else:
        labels_flat = np.full(n_st, -1, dtype=np.int64)

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
    ds = _mark_no_shift_nan(
        xr.Dataset({"clusters": da_clusters, "rate": da_rate}),
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


# Backward-compatible alias (tests, external imports)
_native_edges_from_mask = native_spatial_edges
