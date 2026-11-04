"""Per-voxel member-support spacetime consensus on a HEALPix grid."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from toad.clustering import sorted_cluster_labels
from toad.healpix import build_ring1_spatial_edges
from toad.postprocessing.aggregation import _filter_consensus_labels_min_size
from toad.postprocessing.member_support_consensus import (
    MemberSupport,
    SpatialTolerance,
    TemporalTolerance,
    _spatial_only_component_edges,
    _temporal_only_component_edges,
    build_sign_aware_consensus_labels,
    cluster_signs_flat_for_var,
    has_sign_aware_inputs,
    is_infinite_spatial_tolerance,
    is_infinite_temporal_tolerance,
    min_consensus_members,
    reject_both_infinite_tolerances,
    sorted_consensus_labels_by_area,
    validate_consensus_tolerances,
)
from toad.utils import _attrs


@dataclass(frozen=True)
class HealpixSpacetimeContext:
    """Fixed HEALPix spacetime layout for one member-support consensus run."""

    time_dim: str
    pixel_dim: str
    T: int
    npix: int
    nside: int
    time_coord: xr.DataArray
    spatial_rows: np.ndarray
    spatial_cols: np.ndarray


def build_healpix_spatial_edges(
    nside: int,
    *,
    k_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Undirected ring-1 edges on the full HEALPix pixel set."""
    if k_neighbors != 8:
        warnings.warn(
            "k_neighbors is deprecated and ignored; HEALPix ring-1 neighbours are used.",
            DeprecationWarning,
            stacklevel=2,
        )
    return build_ring1_spatial_edges(nside)


def _spatial_neighbourhoods_for_tolerance(
    *,
    spatial_indices: np.ndarray,
    npix: int,
    spatial_rows: np.ndarray,
    spatial_cols: np.ndarray,
    spatial_tolerance: int,
) -> dict[int, np.ndarray]:
    """All HEALPix pixels within ``spatial_tolerance`` hops on the ring-1 graph."""
    unique_s = np.unique(np.asarray(spatial_indices, dtype=np.int64))
    adjacency: list[list[int]] = [[] for _ in range(npix)]
    for u, v in zip(spatial_rows.tolist(), spatial_cols.tolist()):
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

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


def _healpix_adjacency_csr(
    spatial_rows: np.ndarray,
    spatial_cols: np.ndarray,
    npix: int,
) -> csr_matrix:
    """Symmetric ring-1 HEALPix adjacency as a sparse matrix."""
    rows = np.asarray(spatial_rows, dtype=np.int64)
    cols = np.asarray(spatial_cols, dtype=np.int64)
    both_rows = np.concatenate([rows, cols])
    both_cols = np.concatenate([cols, rows])
    data = np.ones(both_rows.shape[0], dtype=np.float32)
    return csr_matrix((data, (both_rows, both_cols)), shape=(npix, npix))


@lru_cache(maxsize=16)
def _healpix_spatial_reachability_table(nside: int, connect_s: int) -> np.ndarray:
    """All pixels within ``connect_s`` ring-1 hops for every HEALPix cell (includes self)."""
    if connect_s < 1:
        raise ValueError(f"`connect_s` must be >= 1, got {connect_s}.")
    npix = 12 * nside**2
    spatial_rows, spatial_cols = build_healpix_spatial_edges(nside)
    adjacency: list[list[int]] = [[] for _ in range(npix)]
    for u, v in zip(spatial_rows.tolist(), spatial_cols.tolist()):
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    neighbourhoods: list[list[int]] = []
    max_deg = 1
    for root in range(npix):
        seen = {root}
        frontier = {root}
        for _ in range(connect_s):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier -= seen
            if not next_frontier:
                break
            seen.update(next_frontier)
            frontier = next_frontier
        nbr_list = sorted(seen)
        neighbourhoods.append(nbr_list)
        max_deg = max(max_deg, len(nbr_list))

    table = np.full((npix, max_deg), -1, dtype=np.int64)
    for s, nbr_list in enumerate(neighbourhoods):
        table[s, : len(nbr_list)] = nbr_list
    return table


def _dilate_healpix_support_mask(
    mask_tpix: np.ndarray,
    *,
    temporal_tolerance: TemporalTolerance,
    spatial_tolerance: SpatialTolerance,
    spatial_rows: np.ndarray,
    spatial_cols: np.ndarray,
) -> np.ndarray:
    """Dilate a native-event mask on (time, hp_pixel) for support counting.

    Spacetime box dilation is separable on HEALPix: expand each timestep by
    ``spatial_tolerance`` hops on the ring-1 graph, then apply a temporal OR
    window of width ``2 * temporal_tolerance + 1``.

    ``spatial_tolerance=np.inf`` collapses space first (any detection at a
    timestep supports every pixel), then applies the temporal window.
    ``temporal_tolerance=np.inf`` collapses time first (any detection in a pixel
    supports that pixel at every time), then applies the spatial hop window.
    """
    reject_both_infinite_tolerances(temporal_tolerance, spatial_tolerance)
    mask = np.asarray(mask_tpix, dtype=bool)
    if is_infinite_spatial_tolerance(spatial_tolerance):
        k_t = max(0, int(temporal_tolerance))
        any_t = mask.any(axis=1)
        if k_t > 0:
            any_t = ndimage.maximum_filter1d(
                any_t.astype(np.uint8),
                size=2 * k_t + 1,
                mode="constant",
                cval=0,
            ).astype(bool, copy=False)
        return np.broadcast_to(any_t[:, None], mask.shape).copy()

    if is_infinite_temporal_tolerance(temporal_tolerance):
        any_pix = mask.any(axis=0)
        k_s = max(0, int(spatial_tolerance))
        if k_s > 0:
            adj = _healpix_adjacency_csr(spatial_rows, spatial_cols, mask.shape[1])
            current = any_pix.astype(np.float32)
            spatial = any_pix.copy()
            for _ in range(k_s):
                current = adj @ current
                spatial |= current > 0
            any_pix = spatial
        return np.broadcast_to(any_pix[None, :], mask.shape).copy()

    k_t = max(0, int(temporal_tolerance))
    k_s = max(0, int(spatial_tolerance))
    if k_t == 0 and k_s == 0:
        return mask.copy()

    _, npix = mask.shape
    spatial_dilated = mask.copy()
    if k_s > 0:
        adj = _healpix_adjacency_csr(spatial_rows, spatial_cols, npix)
        current = mask.astype(np.float32)
        for _ in range(k_s):
            current = (adj @ current.T).T
            spatial_dilated |= current > 0

    if k_t == 0:
        return spatial_dilated

    size = 2 * k_t + 1
    return ndimage.maximum_filter1d(
        spatial_dilated.astype(np.uint8),
        size=size,
        axis=0,
        mode="constant",
        cval=0,
    ).astype(bool, copy=False)


def _labels_tpix(
    td: Any,
    cvar_name: str,
    time_dim: str,
    pixel_dim: str,
) -> np.ndarray:
    da = td.data[cvar_name]
    return da.transpose(time_dim, pixel_dim).values


def _accumulate_member_support_healpix(
    td: Any,
    *,
    cluster_vars: list[str],
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
    show_progress: bool,
    context: HealpixSpacetimeContext,
) -> MemberSupport:
    n_st = context.T * context.npix
    native_union = np.zeros(n_st, dtype=bool)
    sign_aware = has_sign_aware_inputs(td, cluster_vars)
    native_by_sign = {
        1: np.zeros(n_st, dtype=bool),
        -1: np.zeros(n_st, dtype=bool),
    }
    votes_pos = np.zeros(n_st, dtype=np.int16)
    votes_neg = np.zeros(n_st, dtype=np.int16)
    votes_any = np.zeros(n_st, dtype=np.int16)

    for cvar in tqdm(
        cluster_vars,
        total=len(cluster_vars),
        disable=not show_progress,
        desc="healpix member-support consensus",
    ):
        labels = _labels_tpix(td, cvar, context.time_dim, context.pixel_dim)
        orig_flat = np.asarray(labels).reshape(-1)
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
                dilated = _dilate_healpix_support_mask(
                    native_mask.reshape(context.T, context.npix),
                    temporal_tolerance=temporal_tolerance,
                    spatial_tolerance=spatial_tolerance,
                    spatial_rows=context.spatial_rows,
                    spatial_cols=context.spatial_cols,
                )
                vote_arr[dilated.reshape(-1)] += 1
        else:
            native_union |= valid
            dilated = _dilate_healpix_support_mask(
                valid.reshape(context.T, context.npix),
                temporal_tolerance=temporal_tolerance,
                spatial_tolerance=spatial_tolerance,
                spatial_rows=context.spatial_rows,
                spatial_cols=context.spatial_cols,
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


def _component_graph_edges_for_kept_voxels(
    *,
    keep: np.ndarray,
    context: HealpixSpacetimeContext,
    temporal_tolerance: TemporalTolerance,
    spatial_tolerance: SpatialTolerance,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reject_both_infinite_tolerances(temporal_tolerance, spatial_tolerance)
    if is_infinite_spatial_tolerance(spatial_tolerance):
        return _temporal_only_component_edges(
            keep=keep,
            T=context.T,
            n_space=context.npix,
            temporal_tolerance=int(temporal_tolerance),
        )
    if is_infinite_temporal_tolerance(temporal_tolerance):
        kept_nodes = np.flatnonzero(keep).astype(np.int64, copy=False)
        if kept_nodes.size == 0:
            empty = np.array([], dtype=np.int64)
            return empty, empty, empty
        connect_s = max(1, int(spatial_tolerance))
        kept_s = kept_nodes % context.npix
        neighbourhoods = _spatial_neighbourhoods_for_tolerance(
            spatial_indices=kept_s,
            npix=context.npix,
            spatial_rows=context.spatial_rows,
            spatial_cols=context.spatial_cols,
            spatial_tolerance=connect_s,
        )
        return _spatial_only_component_edges(
            keep=keep,
            n_space=context.npix,
            neighbourhoods=neighbourhoods,
        )

    kept_nodes = np.flatnonzero(keep).astype(np.int64, copy=False)
    if kept_nodes.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    connect_t = max(1, int(temporal_tolerance))
    connect_s = max(1, int(spatial_tolerance))
    kept_t = kept_nodes // context.npix
    kept_s = kept_nodes % context.npix
    node_to_pos = np.full(keep.size, -1, dtype=np.int64)
    node_to_pos[kept_nodes] = np.arange(kept_nodes.size, dtype=np.int64)

    spatial_nbrs = _healpix_spatial_reachability_table(context.nside, connect_s)
    deg = spatial_nbrs.shape[1]
    keep_ts = keep.reshape(context.T, context.npix)

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    for dt in range(-connect_t, connect_t + 1):
        tt = kept_t + dt
        valid_t = (tt >= 0) & (tt < context.T)
        if not np.any(valid_t):
            continue
        idx = np.flatnonzero(valid_t)
        tt_v = tt[valid_t]
        s_v = kept_s[valid_t]
        ss = spatial_nbrs[s_v]
        cand = tt_v[:, None] * context.npix + ss
        valid_nbr = ss >= 0
        tt_rep = np.repeat(tt_v, deg)
        ss_flat = ss.ravel()
        valid_flat = valid_nbr.ravel()
        keep_mask = np.zeros(cand.size, dtype=bool)
        ok = valid_flat
        keep_mask[ok] = keep_ts[tt_rep[ok], ss_flat[ok]]
        j_pos = node_to_pos[cand.ravel()]
        i_rep = np.repeat(idx, deg)
        edge_ok = keep_mask & (j_pos >= 0) & (j_pos > i_rep)
        rows_parts.append(i_rep[edge_ok])
        cols_parts.append(j_pos[edge_ok])

    if rows_parts:
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)

    return kept_nodes, rows, cols


def _label_retained_voxels(
    keep: np.ndarray,
    *,
    context: HealpixSpacetimeContext,
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
) -> np.ndarray:
    n_st = keep.size
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


def _all_inputs_no_shift_mask_flat(
    td: Any,
    cluster_vars: list[str],
    context: HealpixSpacetimeContext,
) -> np.ndarray:
    all_nan = np.ones((context.T, context.npix), dtype=bool)
    for cvar in cluster_vars:
        lab = _labels_tpix(td, cvar, context.time_dim, context.pixel_dim)
        if lab.shape != (context.T, context.npix):
            raise ValueError(
                f"Label field {cvar!r} has shape {lab.shape}, expected "
                f"({context.T}, {context.npix})."
            )
        all_nan &= np.isnan(np.asarray(lab, dtype=np.float64))
    return all_nan.ravel()


def _mark_no_shift_nan(
    ds: xr.Dataset,
    td: Any,
    cluster_vars: list[str],
    context: HealpixSpacetimeContext,
) -> xr.Dataset:
    if not cluster_vars or not all(cvar in td.data for cvar in cluster_vars):
        return ds
    all_none = _all_inputs_no_shift_mask_flat(td, cluster_vars, context)
    da_c = ds["clusters"]
    flat = np.asarray(da_c.data, dtype=np.float64).ravel().copy()
    flat[(flat == -1) & all_none] = np.nan
    new_lab = flat.reshape(context.T, context.npix)
    rate = np.asarray(ds["rate"].data, dtype=np.float32).copy().reshape(-1)
    rate[~np.isfinite(new_lab.ravel())] = np.nan
    rate = rate.reshape(context.T, context.npix)
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
        sign = sign.reshape(context.T, context.npix)
        da_s = ds["signs"]
        out_vars["signs"] = xr.DataArray(
            sign,
            coords=da_s.coords,
            dims=da_s.dims,
            attrs=da_s.attrs,
            name=da_s.name,
        )
    return xr.Dataset(out_vars)


def _mark_healpix_votes_no_shift_nan(
    da_votes: xr.DataArray,
    td: Any,
    cluster_vars: list[str],
    context: HealpixSpacetimeContext,
) -> xr.DataArray:
    all_none = _all_inputs_no_shift_mask_flat(td, cluster_vars, context)
    flat = np.asarray(da_votes.data, dtype=np.float64).ravel().copy()
    flat[(flat == 0) & all_none] = np.nan
    new_votes = flat.reshape(context.T, context.npix)
    return xr.DataArray(
        new_votes,
        coords=da_votes.coords,
        dims=da_votes.dims,
        attrs=da_votes.attrs,
        name=da_votes.name,
    )


def build_healpix_consensus_votes_dataarray(
    td: Any,
    *,
    cluster_vars: list[str],
    native_union: np.ndarray,
    member_vote_count: np.ndarray,
    context: HealpixSpacetimeContext,
) -> xr.DataArray:
    n_st = context.T * context.npix
    votes_flat = np.zeros(n_st, dtype=np.float32)
    if np.any(native_union):
        votes_flat[native_union] = member_vote_count[native_union].astype(
            np.float32, copy=False
        )
    da_votes = xr.DataArray(
        votes_flat.reshape(context.T, context.npix),
        coords={
            context.time_dim: context.time_coord,
            context.pixel_dim: np.arange(context.npix),
        },
        dims=[context.time_dim, context.pixel_dim],
        name="votes",
    )
    return _mark_healpix_votes_no_shift_nan(da_votes, td, cluster_vars, context)


def _healpix_context_from_votes(
    da_votes: xr.DataArray, *, nside: int
) -> HealpixSpacetimeContext:
    time_dim = da_votes.dims[0]
    pixel_dim = "hp_pixel"
    T = int(da_votes.sizes[time_dim])
    npix = int(da_votes.sizes[pixel_dim])
    spatial_rows, spatial_cols = build_healpix_spatial_edges(nside)
    return HealpixSpacetimeContext(
        time_dim=time_dim,
        pixel_dim=pixel_dim,
        T=T,
        npix=npix,
        nside=nside,
        time_coord=da_votes[time_dim],
        spatial_rows=spatial_rows,
        spatial_cols=spatial_cols,
    )


def consensus_clusters_from_healpix_votes(
    td: Any,
    da_votes: xr.DataArray,
    *,
    min_consensus: float,
    temporal_tolerance: int,
    spatial_tolerance: SpatialTolerance,
    context: HealpixSpacetimeContext,
    cluster_vars: list[str],
) -> xr.DataArray:
    n_members_attr = da_votes.attrs.get(_attrs.N_MODELS)
    if n_members_attr is not None:
        n_members = int(n_members_attr)
    else:
        n_members = len(cluster_vars)
    min_votes = min_consensus_members(n_members, min_consensus)
    n_st = context.T * context.npix

    votes_flat = np.nan_to_num(
        np.asarray(da_votes.data, dtype=np.float32).ravel(), nan=0.0
    )
    member_vote_count = votes_flat.astype(np.int16, copy=False)
    native_union = member_vote_count > 0

    if not np.any(native_union):
        da_clusters = xr.DataArray(
            np.full((context.T, context.npix), -1, dtype=np.int32),
            coords={
                context.time_dim: context.time_coord,
                context.pixel_dim: np.arange(context.npix),
            },
            dims=[context.time_dim, context.pixel_dim],
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
            sorted_cluster_labels(labels_flat), n_space=context.npix
        )
    else:
        labels_flat = np.full(n_st, -1, dtype=np.int64)

    da_clusters = xr.DataArray(
        np.asarray(labels_flat, dtype=np.int32).reshape(context.T, context.npix),
        coords={
            context.time_dim: context.time_coord,
            context.pixel_dim: np.arange(context.npix),
        },
        dims=[context.time_dim, context.pixel_dim],
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


def _accumulate_healpix_votes_context(
    td: Any,
    *,
    cluster_vars: list[str],
    temporal_tolerance: TemporalTolerance,
    spatial_tolerance: SpatialTolerance,
    nside: int,
    k_neighbors: int = 8,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, HealpixSpacetimeContext]:
    temporal_tolerance, spatial_tolerance = validate_consensus_tolerances(
        temporal_tolerance, spatial_tolerance
    )
    if not cluster_vars:
        raise ValueError("cluster_vars must not be empty.")

    pixel_dim = "hp_pixel"
    time_dim = td.time_dim
    sample = td.data[cluster_vars[0]]
    if pixel_dim not in sample.dims:
        raise ValueError(
            f"Expected cluster labels on dimension {pixel_dim!r}, got dims={sample.dims}."
        )

    T = int(sample.sizes[time_dim])
    npix = int(sample.sizes[pixel_dim])
    if npix != 12 * nside**2:
        raise ValueError(
            f"hp_pixel size {npix} does not match nside={nside} (expected {12 * nside**2})."
        )

    spatial_rows, spatial_cols = build_healpix_spatial_edges(
        nside, k_neighbors=k_neighbors
    )
    context = HealpixSpacetimeContext(
        time_dim=time_dim,
        pixel_dim=pixel_dim,
        T=T,
        npix=npix,
        nside=nside,
        time_coord=sample[time_dim],
        spatial_rows=spatial_rows,
        spatial_cols=spatial_cols,
    )
    support = _accumulate_member_support_healpix(
        td,
        cluster_vars=cluster_vars,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=spatial_tolerance,
        show_progress=show_progress,
        context=context,
    )
    if support.sign_aware and support.votes_secondary is not None:
        member_vote_count = np.maximum(support.votes_primary, support.votes_secondary)
    else:
        member_vote_count = support.votes_primary
    return support.native_union, member_vote_count, context


def run_healpix_member_support_consensus(
    td: Any,
    *,
    cluster_vars: list[str],
    min_consensus: float,
    temporal_tolerance: TemporalTolerance,
    spatial_tolerance: SpatialTolerance,
    nside: int,
    k_neighbors: int = 8,
    min_cluster_area: int | None = 2,
    show_progress: bool = True,
) -> xr.Dataset:
    """Run member-support consensus on HEALPix cluster label fields.

    ``spatial_tolerance=np.inf`` enables unlimited spatial reach (temporal-only
    member support and connectivity). ``temporal_tolerance=np.inf`` enables
    unlimited temporal reach (spatial-only member support and connectivity).
    The two cannot be combined.
    """
    temporal_tolerance, spatial_tolerance = validate_consensus_tolerances(
        temporal_tolerance, spatial_tolerance
    )
    if not (0.0 <= min_consensus <= 1.0):
        raise ValueError(f"`min_consensus` must be in [0, 1], got {min_consensus}.")
    if not cluster_vars:
        raise ValueError("cluster_vars must not be empty.")

    pixel_dim = "hp_pixel"
    time_dim = td.time_dim
    sample = td.data[cluster_vars[0]]
    if pixel_dim not in sample.dims:
        raise ValueError(
            f"Expected cluster labels on dimension {pixel_dim!r}, got dims={sample.dims}."
        )

    T = int(sample.sizes[time_dim])
    npix = int(sample.sizes[pixel_dim])
    if npix != 12 * nside**2:
        raise ValueError(
            f"hp_pixel size {npix} does not match nside={nside} (expected {12 * nside**2})."
        )

    spatial_rows, spatial_cols = build_healpix_spatial_edges(
        nside, k_neighbors=k_neighbors
    )
    context = HealpixSpacetimeContext(
        time_dim=time_dim,
        pixel_dim=pixel_dim,
        T=T,
        npix=npix,
        nside=nside,
        time_coord=sample[time_dim],
        spatial_rows=spatial_rows,
        spatial_cols=spatial_cols,
    )

    support = _accumulate_member_support_healpix(
        td,
        cluster_vars=cluster_vars,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=spatial_tolerance,
        show_progress=show_progress,
        context=context,
    )

    n_members = len(cluster_vars)
    n_st = context.T * context.npix
    if not np.any(support.native_union):
        da_clusters = xr.DataArray(
            np.full((T, npix), -1, dtype=np.int32),
            coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
            dims=[time_dim, pixel_dim],
            name="clusters",
        )
        da_rate = xr.DataArray(
            np.zeros((T, npix), dtype=np.float32),
            coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
            dims=[time_dim, pixel_dim],
            name="rate",
        )
        da_signs = xr.DataArray(
            np.full((T, npix), np.nan, dtype=np.float32),
            coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
            dims=[time_dim, pixel_dim],
            name="signs",
        )
        return _mark_no_shift_nan(
            xr.Dataset({"clusters": da_clusters, "rate": da_rate, "signs": da_signs}),
            td,
            cluster_vars,
            context,
        )

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
        n_space=context.npix,
        label_fn=label_fn,
        sign_aware=support.sign_aware,
        votes_any=support.votes_primary if not support.sign_aware else None,
        native_pos=support.native_pos,
        native_neg=support.native_neg,
    )

    da_clusters = xr.DataArray(
        np.asarray(labels_flat, dtype=np.int32).reshape(T, npix),
        coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
        dims=[time_dim, pixel_dim],
        name="clusters",
    )
    da_rate = xr.DataArray(
        np.asarray(rate_flat, dtype=np.float32).reshape(T, npix),
        coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
        dims=[time_dim, pixel_dim],
        name="rate",
    )
    da_signs = xr.DataArray(
        np.asarray(signs_flat, dtype=np.float32).reshape(T, npix),
        coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
        dims=[time_dim, pixel_dim],
        name="signs",
    )
    ds = _mark_no_shift_nan(
        xr.Dataset({"clusters": da_clusters, "rate": da_rate, "signs": da_signs}),
        td,
        cluster_vars,
        context,
    )
    if min_cluster_area is not None and min_cluster_area > 0:
        da_c, da_r, da_s = _filter_consensus_labels_min_size(
            ds["clusters"],
            ds["rate"],
            min_cluster_area,
            time_dim=time_dim,
            da_sign=ds["signs"],
        )
        ds = xr.Dataset({"clusters": da_c, "rate": da_r, "signs": da_s})
    return ds
