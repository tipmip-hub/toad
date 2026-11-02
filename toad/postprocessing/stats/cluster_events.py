"""Shared spacetime index of labeled cluster cells (for batch stats)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toad.utils import detect_latlon_names


@dataclass(frozen=True)
class ClusterLabelEvents:
    """Flat (label, time-index, space-index) triples for finite cluster labels."""

    labels: np.ndarray  # int64, shape (N,)
    t_idx: np.ndarray  # int64, shape (N,)
    p_idx: np.ndarray  # int64, shape (N,)
    numeric_times: np.ndarray  # float64, shape (T,)
    y_flat: np.ndarray  # float64, shape (P,)
    x_flat: np.ndarray  # float64, shape (P,)


def _footprint_coord_flats(td, space_template) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast y/x (or lat/lon) coords to the cluster spatial grid and ravel."""
    lat_name, lon_name = detect_latlon_names(td.data)
    if lat_name is not None and lon_name is not None:
        y_da = td.data[lat_name]
        x_da = td.data[lon_name]
    else:
        sd0, sd1 = td.space_dims
        y_da = td.data[sd0]
        x_da = td.data[sd1]
    y_flat = np.asarray(
        y_da.broadcast_like(space_template).values, dtype=np.float64
    ).ravel()
    x_flat = np.asarray(
        x_da.broadcast_like(space_template).values, dtype=np.float64
    ).ravel()
    return y_flat, x_flat


def load_cluster_label_events(
    td,
    var: str,
    cluster_ids: list[int] | np.ndarray | None = None,
) -> ClusterLabelEvents:
    """Extract labeled spacetime cells once for vectorised cluster stats.

    Matches the points where ``get_cluster_mask`` would be True for each id:
    finite labels in the cluster field (including noise ``-1`` when requested).
    """
    time_dim = td.time_dim
    space_dims = list(td.space_dims)
    clusters = td.get_clusters(var).transpose(time_dim, *space_dims)
    lab = np.asarray(clusters.values)
    T = lab.shape[0]
    P = int(np.prod(lab.shape[1:])) if lab.ndim > 1 else 1
    lab_TP = lab.reshape(T, P)

    t_idx, p_idx = np.nonzero(np.isfinite(lab_TP))
    labels = lab_TP[t_idx, p_idx].astype(np.int64)
    if cluster_ids is not None:
        want = np.asarray(list(cluster_ids), dtype=np.int64)
        keep = np.isin(labels, want)
        labels = labels[keep]
        t_idx = t_idx[keep]
        p_idx = p_idx[keep]

    y_flat, x_flat = _footprint_coord_flats(td, clusters.isel({time_dim: 0}, drop=True))
    return ClusterLabelEvents(
        labels=labels,
        t_idx=t_idx.astype(np.int64, copy=False),
        p_idx=p_idx.astype(np.int64, copy=False),
        numeric_times=np.asarray(td.numeric_time_values, dtype=np.float64),
        y_flat=y_flat,
        x_flat=x_flat,
    )


def _group_slices(sorted_keys: np.ndarray) -> list[tuple[int, int, int]]:
    """Return ``(key, start, end)`` slices for a sorted 1-d key array."""
    if sorted_keys.size == 0:
        return []
    cuts = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [sorted_keys.size]))
    return [(int(sorted_keys[s]), int(s), int(e)) for s, e in zip(starts, ends)]
