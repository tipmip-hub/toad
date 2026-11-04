import numpy as np
import pytest
import xarray as xr

from toad.clustering import sorted_cluster_labels
from toad.healpix import build_ring1_spatial_edges
from toad.postprocessing.healpix_member_support_consensus import (
    HealpixSpacetimeContext,
    _component_graph_edges_for_kept_voxels,
    _dilate_healpix_support_mask,
    _spatial_neighbourhoods_for_tolerance,
    run_healpix_member_support_consensus,
)


def _component_graph_edges_reference(
    *,
    keep: np.ndarray,
    context: HealpixSpacetimeContext,
    temporal_tolerance: int,
    spatial_tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept_nodes = np.flatnonzero(keep).astype(np.int64, copy=False)
    if kept_nodes.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    connect_t = max(1, int(temporal_tolerance))
    connect_s = max(1, int(spatial_tolerance))
    kept_t = kept_nodes // context.npix
    kept_s = kept_nodes % context.npix
    node_to_pos = {int(node): i for i, node in enumerate(kept_nodes.tolist())}
    neighbourhoods = _spatial_neighbourhoods_for_tolerance(
        spatial_indices=kept_s,
        npix=context.npix,
        spatial_rows=context.spatial_rows,
        spatial_cols=context.spatial_cols,
        spatial_tolerance=connect_s,
    )
    keep_ts = keep.reshape(context.T, context.npix)
    rows: list[int] = []
    cols: list[int] = []
    for i, (t, s) in enumerate(zip(kept_t.tolist(), kept_s.tolist())):
        lo_t = max(0, int(t) - connect_t)
        hi_t = min(context.T - 1, int(t) + connect_t)
        for tt in range(lo_t, hi_t + 1):
            for ss in neighbourhoods[int(s)].tolist():
                if not keep_ts[tt, ss]:
                    continue
                cand = int(tt) * context.npix + int(ss)
                j = node_to_pos.get(cand)
                if j is None or j <= i:
                    continue
                rows.append(i)
                cols.append(j)

    return (
        kept_nodes,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
    )


def _dilate_healpix_support_mask_reference(
    mask_tpix: np.ndarray,
    *,
    temporal_tolerance: int,
    spatial_tolerance: int,
    spatial_rows: np.ndarray,
    spatial_cols: np.ndarray,
) -> np.ndarray:
    """Reference implementation kept for regression tests."""
    from toad.postprocessing.healpix_member_support_consensus import (
        _spatial_neighbourhoods_for_tolerance,
    )

    k_t = max(0, int(temporal_tolerance))
    k_s = max(0, int(spatial_tolerance))
    mask = np.asarray(mask_tpix, dtype=bool)
    if k_t == 0 and k_s == 0:
        return mask.copy()

    T, npix = mask.shape
    spatial_nbrs = _spatial_neighbourhoods_for_tolerance(
        spatial_indices=np.arange(npix, dtype=np.int64),
        npix=npix,
        spatial_rows=spatial_rows,
        spatial_cols=spatial_cols,
        spatial_tolerance=k_s,
    )

    out = np.zeros_like(mask)
    for t in range(T):
        t_lo = max(0, t - k_t)
        t_hi = min(T - 1, t + k_t)
        for tt in range(t_lo, t_hi + 1):
            for s in range(npix):
                if np.any(mask[tt, spatial_nbrs[s]]):
                    out[t, s] = True
    return out


def test_healpix_dilation_matches_reference():
    rng = np.random.default_rng(0)
    for nside in (4, 8):
        npix = 12 * nside**2
        rows, cols = build_ring1_spatial_edges(nside)
        for k_t, k_s in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (5, 1)):
            mask = np.zeros((12, npix), dtype=bool)
            active = rng.choice(npix, size=max(3, npix // 200), replace=False)
            times = rng.integers(0, 12, size=active.size)
            mask[times, active] = True

            fast = _dilate_healpix_support_mask(
                mask,
                temporal_tolerance=k_t,
                spatial_tolerance=k_s,
                spatial_rows=rows,
                spatial_cols=cols,
            )
            ref = _dilate_healpix_support_mask_reference(
                mask,
                temporal_tolerance=k_t,
                spatial_tolerance=k_s,
                spatial_rows=rows,
                spatial_cols=cols,
            )
            np.testing.assert_array_equal(fast, ref)


def test_healpix_component_graph_edges_match_reference():
    rng = np.random.default_rng(1)
    for nside in (4, 8):
        npix = 12 * nside**2
        rows, cols = build_ring1_spatial_edges(nside)
        context = HealpixSpacetimeContext(
            time_dim="time",
            pixel_dim="hp_pixel",
            T=12,
            npix=npix,
            nside=nside,
            time_coord=xr.DataArray(np.arange(12), dims=("time",)),
            spatial_rows=rows,
            spatial_cols=cols,
        )
        for connect_t, connect_s in ((1, 1), (5, 1), (2, 2)):
            keep = np.zeros(context.T * context.npix, dtype=bool)
            active = rng.choice(
                context.T * context.npix, size=max(20, npix // 8), replace=False
            )
            keep[active] = True
            ref = _component_graph_edges_reference(
                keep=keep,
                context=context,
                temporal_tolerance=connect_t,
                spatial_tolerance=connect_s,
            )
            fast = _component_graph_edges_for_kept_voxels(
                keep=keep,
                context=context,
                temporal_tolerance=connect_t,
                spatial_tolerance=connect_s,
            )
            np.testing.assert_array_equal(ref[0], fast[0])
            ref_edges = np.column_stack((ref[1], ref[2]))
            fast_edges = np.column_stack((fast[1], fast[2]))
            ref_sorted = ref_edges[np.lexsort((ref_edges[:, 1], ref_edges[:, 0]))]
            fast_sorted = fast_edges[np.lexsort((fast_edges[:, 1], fast_edges[:, 0]))]
            np.testing.assert_array_equal(ref_sorted, fast_sorted)


def test_sorted_cluster_labels_matches_reference():
    rng = np.random.default_rng(2)

    def _reference(cluster_labels: np.ndarray) -> np.ndarray:
        original_shape = cluster_labels.shape
        flat = np.ravel(np.asarray(cluster_labels))
        valid = np.isfinite(flat) & (flat != -1)
        if not np.any(valid):
            return np.asarray(cluster_labels).copy()

        unique_labels, counts = np.unique(flat[valid], return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_unique_labels = unique_labels[sorted_indices]
        label_mapping = {int(old): new for new, old in enumerate(sorted_unique_labels)}
        label_mapping[-1] = -1

        out = np.empty(flat.shape, dtype=np.float64)
        for i, label in enumerate(flat):
            if not np.isfinite(label):
                out[i] = np.nan
            elif int(label) == -1:
                out[i] = -1.0
            else:
                out[i] = float(label_mapping[int(label)])
        out = out.reshape(original_shape)
        if np.issubdtype(cluster_labels.dtype, np.integer) and np.all(
            np.isfinite(flat)
        ):
            return np.round(out).astype(np.int64)
        return out

    flat = np.full(500_000, -1.0, dtype=np.float64)
    flat[rng.integers(0, flat.size, size=80_000)] = rng.integers(0, 400, size=80_000)
    flat[rng.integers(0, flat.size, size=5_000)] = np.nan
    np.testing.assert_array_equal(
        sorted_cluster_labels(flat),
        _reference(flat),
    )


class _Store:
    def __init__(self, data: xr.Dataset, time_dim: str):
        self.data = data
        self.time_dim = time_dim


def test_healpix_member_support_basic():
    T, nside, npix = 5, 4, 12 * 4**2
    coords = {"time": np.arange(T), "hp_pixel": np.arange(npix)}
    a = np.full((T, npix), np.nan, dtype=np.float32)
    b = np.full((T, npix), np.nan, dtype=np.float32)
    a[2, 10:14] = 0
    a[3, 10:14] = 0
    b[2, 12:16] = 0
    b[3, 12:16] = 0
    ds = xr.Dataset(
        {
            "model_a_cluster": (("time", "hp_pixel"), a),
            "model_b_cluster": (("time", "hp_pixel"), b),
        },
        coords=coords,
    )
    out = run_healpix_member_support_consensus(
        _Store(ds, "time"),
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        nside=nside,
        min_cluster_area=1,
        show_progress=False,
    )
    labels = out["clusters"].values
    rate = out["rate"].values
    assert labels.shape == (T, npix)
    assert rate.shape == (T, npix)
    assert np.any(labels >= 0)


def test_healpix_infinite_spatial_tolerance_temporal_only():
    """Distant same-time HEALPix detections agree under ``spatial_tolerance=np.inf``."""
    T, nside, npix = 3, 4, 12 * 4**2
    coords = {"time": np.arange(T), "hp_pixel": np.arange(npix)}
    a = np.full((T, npix), np.nan, dtype=np.float32)
    b = np.full((T, npix), np.nan, dtype=np.float32)
    pix_a, pix_b = 0, npix - 1
    a[1, pix_a] = 0
    b[1, pix_b] = 0
    ds = xr.Dataset(
        {
            "model_a_cluster": (("time", "hp_pixel"), a),
            "model_b_cluster": (("time", "hp_pixel"), b),
        },
        coords=coords,
    )
    store = _Store(ds, "time")

    out_exact = run_healpix_member_support_consensus(
        store,
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        nside=nside,
        min_cluster_area=None,
        show_progress=False,
    )
    assert not np.any(out_exact["clusters"].values >= 0)

    out_inf = run_healpix_member_support_consensus(
        store,
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=np.inf,
        nside=nside,
        min_cluster_area=None,
        show_progress=False,
    )
    labels = out_inf["clusters"].values
    assert labels[1, pix_a] >= 0
    assert labels[1, pix_b] >= 0
    assert labels[1, pix_a] == labels[1, pix_b]

    # Dilation: any detection at t fills the whole time slice after temporal window
    rows, cols = build_ring1_spatial_edges(nside)
    mask = np.zeros((T, npix), dtype=bool)
    mask[1, pix_a] = True
    dilated = _dilate_healpix_support_mask(
        mask,
        temporal_tolerance=0,
        spatial_tolerance=np.inf,
        spatial_rows=rows,
        spatial_cols=cols,
    )
    assert dilated[1].all()
    assert not dilated[0].any()
    assert not dilated[2].any()


def test_healpix_infinite_temporal_tolerance_spatial_only():
    """Distant-time HEALPix detections agree under ``temporal_tolerance=np.inf``."""
    T, nside, npix = 4, 4, 12 * 4**2
    rows, cols = build_ring1_spatial_edges(nside)
    nbrs0 = set(cols[rows == 0].tolist()) | set(rows[cols == 0].tolist())
    far = next(i for i in range(npix) if i != 0 and i not in nbrs0)
    near = next(iter(nbrs0))
    coords = {"time": np.arange(T), "hp_pixel": np.arange(npix)}
    a = np.full((T, npix), np.nan, dtype=np.float32)
    b = np.full((T, npix), np.nan, dtype=np.float32)
    a[0, 0] = 0
    a[0, far] = 0
    b[3, 0] = 0
    b[3, far] = 0
    ds = xr.Dataset(
        {
            "model_a_cluster": (("time", "hp_pixel"), a),
            "model_b_cluster": (("time", "hp_pixel"), b),
        },
        coords=coords,
    )
    store = _Store(ds, "time")

    out_exact = run_healpix_member_support_consensus(
        store,
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        nside=nside,
        min_cluster_area=None,
        show_progress=False,
    )
    assert not np.any(out_exact["clusters"].values >= 0)

    out_inf = run_healpix_member_support_consensus(
        store,
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=1.0,
        temporal_tolerance=np.inf,
        spatial_tolerance=0,
        nside=nside,
        min_cluster_area=None,
        show_progress=False,
    )
    labels = out_inf["clusters"].values
    assert labels[0, 0] >= 0
    assert labels[3, 0] == labels[0, 0]
    assert labels[0, far] == labels[3, far]
    assert labels[0, 0] != labels[0, far]

    a_near = np.full((T, npix), np.nan, dtype=np.float32)
    b_near = np.full((T, npix), np.nan, dtype=np.float32)
    a_near[0, 0] = 0
    b_near[3, near] = 0
    near_ds = xr.Dataset(
        {
            "model_a_cluster": (("time", "hp_pixel"), a_near),
            "model_b_cluster": (("time", "hp_pixel"), b_near),
        },
        coords=coords,
    )
    near_store = _Store(near_ds, "time")
    out_apart = run_healpix_member_support_consensus(
        near_store,
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=1.0,
        temporal_tolerance=np.inf,
        spatial_tolerance=0,
        nside=nside,
        min_cluster_area=None,
        show_progress=False,
    )
    assert not np.any(out_apart["clusters"].values >= 0)
    out_near = run_healpix_member_support_consensus(
        near_store,
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=1.0,
        temporal_tolerance=np.inf,
        spatial_tolerance=1,
        nside=nside,
        min_cluster_area=None,
        show_progress=False,
    )
    near_labels = out_near["clusters"].values
    assert near_labels[0, 0] >= 0
    assert near_labels[3, near] == near_labels[0, 0]

    mask = np.zeros((T, npix), dtype=bool)
    mask[0, 0] = True
    exact = _dilate_healpix_support_mask(
        mask,
        temporal_tolerance=np.inf,
        spatial_tolerance=0,
        spatial_rows=rows,
        spatial_cols=cols,
    )
    assert exact[:, 0].all()
    assert int(exact.sum()) == T

    wide = _dilate_healpix_support_mask(
        mask,
        temporal_tolerance=np.inf,
        spatial_tolerance=1,
        spatial_rows=rows,
        spatial_cols=cols,
    )
    assert wide[:, 0].all()
    assert wide[:, near].all()
    assert not wide[:, far].any()

    with pytest.raises(ValueError, match="cannot both be np.inf"):
        run_healpix_member_support_consensus(
            store,
            cluster_vars=["model_a_cluster", "model_b_cluster"],
            min_consensus=1.0,
            temporal_tolerance=np.inf,
            spatial_tolerance=np.inf,
            nside=nside,
            show_progress=False,
        )
