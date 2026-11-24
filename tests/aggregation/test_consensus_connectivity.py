"""Connectivity of kept voxels in the member-support consensus (audit item 5.3)."""

import numpy as np
import pytest
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from toad.clustering import sorted_cluster_labels
from toad.postprocessing.member_support_consensus import (
    _build_grid_context,
    _label_retained_voxels,
)

T, Y, X = 40, 10, 12

# Labels produced by the code before the float64/integer-scaling fix (toad 1.0.8),
# for random sparse keep masks: ``rng = np.random.default_rng(seed)``,
# ``keep = rng.random(T * Y * X) < 0.012``. Key: (temporal_tol, spatial_tol, seed).
# Value: (flat indices of kept voxels, sorted consensus labels at those voxels).
# Tolerances 4/2 (KD-tree path, exact in float32) and 1/1 (scipy.ndimage.label path)
# must stay bit-identical.
OLD_LABELS = {
    (4, 2, 0): (
        [
            11,
            150,
            196,
            269,
            403,
            600,
            608,
            657,
            781,
            850,
            855,
            921,
            1148,
            1185,
            1271,
            1273,
            1308,
            1327,
            1446,
            1502,
            1526,
            1555,
            1607,
            1678,
            1706,
            1776,
            1808,
            1839,
            1945,
            1966,
            2104,
            2238,
            2318,
            2454,
            2602,
            2693,
            2761,
            3108,
            3274,
            3291,
            3309,
            3348,
            3377,
            3423,
            3438,
            3697,
            3742,
            3744,
            3835,
            4077,
            4139,
            4212,
            4264,
            4419,
            4473,
            4523,
            4528,
            4539,
            4675,
            4723,
        ],
        [
            19,
            0,
            10,
            0,
            0,
            11,
            0,
            0,
            1,
            0,
            12,
            0,
            0,
            0,
            0,
            1,
            1,
            5,
            5,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            5,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            14,
            1,
            15,
            4,
            9,
            8,
            16,
            4,
            17,
            8,
            18,
            4,
            9,
            7,
            6,
            6,
            2,
            7,
            3,
            3,
            2,
            2,
            3,
            3,
            13,
            2,
        ],
    ),
    (4, 2, 1): (
        [
            61,
            93,
            323,
            458,
            622,
            774,
            800,
            813,
            828,
            900,
            1329,
            1337,
            1355,
            1375,
            1514,
            1517,
            1553,
            1574,
            1721,
            1799,
            2001,
            2046,
            2098,
            2111,
            2308,
            2373,
            2530,
            2626,
            2642,
            2713,
            2752,
            2774,
            2777,
            2999,
            3165,
            3200,
            3205,
            3226,
            3245,
            3254,
            3429,
            3674,
            3690,
            3700,
            3764,
            3835,
            3933,
            3970,
            4099,
            4196,
            4217,
            4401,
            4451,
            4770,
        ],
        [
            16,
            2,
            2,
            4,
            8,
            2,
            2,
            2,
            4,
            9,
            6,
            3,
            6,
            3,
            11,
            3,
            10,
            12,
            3,
            13,
            0,
            1,
            0,
            0,
            1,
            0,
            14,
            0,
            1,
            0,
            15,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            5,
            0,
            0,
            0,
            0,
            5,
            7,
        ],
    ),
    (1, 1, 0): (
        [
            11,
            150,
            196,
            269,
            403,
            600,
            608,
            657,
            781,
            850,
            855,
            921,
            1148,
            1185,
            1271,
            1273,
            1308,
            1327,
            1446,
            1502,
            1526,
            1555,
            1607,
            1678,
            1706,
            1776,
            1808,
            1839,
            1945,
            1966,
            2104,
            2238,
            2318,
            2454,
            2602,
            2693,
            2761,
            3108,
            3274,
            3291,
            3309,
            3348,
            3377,
            3423,
            3438,
            3697,
            3742,
            3744,
            3835,
            4077,
            4139,
            4212,
            4264,
            4419,
            4473,
            4523,
            4528,
            4539,
            4675,
            4723,
        ],
        [
            53,
            1,
            26,
            1,
            25,
            24,
            23,
            22,
            21,
            20,
            19,
            18,
            17,
            15,
            6,
            14,
            13,
            3,
            3,
            12,
            11,
            10,
            9,
            8,
            2,
            7,
            27,
            2,
            16,
            28,
            42,
            52,
            51,
            50,
            49,
            48,
            47,
            46,
            45,
            4,
            44,
            43,
            41,
            4,
            30,
            40,
            39,
            38,
            37,
            36,
            35,
            34,
            33,
            0,
            32,
            31,
            0,
            0,
            29,
            5,
        ],
    ),
    (1, 1, 1): (
        [
            61,
            93,
            323,
            458,
            622,
            774,
            800,
            813,
            828,
            900,
            1329,
            1337,
            1355,
            1375,
            1514,
            1517,
            1553,
            1574,
            1721,
            1799,
            2001,
            2046,
            2098,
            2111,
            2308,
            2373,
            2530,
            2626,
            2642,
            2713,
            2752,
            2774,
            2777,
            2999,
            3165,
            3200,
            3205,
            3226,
            3245,
            3254,
            3429,
            3674,
            3690,
            3700,
            3764,
            3835,
            3933,
            3970,
            4099,
            4196,
            4217,
            4401,
            4451,
            4770,
        ],
        [
            50,
            24,
            23,
            22,
            21,
            20,
            2,
            2,
            19,
            18,
            17,
            16,
            15,
            13,
            4,
            12,
            11,
            10,
            9,
            8,
            7,
            6,
            1,
            1,
            5,
            25,
            14,
            26,
            0,
            28,
            49,
            0,
            48,
            47,
            46,
            45,
            44,
            43,
            42,
            41,
            40,
            39,
            38,
            37,
            36,
            35,
            34,
            33,
            32,
            31,
            30,
            29,
            27,
            3,
        ],
    ),
}


def _context(T_=T, Y_=Y, X_=X):
    sample = xr.DataArray(
        np.zeros((T_, Y_, X_)),
        dims=("time", "y", "x"),
        coords={"time": np.arange(T_), "y": np.arange(Y_), "x": np.arange(X_)},
    )
    return _build_grid_context(
        sample, spatial_dims=("y", "x"), time_dim="time", stitch_meridian=False
    )


def _labels(keep, ctx, tt, ts):
    return sorted_cluster_labels(
        _label_retained_voxels(
            keep, context=ctx, temporal_tolerance=tt, spatial_tolerance=ts
        )
    )


def _brute_force_labels(keep, ctx, tt, ts):
    """Reference: link every pair with |dt| <= tt and |dy|, |dx| <= ts (integers)."""
    tt, ts = max(1, tt), max(1, ts)
    nodes = np.flatnonzero(keep)
    t = nodes // ctx.n_space
    s = nodes % ctx.n_space
    y, x = s // ctx.x_len, s % ctx.x_len
    near = (
        (np.abs(t[:, None] - t[None, :]) <= tt)
        & (np.abs(y[:, None] - y[None, :]) <= ts)
        & (np.abs(x[:, None] - x[None, :]) <= ts)
    )
    r, c = np.nonzero(near)
    g = coo_matrix((np.ones(r.size), (r, c)), shape=(nodes.size, nodes.size))
    _, lab = connected_components(g, directed=False)
    out = np.full(keep.size, -1, dtype=np.int64)
    out[nodes] = lab
    return sorted_cluster_labels(out)


@pytest.mark.parametrize("key", sorted(OLD_LABELS))
def test_labels_unchanged_for_exact_tolerances(key):
    tt, ts, seed = key
    idx, expected = OLD_LABELS[key]
    rng = np.random.default_rng(seed)
    keep = rng.random(T * Y * X) < 0.012
    assert np.flatnonzero(keep).tolist() == idx
    lab = _labels(keep, _context(), tt, ts)
    np.testing.assert_array_equal(lab[idx], np.asarray(expected))
    assert np.all(lab[~keep] == -1)


@pytest.mark.parametrize("tt,ts", [(3, 3), (3, 1), (1, 3), (5, 2), (6, 3), (7, 5)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_labels_match_brute_force_for_any_tolerance(tt, ts, seed):
    rng = np.random.default_rng(seed)
    keep = rng.random(T * Y * X) < 0.012
    ctx = _context()
    np.testing.assert_array_equal(
        _labels(keep, ctx, tt, ts), _brute_force_labels(keep, ctx, tt, ts)
    )


@pytest.mark.parametrize("axis", ["t", "y", "x"])
def test_tolerance_three_links_every_pair_exactly_three_apart(axis):
    T_, Y_, X_ = 60, 3, 60
    ctx = _context(T_, Y_, X_)
    for a in range(0, (T_ if axis == "t" else (Y_ if axis == "y" else X_)) - 3):
        keep = np.zeros((T_, Y_, X_), dtype=bool)
        if axis == "t":
            keep[a, 1, 1] = keep[a + 3, 1, 1] = True
        elif axis == "y":
            keep[5, a, 1] = keep[5, a + 3, 1] = True
        else:
            keep[5, 1, a] = keep[5, 1, a + 3] = True
        tt, ts = (3, 1) if axis == "t" else (1, 3)
        lab = _labels(keep.ravel(), ctx, tt, ts)
        assert np.unique(lab[lab >= 0]).tolist() == [0], (axis, a)
        # ...and a pair four apart is never linked.
        keep4 = np.zeros_like(keep)
        if axis == "t":
            keep4[a, 1, 1] = True
            if a + 4 < T_:
                keep4[a + 4, 1, 1] = True
        elif axis == "x":
            keep4[5, 1, a] = True
            if a + 4 < X_:
                keep4[5, 1, a + 4] = True
        else:
            continue
        lab4 = _labels(keep4.ravel(), ctx, tt, ts)
        assert np.unique(lab4[lab4 >= 0]).size == int(keep4.sum()), (axis, a)
