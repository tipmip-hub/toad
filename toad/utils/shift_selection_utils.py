"""
Utility functions for shift selection. Used primarily in clustering.
"""

from typing import Literal

from toad.utils import DEFAULT_SHIFT_THRESHOLD

import numpy as np
import xarray as xr
from numba import njit


@njit(cache=True)
def _plateau_magnitude(
    base: np.ndarray,
    peak: int,
    L: int,
    G: int,
) -> float:
    """Peak-anchored |Δbase| with outer half-width ``L`` and guard ``G``.

    Uses ``mean[peak-L, peak-G)`` vs ``mean[peak+G+1, peak+L+1)``. Independent of
    dts episode edges. Requires ``L > G >= 0``; returns NaN if either side is empty.
    """
    if L <= G or G < 0:
        return np.nan

    n = base.size
    pre_lo = peak - L
    pre_hi = peak - G
    post_lo = peak + G + 1
    post_hi = peak + L + 1

    if pre_lo < 0:
        pre_lo = 0
    if pre_hi < 0:
        pre_hi = 0
    if pre_hi > n:
        pre_hi = n
    if pre_lo > pre_hi:
        pre_lo = pre_hi

    if post_lo < 0:
        post_lo = 0
    if post_hi < 0:
        post_hi = 0
    if post_hi > n:
        post_hi = n
    if post_lo > post_hi:
        post_lo = post_hi

    pre_sum = 0.0
    pre_count = 0
    for t in range(pre_lo, pre_hi):
        v = base[t]
        if not np.isnan(v):
            pre_sum += v
            pre_count += 1

    post_sum = 0.0
    post_count = 0
    for t in range(post_lo, post_hi):
        v = base[t]
        if not np.isnan(v):
            post_sum += v
            post_count += 1

    if pre_count == 0 or post_count == 0:
        return np.nan

    return abs(post_sum / post_count - pre_sum / pre_count)


@njit(cache=True)
def _peaks_local_for_ts_filtered(
    ts: np.ndarray,
    base: np.ndarray,
    thr: float,
    min_magnitude: float,
    L: int,
    G: int,
    eps: float = 1e-12,
):
    """Local dts peaks whose plateau |Δbase| meets ``min_magnitude``."""
    n = ts.size
    idxs = np.empty(n, dtype=np.int64)
    sgns = np.empty(n, dtype=np.int8)
    k = 0
    i = 0

    while i < n:
        while i < n:
            v = ts[i]
            if not np.isnan(v) and (abs(v) > thr):
                break
            i += 1
        if i >= n:
            break

        max_abs = abs(ts[i])
        plat_start = i
        plat_end = i
        i += 1

        while i < n:
            v = ts[i]
            if np.isnan(v):
                break
            av = abs(v)
            if not (av > thr):
                break
            if av > max_abs + eps:
                max_abs = av
                plat_start = i
                plat_end = i
            elif abs(av - max_abs) <= eps:
                plat_end = i
            i += 1

        max_idx = plat_start + (plat_end - plat_start) // 2
        if max_abs > thr:
            mag = _plateau_magnitude(base, max_idx, L, G)
            if not np.isnan(mag) and mag >= min_magnitude:
                idxs[k] = max_idx
                sgns[k] = np.int8(-1 if np.signbit(ts[max_idx]) else 1)
                k += 1

    return idxs[:k], sgns[:k]


@njit(cache=True)
def _episode_magnitudes_for_ts(
    ts: np.ndarray,
    base: np.ndarray,
    thr: float,
    L: int,
    G: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Collect plateau |Δbase| for every local dts episode with |dts| > ``thr``."""
    n = ts.size
    mags = np.empty(n, dtype=np.float64)
    k = 0
    i = 0

    while i < n:
        while i < n:
            v = ts[i]
            if not np.isnan(v) and (abs(v) > thr):
                break
            i += 1
        if i >= n:
            break

        max_abs = abs(ts[i])
        plat_start = i
        plat_end = i
        i += 1

        while i < n:
            v = ts[i]
            if np.isnan(v):
                break
            av = abs(v)
            if not (av > thr):
                break
            if av > max_abs + eps:
                max_abs = av
                plat_start = i
                plat_end = i
            elif abs(av - max_abs) <= eps:
                plat_end = i
            i += 1

        max_idx = plat_start + (plat_end - plat_start) // 2
        if max_abs > thr:
            mag = _plateau_magnitude(base, max_idx, L, G)
            if not np.isnan(mag):
                mags[k] = mag
                k += 1

    return mags[:k]


def collect_episode_magnitudes(
    shifts: xr.DataArray,
    base: xr.DataArray,
    time_dim: str,
    shift_threshold: float,
    *,
    L: int = 25,
    G: int = 10,
) -> np.ndarray:
    """Flat array of plateau |Δbase| for every local dts episode above ``shift_threshold``.

    Uses the same peak-anchored ``L``/``G`` means as ``min_event_magnitude`` filtering
    in clustering, but does not apply a magnitude cutoff.
    """
    if L <= G or G < 0:
        raise ValueError("require L > G >= 0")

    space_dims = tuple(d for d in shifts.dims if d != time_dim)
    da_t_first = shifts.transpose(time_dim, *space_dims)
    base_t_first = base.transpose(time_dim, *space_dims)

    vals = np.asarray(da_t_first.data)
    base_vals = np.asarray(base_t_first.data)
    T = vals.shape[0]
    space_shape = vals.shape[1:]
    P = int(np.prod(space_shape)) if space_shape else 1

    dts_TP = vals.reshape(T, P)
    base_TP = base_vals.reshape(T, P)
    thr = float(shift_threshold)

    chunks: list[np.ndarray] = []
    for p in range(P):
        mags = _episode_magnitudes_for_ts(
            dts_TP[:, p], base_TP[:, p], thr, int(L), int(G)
        )
        if mags.size:
            chunks.append(mags)
    if not chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(chunks)


@njit(cache=True)
def _peak_global_for_ts_filtered(
    ts: np.ndarray,
    base: np.ndarray,
    thr: float,
    min_magnitude: float,
    L: int,
    G: int,
    eps: float = 1e-12,
):
    """Global dts peak among episodes whose plateau |Δbase| meets ``min_magnitude``."""
    n = ts.size
    best_idx = np.int64(-1)
    best_sgn = np.int8(0)
    best_abs = -1.0
    i = 0

    while i < n:
        while i < n:
            v = ts[i]
            if not np.isnan(v) and (abs(v) > thr):
                break
            i += 1
        if i >= n:
            break

        max_abs = abs(ts[i])
        plat_start = i
        plat_end = i
        i += 1

        while i < n:
            v = ts[i]
            if np.isnan(v):
                break
            av = abs(v)
            if not (av > thr):
                break
            if av > max_abs + eps:
                max_abs = av
                plat_start = i
                plat_end = i
            elif abs(av - max_abs) <= eps:
                plat_end = i
            i += 1

        max_idx = plat_start + (plat_end - plat_start) // 2
        if max_abs > thr:
            mag = _plateau_magnitude(base, max_idx, L, G)
            if not np.isnan(mag) and mag >= min_magnitude and max_abs > best_abs + eps:
                best_abs = max_abs
                best_idx = np.int64(max_idx)
                best_sgn = np.int8(-1 if np.signbit(ts[max_idx]) else 1)

    return best_idx, best_sgn


@njit(cache=True)
def _compute_local_mask_TP_filtered(
    dts_TP: np.ndarray,
    base_TP: np.ndarray,
    thr: float,
    min_magnitude: float,
    L: int,
    G: int,
    out_TP: np.ndarray,
):
    T, P = dts_TP.shape
    for p in range(P):
        ts = dts_TP[:, p]
        base = base_TP[:, p]
        idxs, sgns = _peaks_local_for_ts_filtered(ts, base, thr, min_magnitude, L, G)
        for m in range(idxs.size):
            out_TP[idxs[m], p] = sgns[m]


@njit(cache=True)
def _compute_global_mask_TP_filtered(
    dts_TP: np.ndarray,
    base_TP: np.ndarray,
    thr: float,
    min_magnitude: float,
    L: int,
    G: int,
    out_TP: np.ndarray,
):
    T, P = dts_TP.shape
    for p in range(P):
        ts = dts_TP[:, p]
        base = base_TP[:, p]
        idx, sgn = _peak_global_for_ts_filtered(ts, base, thr, min_magnitude, L, G)
        if idx >= 0:
            out_TP[idx, p] = sgn


@njit(cache=True)
def _compute_episode_pass_mask_TP(
    dts_TP: np.ndarray,
    base_TP: np.ndarray,
    thr: float,
    min_magnitude: float,
    L: int,
    G: int,
    out_TP: np.ndarray,
    eps: float = 1e-12,
):
    """Mark timesteps inside dts episodes whose plateau |Δbase| passes."""
    T, P = dts_TP.shape
    for p in range(P):
        ts = dts_TP[:, p]
        base = base_TP[:, p]
        i = 0
        while i < T:
            while i < T:
                v = ts[i]
                if not np.isnan(v) and (abs(v) > thr):
                    break
                i += 1
            if i >= T:
                break

            start = i
            max_abs = abs(ts[i])
            plat_start = i
            plat_end = i
            i += 1
            while i < T:
                v = ts[i]
                if np.isnan(v):
                    break
                av = abs(v)
                if not (av > thr):
                    break
                if av > max_abs + eps:
                    max_abs = av
                    plat_start = i
                    plat_end = i
                elif abs(av - max_abs) <= eps:
                    plat_end = i
                i += 1

            end = i - 1
            max_idx = plat_start + (plat_end - plat_start) // 2
            if max_abs > thr:
                mag = _plateau_magnitude(base, max_idx, L, G)
                if not np.isnan(mag) and mag >= min_magnitude:
                    for t in range(start, end + 1):
                        out_TP[t, p] = 1


@njit(cache=True)
def _peaks_local_for_ts(ts: np.ndarray, thr: float, eps: float = 1e-12):
    """Finds local peaks in segments of a time series where values exceed a threshold.

    For each segment where absolute values exceed the threshold, identifies the maximum
    absolute value peak. For plateaus (consecutive equal maximum values), selects the
    middle point as the peak. NaN values break segments.

    Args:
        ts: 1D numpy array containing the time series data.
        thr: Threshold value that peaks must exceed in absolute value.
        eps: Small value for floating point comparisons. Defaults to 1e-12.

    Returns:
        tuple:
            - idxs (np.ndarray): Array of indices where peaks were found.
            - sgns (np.ndarray): Array of signs (-1 for negative peaks, +1 for positive peaks)
              corresponding to each index.

    Note:
        This is a numba-optimized implementation that uses @njit for performance.
    """
    n = ts.size
    idxs = np.empty(n, dtype=np.int64)  # over-alloc; trimmed later
    sgns = np.empty(n, dtype=np.int8)
    k = 0
    i = 0

    while i < n:
        # Skip below-threshold or NaN
        while i < n:
            v = ts[i]
            if not np.isnan(v) and (abs(v) > thr):
                break
            i += 1
        if i >= n:
            break

        # Start of segment
        max_abs = abs(ts[i])
        plat_start = i
        plat_end = i
        i += 1

        # Walk segment
        while i < n:
            v = ts[i]
            if np.isnan(v):
                break
            av = abs(v)
            if not (av > thr):
                break

            if av > max_abs + eps:
                max_abs = av
                plat_start = i
                plat_end = i
            elif abs(av - max_abs) <= eps:
                plat_end = i
            i += 1

        # Middle of the segment's max plateau
        max_idx = plat_start + (plat_end - plat_start) // 2
        # Verify peak exceeds threshold (safety check, should always be true)
        if max_abs > thr:
            idxs[k] = max_idx
            sgns[k] = np.int8(-1 if np.signbit(ts[max_idx]) else 1)
            k += 1

    return idxs[:k], sgns[:k]


@njit(cache=True)
def _peak_global_for_ts(ts: np.ndarray, thr: float, eps: float = 1e-12):
    """Finds the global peak in a time series using middle-of-plateau tie rule.

    Performs a single pass through the time series to find the global maximum absolute value
    peak that exceeds the threshold. For plateaus (consecutive equal maximum values), the
    middle point is selected as the peak. NaN values break plateaus.

    Args:
        ts: 1D numpy array containing the time series data
        thr: Threshold value that peaks must exceed in absolute value
        eps: Small value for floating point comparisons. Defaults to 1e-12.

    Returns:
        tuple:
            - idx (np.int64): Index of the peak, or -1 if no peak passes threshold
            - sgn (np.int8): Sign of the peak (-1 for negative, +1 for positive, 0 if no peak)

    Note:
        This is a numba-optimized implementation that uses @njit for performance.
    """
    n = ts.size
    max_abs = -1.0
    have_max = False
    plat_start = 0
    plat_end = -1
    in_equal_run = False  # are we currently extending a contiguous max plateau?

    for i in range(n):
        v = ts[i]
        if np.isnan(v):
            in_equal_run = False
            continue
        av = abs(v)

        if av > max_abs + eps:
            max_abs = av
            have_max = True
            plat_start = i
            plat_end = i
            in_equal_run = True
        elif have_max and abs(av - max_abs) <= eps:
            # extend only if contiguous with current max plateau
            if in_equal_run:
                plat_end = i
            in_equal_run = True
        else:
            in_equal_run = False

    if (not have_max) or (max_abs <= thr):
        return np.int64(-1), np.int8(0)

    mid = plat_start + (plat_end - plat_start) // 2
    return np.int64(mid), np.int8(-1 if np.signbit(ts[mid]) else 1)


@njit(cache=True)
def _compute_local_mask_TP(dts_TP: np.ndarray, thr: float, out_TP: np.ndarray):
    """Computes local peak mask for time series data.

    For each time series in dts_TP, identifies local peaks within segments where values exceed the threshold.
    Peaks are marked in out_TP as -1 for negative peaks and +1 for positive peaks. For plateaus (consecutive
    equal maximum values), only the middle point is marked as a peak.

    Args:
        dts_TP: Input array of shape (T, P) containing P time series of length T.
        thr: Threshold value that peaks must exceed in absolute value.
        out_TP: Output array of shape (T, P) that will be modified in-place.
            Values will be in {-1, 0, +1} indicating peak signs.

    Note:
        This is a numba-optimized implementation that modifies out_TP in-place.
        The @njit decorator compiles this function to machine code.
    """
    T, P = dts_TP.shape
    for p in range(P):
        ts = dts_TP[:, p]
        idxs, sgns = _peaks_local_for_ts(ts, thr)
        for m in range(idxs.size):
            out_TP[idxs[m], p] = sgns[m]


@njit(cache=True)
def _compute_global_mask_TP(dts_TP: np.ndarray, thr: float, out_TP: np.ndarray):
    """Computes global peak mask for time series data.

    For each time series in dts_TP, finds the global peak and marks it in out_TP.
    A peak is marked with -1 for negative peaks or +1 for positive peaks that exceed
    the threshold. Only the middle point of the maximum plateau is marked.

    Args:
        dts_TP: Input array of shape (T, P) containing P time series of length T.
        thr: Threshold value that peaks must exceed in absolute value.
        out_TP: Output array of shape (T, P) that will be modified in-place.
            Values will be in {-1, 0, +1} indicating peak signs.

    Note:
        This is a numba-optimized implementation that modifies out_TP in-place.
        The @njit decorator compiles this function to machine code.
    """
    T, P = dts_TP.shape
    for p in range(P):
        ts = dts_TP[:, p]
        idx, sgn = _peak_global_for_ts(ts, thr)
        if idx >= 0:
            out_TP[idx, p] = sgn


def _compute_dts_peak_sign_mask(
    shifts: xr.DataArray,
    time_dim: str,
    shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    shift_selection: Literal["local", "global"] | str = "local",
    base: xr.DataArray | None = None,
    min_event_magnitude: float | None = None,
    min_event_magnitude_L: int = 25,
    min_event_magnitude_G: int = 10,
) -> xr.DataArray:
    """Computes a dense mask indicating peak signs in the shifts data.

    Creates an int8 mask with values in {-1, 0, +1} marking peaks in the shifts data.
    For local selection, marks the middle of max-|value| plateaus within each |shifts|>threshold segment.
    For global selection, marks the middle of the global max-|value| plateau (only if max > threshold).
    NaN values break segments/plateaus.

    When ``min_event_magnitude`` is set, episodes must also show at least that absolute
    change in ``base`` using peak-anchored plateau means with outer half-width
    ``min_event_magnitude_L`` and guard ``min_event_magnitude_G``. ``base`` must be
    supplied and aligned with ``shifts``.
    """
    if shift_selection not in ("local", "global"):
        raise ValueError('shift_selection must be "local" or "global"')
    if min_event_magnitude is not None and base is None:
        raise ValueError("base is required when min_event_magnitude is set")
    if min_event_magnitude_L <= min_event_magnitude_G or min_event_magnitude_G < 0:
        raise ValueError("require min_event_magnitude_L > min_event_magnitude_G >= 0")

    space_dims = tuple(d for d in shifts.dims if d != time_dim)
    da_t_first = shifts.transpose(time_dim, *space_dims)

    vals = np.asarray(da_t_first.data)
    T = vals.shape[0]
    space_shape = vals.shape[1:]
    P = int(np.prod(space_shape)) if space_shape else 1

    dts_TP = vals.reshape(T, P)
    out_TP = np.zeros((T, P), dtype=np.int8)

    if min_event_magnitude is not None:
        base_t_first = base.transpose(time_dim, *space_dims)
        base_TP = np.asarray(base_t_first.data).reshape(T, P)
        min_mag = float(min_event_magnitude)
        L = int(min_event_magnitude_L)
        G = int(min_event_magnitude_G)
        if shift_selection == "local":
            _compute_local_mask_TP_filtered(
                dts_TP, base_TP, float(shift_threshold), min_mag, L, G, out_TP
            )
        else:
            _compute_global_mask_TP_filtered(
                dts_TP, base_TP, float(shift_threshold), min_mag, L, G, out_TP
            )
    elif shift_selection == "local":
        _compute_local_mask_TP(dts_TP, float(shift_threshold), out_TP)
    else:
        _compute_global_mask_TP(dts_TP, float(shift_threshold), out_TP)

    out = out_TP.reshape((T, *space_shape))
    out_da_t_first = xr.DataArray(
        out,
        coords=da_t_first.coords,
        dims=(time_dim, *space_dims),
        name=shifts.name,
    )
    return out_da_t_first.transpose(*shifts.dims)


def _compute_episode_pass_mask(
    shifts: xr.DataArray,
    base: xr.DataArray,
    time_dim: str,
    shift_threshold: float,
    min_event_magnitude: float,
    min_event_magnitude_L: int = 25,
    min_event_magnitude_G: int = 10,
) -> xr.DataArray:
    """Boolean mask of timesteps inside magnitude-qualified dts episodes."""
    if min_event_magnitude_L <= min_event_magnitude_G or min_event_magnitude_G < 0:
        raise ValueError("require min_event_magnitude_L > min_event_magnitude_G >= 0")

    space_dims = tuple(d for d in shifts.dims if d != time_dim)
    da_t_first = shifts.transpose(time_dim, *space_dims)
    base_t_first = base.transpose(time_dim, *space_dims)

    vals = np.asarray(da_t_first.data)
    base_vals = np.asarray(base_t_first.data)
    T = vals.shape[0]
    space_shape = vals.shape[1:]
    P = int(np.prod(space_shape)) if space_shape else 1

    dts_TP = vals.reshape(T, P)
    base_TP = base_vals.reshape(T, P)
    out_TP = np.zeros((T, P), dtype=np.int8)
    _compute_episode_pass_mask_TP(
        dts_TP,
        base_TP,
        float(shift_threshold),
        float(min_event_magnitude),
        int(min_event_magnitude_L),
        int(min_event_magnitude_G),
        out_TP,
    )

    out = out_TP.reshape((T, *space_shape))
    out_da = xr.DataArray(
        out.astype(bool),
        coords=da_t_first.coords,
        dims=(time_dim, *space_dims),
    )
    return out_da.transpose(*shifts.dims)


@njit(cache=True)
def _episode_overlap_mask_for_ts(
    ts: np.ndarray,
    thr: float,
    win_start: int,
    win_end: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Mark full dts episodes that overlap ``[win_start, win_end]`` (inclusive indices)."""
    n = ts.size
    out = np.zeros(n, dtype=np.bool_)
    i = 0

    while i < n:
        while i < n:
            v = ts[i]
            if not np.isnan(v) and (abs(v) > thr):
                break
            i += 1
        if i >= n:
            break

        seg_start = i
        i += 1
        while i < n:
            v = ts[i]
            if np.isnan(v) or not (abs(v) > thr):
                break
            i += 1
        seg_end = i - 1

        if seg_start <= win_end and seg_end >= win_start:
            for t in range(seg_start, seg_end + 1):
                out[t] = True

    return out


@njit(cache=True)
def _compute_episode_overlap_mask_TP(
    dts_TP: np.ndarray,
    thr: float,
    win_start: int,
    win_end: int,
    out_TP: np.ndarray,
):
    T, P = dts_TP.shape
    for p in range(P):
        mask = _episode_overlap_mask_for_ts(dts_TP[:, p], thr, win_start, win_end)
        for t in range(T):
            if mask[t]:
                out_TP[t, p] = 1


def _compute_episode_overlap_mask(
    shifts: xr.DataArray,
    time_dim: str,
    shift_threshold: float,
    window_start: int,
    window_end: int,
) -> xr.DataArray:
    """Boolean mask of timesteps in dts episodes overlapping a time index window."""
    space_dims = tuple(d for d in shifts.dims if d != time_dim)
    da_t_first = shifts.transpose(time_dim, *space_dims)

    vals = np.asarray(da_t_first.data)
    T = vals.shape[0]
    space_shape = vals.shape[1:]
    P = int(np.prod(space_shape)) if space_shape else 1

    dts_TP = vals.reshape(T, P)
    out_TP = np.zeros((T, P), dtype=np.int8)
    _compute_episode_overlap_mask_TP(
        dts_TP,
        float(shift_threshold),
        int(window_start),
        int(window_end),
        out_TP,
    )

    out = out_TP.reshape((T, *space_shape))
    out_da = xr.DataArray(
        out.astype(bool),
        coords=da_t_first.coords,
        dims=(time_dim, *space_dims),
    )
    return out_da.transpose(*shifts.dims)
