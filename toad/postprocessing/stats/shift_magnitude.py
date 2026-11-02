"""Per-event and cluster-level shift magnitude (ordered extremes).

For an event at index ``t_star`` with ASDETECT ``lmin`` / ``lmax``:

1. Smooth ``y`` with a centred running mean of width ``lmin`` (truncated at
   the ends; no NaN padding).
2. Windows (inclusive, truncated): pre ``[t* - lmax, t*]``, post
   ``[t*, t* + lmax]``. If other detected events exist in the same cell,
   cut at the midpoint between events (same rule as the reference
   ``neighbours=`` truncation).
3. Sign from ``dts[t*]``. Magnitude on the smoothed series::

       negative:  magnitude = min(smoothed[post]) - max(smoothed[pre])
       positive:  magnitude = max(smoothed[post]) - min(smoothed[pre])

4. ``duration`` is the time-coordinate difference between those two extreme
   points. Events with fewer than ``lmin`` finite points in either window are
   flagged ``edge``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from numba import njit

from toad.utils import DEFAULT_SHIFT_THRESHOLD, _attrs
from toad.utils.shift_selection_utils import _compute_dts_peak_sign_mask

# mag_flag codes (dense int8; -1 = no event at this spacetime point)
FLAG_NONE = np.int8(-1)
FLAG_OK = np.int8(0)
FLAG_EDGE = np.int8(1)

_FLAG_LABELS = {
    int(FLAG_NONE): "none",
    int(FLAG_OK): "ok",
    int(FLAG_EDGE): "edge",
}

__all__ = [
    "event_shift_magnitude",
    "compute_shift_magnitudes",
    "aggregate_shift_magnitudes_by_cluster",
    "resolve_lmin_lmax",
    "magnitude_var_names",
]


def resolve_lmin_lmax(shifts: xr.DataArray, n_time: int) -> tuple[int, int]:
    """Read ASDETECT ``lmin``/``lmax`` from shifts attrs; default ``lmax = n // 3``.

    Accepts bare ``lmin``/``lmax`` or the ``shifts_``-prefixed keys written by
    ``compute_shifts``.
    """
    lmin = shifts.attrs.get("lmin", shifts.attrs.get("shifts_lmin", 5))
    lmax = shifts.attrs.get("lmax", shifts.attrs.get("shifts_lmax", None))
    lmin = int(lmin) if lmin is not None else 5
    if lmax is None:
        lmax = n_time // 3
    else:
        lmax = int(lmax)
    if lmin < 1:
        raise ValueError(f"lmin must be >= 1; got {lmin}")
    if lmax < lmin:
        raise ValueError(f"lmax ({lmax}) must be >= lmin ({lmin})")
    return lmin, lmax


@njit(cache=True)
def _centred_running_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centred mean of width ``window``; truncated at ends (no NaN padding)."""
    n = y.size
    out = np.empty(n, dtype=np.float64)
    half_left = (window - 1) // 2
    half_right = window // 2
    for i in range(n):
        lo = i - half_left
        if lo < 0:
            lo = 0
        hi = i + half_right
        if hi > n - 1:
            hi = n - 1
        s = 0.0
        c = 0
        for t in range(lo, hi + 1):
            v = y[t]
            if not np.isnan(v):
                s += v
                c += 1
        if c == 0:
            out[i] = np.nan
        else:
            out[i] = s / c
    return out


@njit(cache=True)
def _window_bounds(
    t_star: int,
    n: int,
    lmax: int,
    prev_ev: int,
    next_ev: int,
) -> tuple[int, int, int, int]:
    """Return ``(pre_lo, pre_hi, post_lo, post_hi)`` with neighbour midpoints.

    ``prev_ev`` / ``next_ev`` are ``-1`` when absent. Midpoint rule matches the
    reference ``neighbours=`` truncation: previous event raises ``pre_lo`` to
    ``(prev + t*) // 2 + 1``; next event lowers ``post_hi`` to ``(next + t*) // 2``.
    """
    pre_lo = t_star - lmax
    if pre_lo < 0:
        pre_lo = 0
    if prev_ev >= 0:
        mid = (prev_ev + t_star) // 2 + 1
        if mid > pre_lo:
            pre_lo = mid
    pre_hi = t_star
    post_lo = t_star
    post_hi = t_star + lmax
    if post_hi > n - 1:
        post_hi = n - 1
    if next_ev >= 0:
        mid = (next_ev + t_star) // 2
        if mid < post_hi:
            post_hi = mid
    return pre_lo, pre_hi, post_lo, post_hi


@njit(cache=True)
def _ordered_extreme_delta(
    y: np.ndarray,
    sm: np.ndarray,
    time_vals: np.ndarray,
    t_star: int,
    lmin: int,
    lmax: int,
    sign: float,
    prev_ev: int = -1,
    next_ev: int = -1,
) -> tuple[float, float, np.int8, int, int]:
    """Return ``(delta, duration, flag, t_pre_ext, t_post_ext)`` on ``sm``."""
    n = y.size
    pre_lo, pre_hi, post_lo, post_hi = _window_bounds(t_star, n, lmax, prev_ev, next_ev)
    if pre_lo > pre_hi or post_lo > post_hi:
        return np.nan, np.nan, FLAG_EDGE, -1, -1

    n_pre = 0
    n_post = 0
    for t in range(pre_lo, pre_hi + 1):
        if not np.isnan(y[t]):
            n_pre += 1
    for t in range(post_lo, post_hi + 1):
        if not np.isnan(y[t]):
            n_post += 1

    if n_pre < lmin or n_post < lmin:
        return np.nan, np.nan, FLAG_EDGE, -1, -1

    t_pre = -1
    t_post = -1
    if sign < 0.0:
        max_pre = -np.inf
        min_post = np.inf
        for t in range(pre_lo, pre_hi + 1):
            v = sm[t]
            if not np.isnan(v) and v > max_pre:
                max_pre = v
                t_pre = t
        for t in range(post_lo, post_hi + 1):
            v = sm[t]
            if not np.isnan(v) and v < min_post:
                min_post = v
                t_post = t
        if t_pre < 0 or t_post < 0:
            return np.nan, np.nan, FLAG_EDGE, -1, -1
        delta = min_post - max_pre
    else:
        min_pre = np.inf
        max_post = -np.inf
        for t in range(pre_lo, pre_hi + 1):
            v = sm[t]
            if not np.isnan(v) and v < min_pre:
                min_pre = v
                t_pre = t
        for t in range(post_lo, post_hi + 1):
            v = sm[t]
            if not np.isnan(v) and v > max_post:
                max_post = v
                t_post = t
        if t_pre < 0 or t_post < 0:
            return np.nan, np.nan, FLAG_EDGE, -1, -1
        delta = max_post - min_pre

    duration = time_vals[t_post] - time_vals[t_pre]
    return delta, duration, FLAG_OK, t_pre, t_post


@njit(cache=True)
def _apply_magnitudes_at_peaks(
    y_TP: np.ndarray,
    dts_TP: np.ndarray,
    time_vals: np.ndarray,
    peak_idx: np.ndarray,
    lmin: int,
    lmax: int,
    mag_TP: np.ndarray,
    duration_TP: np.ndarray,
    flag_TP: np.ndarray,
):
    """Write ordered-extreme magnitudes at peaks.

    ``peak_idx`` must be sorted by column ``p`` then time ``t`` so neighbouring
    events in the same cell are adjacent and windows can be midpoint-truncated.
    Edge failures leave ``mag_TP`` / ``duration_TP`` as NaN; ``flag_TP`` is
    internal only (not exported).
    """
    last_p = -1
    sm = np.empty(y_TP.shape[0], dtype=np.float64)
    n_peaks = peak_idx.shape[0]
    for k in range(n_peaks):
        t = int(peak_idx[k, 0])
        p = int(peak_idx[k, 1])
        if p != last_p:
            sm = _centred_running_mean(y_TP[:, p], lmin)
            last_p = p
        prev_ev = -1
        next_ev = -1
        if k > 0 and int(peak_idx[k - 1, 1]) == p:
            prev_ev = int(peak_idx[k - 1, 0])
        if k + 1 < n_peaks and int(peak_idx[k + 1, 1]) == p:
            next_ev = int(peak_idx[k + 1, 0])
        sign = 1.0 if dts_TP[t, p] >= 0.0 else -1.0
        d, dur, fl, _, _ = _ordered_extreme_delta(
            y_TP[:, p], sm, time_vals, t, lmin, lmax, sign, prev_ev, next_ev
        )
        mag_TP[t, p] = d
        duration_TP[t, p] = dur
        flag_TP[t, p] = fl


def event_shift_magnitude(
    y: np.ndarray,
    t_star: int,
    lmin: int = 5,
    lmax: int | None = None,
    *,
    sign: float = 1.0,
    neighbours: tuple[int | None, int | None] = (None, None),
    time: np.ndarray | None = None,
) -> dict[str, Any]:
    """Ordered-extreme shift magnitude for one event in one 1-D series.

    Parameters
    ----------
    sign:
        Sign of the event (e.g. ``dts[t_star]``). Negative → min(post)−max(pre);
        non-negative → max(post)−min(pre).
    neighbours:
        ``(prev_event_index, next_event_index)`` in the same cell, or ``None``
        on either side. Windows are truncated at the midpoint between events.
    time:
        Optional time-coordinate values (length ``len(y)``). Duration is
        ``time[t_post] - time[t_pre]``; defaults to integer indices.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if lmax is None:
        lmax = n // 3
    lmin = int(lmin)
    lmax = int(lmax)
    t_star = int(t_star)
    prev_ev = -1 if neighbours[0] is None else int(neighbours[0])
    next_ev = -1 if neighbours[1] is None else int(neighbours[1])
    if time is None:
        time_vals = np.arange(n, dtype=np.float64)
    else:
        time_vals = np.asarray(time, dtype=np.float64)
        if time_vals.size != n:
            raise ValueError("time must have the same length as y")

    sm = _centred_running_mean(y, lmin)
    delta, duration, flag_code, t_pre, t_post = _ordered_extreme_delta(
        y, sm, time_vals, t_star, lmin, lmax, float(sign), prev_ev, next_ev
    )

    pre_lo, pre_hi, post_lo, post_hi = _window_bounds(t_star, n, lmax, prev_ev, next_ev)
    n_pre = 0
    n_post = 0
    for t in range(pre_lo, pre_hi + 1):
        if not np.isnan(y[t]):
            n_pre += 1
    for t in range(post_lo, post_hi + 1):
        if not np.isnan(y[t]):
            n_post += 1

    return {
        "magnitude": float(delta) if flag_code == FLAG_OK else float("nan"),
        "duration": float(duration) if flag_code == FLAG_OK else float("nan"),
        "flag": _FLAG_LABELS.get(int(flag_code), "none"),
        "n_pre": n_pre,
        "n_post": n_post,
        "pre_lo": pre_lo,
        "post_hi": post_hi,
        "t_pre": int(t_pre) if t_pre >= 0 else None,
        "t_post": int(t_post) if t_post >= 0 else None,
    }


def magnitude_var_names(shifts_name: str) -> tuple[str, str]:
    """Return ``(magnitude, duration)`` names for a shifts variable name."""
    return (
        f"{shifts_name}_magnitude",
        f"{shifts_name}_duration",
    )


def compute_shift_magnitudes(
    base: xr.DataArray,
    shifts: xr.DataArray,
    time_dim: str,
    *,
    shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    lmin: int | None = None,
    lmax: int | None = None,
) -> xr.Dataset:
    """Compute dense per-event magnitude fields aligned with ``shifts``.

    Values are written only at local ``|dts|`` peaks above ``shift_threshold``;
    elsewhere ``magnitude`` / ``duration`` are NaN (edge cases included).
    """
    if tuple(base.dims) != tuple(shifts.dims):
        base = base.transpose(*shifts.dims)

    n_time = int(shifts.sizes[time_dim])
    resolved_lmin, resolved_lmax = resolve_lmin_lmax(shifts, n_time)
    if lmin is None:
        lmin = resolved_lmin
    if lmax is None:
        lmax = resolved_lmax
    lmin = int(lmin)
    lmax = int(lmax)

    space_dims = tuple(d for d in shifts.dims if d != time_dim)
    y_t = base.transpose(time_dim, *space_dims)
    dts_t = shifts.transpose(time_dim, *space_dims)

    peak_mask = _compute_dts_peak_sign_mask(
        shifts,
        time_dim,
        shift_threshold=float(shift_threshold),
        shift_selection="local",
    )
    mask_t = peak_mask.transpose(time_dim, *space_dims)

    y_vals = np.ascontiguousarray(np.asarray(y_t.data, dtype=np.float64))
    dts_vals = np.ascontiguousarray(np.asarray(dts_t.data, dtype=np.float64))
    raw_time = np.asarray(y_t[time_dim].values)
    try:
        time_vals = np.ascontiguousarray(raw_time, dtype=np.float64)
    except (TypeError, ValueError):
        if np.issubdtype(raw_time.dtype, np.datetime64):
            time_vals = np.ascontiguousarray(
                raw_time.astype("datetime64[ns]").astype(np.float64)
            )
        else:
            # Non-numeric calendars (e.g. cftime): fall back to index spacing
            time_vals = np.ascontiguousarray(
                np.arange(raw_time.shape[0], dtype=np.float64)
            )
    mask_vals = np.asarray(mask_t.data)
    T = y_vals.shape[0]
    space_shape = y_vals.shape[1:]
    P = int(np.prod(space_shape)) if space_shape else 1
    y_TP = y_vals.reshape(T, P)
    dts_TP = dts_vals.reshape(T, P)
    mask_flat = mask_vals.reshape(T, P)
    # Sparse peak list sorted by column for smooth-cache reuse
    peak_idx = np.argwhere(mask_flat != 0).astype(np.int64)
    if peak_idx.size > 0:
        order = np.lexsort((peak_idx[:, 0], peak_idx[:, 1]))
        peak_idx = peak_idx[order]

    mag_TP = np.full((T, P), np.nan, dtype=np.float64)
    duration_TP = np.full((T, P), np.nan, dtype=np.float64)
    flag_TP = np.full((T, P), FLAG_NONE, dtype=np.int8)

    if peak_idx.size > 0:
        _apply_magnitudes_at_peaks(
            y_TP,
            dts_TP,
            time_vals,
            peak_idx,
            lmin,
            lmax,
            mag_TP,
            duration_TP,
            flag_TP,
        )

    def _to_da(data: np.ndarray, name: str) -> xr.DataArray:
        arr = data.reshape((T, *space_shape))
        da = xr.DataArray(
            arr,
            coords=y_t.coords,
            dims=(time_dim, *space_dims),
            name=name,
        )
        return da.transpose(*shifts.dims)

    shift_name = str(shifts.name) if shifts.name else "shifts"
    base_name = shifts.attrs.get(_attrs.BASE_VARIABLE, str(base.name))
    mag_name, duration_name = magnitude_var_names(shift_name)

    time_units = ""
    if hasattr(y_t[time_dim], "attrs"):
        time_units = str(y_t[time_dim].attrs.get("units", "") or "")

    common_attrs = {
        _attrs.BASE_VARIABLE: base_name,
        _attrs.SHIFTS_VARIABLE: shift_name,
        _attrs.TIME_DIM: time_dim,
        "lmin": lmin,
        "lmax": lmax,
        _attrs.SHIFT_THRESHOLD: float(shift_threshold),
        "magnitude_method": "ordered_extremes",
    }

    mag_da = _to_da(mag_TP, mag_name)
    mag_da.attrs.update(
        {
            **common_attrs,
            _attrs.VARIABLE_TYPE: _attrs.TYPE_SHIFT_MAGNITUDE,
            "long_name": "Ordered-extreme shift magnitude",
            "units": base.attrs.get("units", ""),
        }
    )

    duration_da = _to_da(duration_TP, duration_name)
    duration_da.attrs.update(
        {
            **common_attrs,
            _attrs.VARIABLE_TYPE: _attrs.TYPE_SHIFT_MAGNITUDE_DURATION,
            "long_name": "Duration between ordered-extreme points",
            "units": time_units,
        }
    )

    return xr.Dataset({mag_name: mag_da, duration_name: duration_da})


def _empty_cluster_magnitude() -> dict[str, float | int]:
    return {
        "median_event_magnitude": float("nan"),
        "event_magnitude_q25": float("nan"),
        "event_magnitude_q75": float("nan"),
        "median_event_duration": float("nan"),
        "event_duration_q25": float("nan"),
        "event_duration_q75": float("nan"),
        "n_events": 0,
    }


def _stats_from_event_arrays(
    mag: np.ndarray, dur: np.ndarray | None = None
) -> dict[str, float | int]:
    """Event-level median / IQR for magnitude (and optional duration)."""
    n = int(mag.size)
    if n == 0:
        return _empty_cluster_magnitude()
    out: dict[str, float | int] = {
        "median_event_magnitude": float(np.median(mag)),
        "event_magnitude_q25": float(np.percentile(mag, 25)),
        "event_magnitude_q75": float(np.percentile(mag, 75)),
        "median_event_duration": float("nan"),
        "event_duration_q25": float("nan"),
        "event_duration_q75": float("nan"),
        "n_events": n,
    }
    if dur is not None and dur.size == n:
        finite_dur = dur[np.isfinite(dur)]
        if finite_dur.size:
            out["median_event_duration"] = float(np.median(finite_dur))
            out["event_duration_q25"] = float(np.percentile(finite_dur, 25))
            out["event_duration_q75"] = float(np.percentile(finite_dur, 75))
    return out


def _aggregate_cluster_shift_magnitudes(
    magnitude: xr.DataArray | np.ndarray,
    cluster_mask: xr.DataArray | np.ndarray,
    duration: xr.DataArray | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Event-level median / IQR under a cluster mask.

    Only events with finite ``magnitude`` are included (edge failures are NaN).
    When ``duration`` is given, its stats use the same event set (finite
    durations only for the duration quantiles).
    """
    d = np.asarray(
        magnitude.values if isinstance(magnitude, xr.DataArray) else magnitude,
        dtype=np.float64,
    ).ravel()
    m = np.asarray(
        cluster_mask.values if isinstance(cluster_mask, xr.DataArray) else cluster_mask,
        dtype=bool,
    ).ravel()
    ok = m & np.isfinite(d)
    dur_ok = None
    if duration is not None:
        dur = np.asarray(
            duration.values if isinstance(duration, xr.DataArray) else duration,
            dtype=np.float64,
        ).ravel()
        if dur.shape != d.shape:
            raise ValueError(
                "magnitude and duration must have the same shape "
                f"(got {d.shape}, {dur.shape})"
            )
        dur_ok = dur[ok]
    return _stats_from_event_arrays(d[ok], dur_ok)


def aggregate_shift_magnitudes_by_cluster(
    magnitude: xr.DataArray | np.ndarray,
    cluster_labels: xr.DataArray | np.ndarray,
    cluster_ids: list[int] | np.ndarray,
    duration: xr.DataArray | np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Vectorised per-cluster event-level median / IQR / count.

    Only finite ``magnitude`` values are included. Optional ``duration`` is
    aggregated over the same events. Returns one dict per id in ``cluster_ids``.
    """
    d = np.asarray(
        magnitude.values if isinstance(magnitude, xr.DataArray) else magnitude,
        dtype=np.float64,
    ).ravel()
    lab = np.asarray(
        cluster_labels.values
        if isinstance(cluster_labels, xr.DataArray)
        else cluster_labels
    ).ravel()
    if d.shape != lab.shape:
        raise ValueError(
            "magnitude and cluster_labels must have the same shape "
            f"(got {d.shape}, {lab.shape})"
        )
    dur = None
    if duration is not None:
        dur = np.asarray(
            duration.values if isinstance(duration, xr.DataArray) else duration,
            dtype=np.float64,
        ).ravel()
        if dur.shape != d.shape:
            raise ValueError(
                "magnitude and duration must have the same shape "
                f"(got {d.shape}, {dur.shape})"
            )

    # Labels are often float with NaN for noise; filter before int cast
    lab_f = np.asarray(lab, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(lab_f)
    d_ok = d[ok]
    lab_ok = lab_f[ok].astype(np.int64)
    dur_ok = dur[ok] if dur is not None else None

    by_id_mag: dict[int, np.ndarray] = {}
    by_id_dur: dict[int, np.ndarray] = {}
    if d_ok.size:
        order = np.argsort(lab_ok, kind="mergesort")
        lab_sorted = lab_ok[order]
        d_sorted = d_ok[order]
        dur_sorted = dur_ok[order] if dur_ok is not None else None
        cuts = np.flatnonzero(lab_sorted[1:] != lab_sorted[:-1]) + 1
        starts = np.concatenate((np.array([0], dtype=np.int64), cuts))
        ends = np.concatenate((cuts, np.array([lab_sorted.size], dtype=np.int64)))
        for s, e in zip(starts, ends):
            cid = int(lab_sorted[s])
            by_id_mag[cid] = d_sorted[s:e]
            if dur_sorted is not None:
                by_id_dur[cid] = dur_sorted[s:e]

    rows: list[dict[str, float | int]] = []
    for cid in cluster_ids:
        cid_i = int(cid)
        rows.append(
            _stats_from_event_arrays(
                by_id_mag.get(cid_i, np.empty(0)),
                by_id_dur.get(cid_i) if by_id_dur else None,
            )
        )
    return rows
