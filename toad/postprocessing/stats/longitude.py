"""Longitude statistics that are correct for clusters crossing the grid's seam.

A plain arithmetic mean of longitudes is wrong for a set of points that crosses
the seam of the grid's own longitude convention (0/360 on a 0-360 grid, the
dateline on a -180-180 grid): cells at 350 and 10 degrees average to 180
instead of 0. The helpers here first move the longitudes onto the shortest arc
that contains them all, compute the statistic there, and map the result back to
the input's convention.

For points that do not cross the seam (their numeric extent already equals the
shortest arc) the plain statistic of the input values is returned unchanged, so
results for such clusters are identical to an arithmetic mean/median/std.

This module only depends on numpy so that other parts of toad (e.g. the
consensus summary) can import it without import cycles.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "unwrap_longitude",
    "wrap_longitude",
    "mean_longitude",
    "median_longitude",
    "std_longitude",
]

_EPS = 1e-9


def unwrap_longitude(lon) -> tuple[np.ndarray, bool]:
    """Move longitudes onto the shortest arc that contains all of them.

    Args:
        lon: Longitudes in degrees (any shape, NaNs allowed), in any convention.

    Returns:
        ``(values, crossed)``. If the finite values already lie on their shortest
        arc without wrapping (they do not cross the seam), ``values`` is the
        input as a float array and ``crossed`` is False. Otherwise ``values`` are
        the longitudes shifted by multiples of 360 so that they are continuous
        (e.g. 350, 10 -> 350, 370) and ``crossed`` is True. NaNs are kept.
    """
    values = np.asarray(lon, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return values, False
    a = np.mod(finite, 360.0)
    u = np.unique(a)
    if u.size < 2:
        return values, False
    gaps = np.diff(np.append(u, u[0] + 360.0))
    k = int(np.argmax(gaps))
    arc = 360.0 - gaps[k]
    if finite.max() - finite.min() <= arc + _EPS:
        return values, False
    start = u[(k + 1) % u.size]
    out = start + np.mod(np.mod(values, 360.0) - start, 360.0)
    return out, True


def wrap_longitude(x: float, reference) -> float:
    """Map a longitude into the convention of ``reference``.

    The convention is -180-180 if ``reference`` has any negative value,
    otherwise 0-360.
    """
    ref = np.asarray(reference, dtype=float)
    if np.nanmin(ref) < 0:
        return float(np.mod(x + 180.0, 360.0) - 180.0)
    return float(np.mod(x, 360.0))


def mean_longitude(lon) -> float:
    """Mean longitude, correct for points that cross the grid's seam.

    Returns the arithmetic mean of the finite values when they do not cross the
    seam; otherwise the mean on the shortest arc, in the input's convention
    (see :func:`wrap_longitude`). NaN if there are no finite values.
    """
    values = np.asarray(lon, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    unwrapped, crossed = unwrap_longitude(finite)
    if not crossed:
        return float(np.mean(finite))
    return wrap_longitude(float(np.mean(unwrapped)), finite)


def median_longitude(lon) -> float:
    """Median longitude, correct for points that cross the grid's seam."""
    values = np.asarray(lon, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    unwrapped, crossed = unwrap_longitude(finite)
    if not crossed:
        return float(np.median(finite))
    return wrap_longitude(float(np.median(unwrapped)), finite)


def std_longitude(lon) -> float:
    """Standard deviation of longitudes (degrees) on their shortest arc."""
    values = np.asarray(lon, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    unwrapped, _ = unwrap_longitude(finite)
    return float(np.std(unwrapped))
