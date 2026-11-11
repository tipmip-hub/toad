"""ASDETECT method for shifts detection.

Contains the ASDETECT algorithm with associated helper functions.

Created: January 22, ? (Sina)
Refactored: Nov, 2024 (Jakob)
"""

import logging
from typing import Literal, Optional

import numpy as np
from numba import njit
from numpy.linalg import lstsq
from numpy.typing import NDArray

from .base import ShiftsMethod


class ASDETECT(ShiftsMethod):
    """Detect abrupt shifts in a time series using gradient-based analysis related to [Boulton+Lenton2019]_.

    The algorithm works by:
        1. Dividing the time series into segments of size `l`.
        2. Performing linear regression within each segment to calculate gradients.
        3. Identifying significant gradients exceeding ±3 Median Absolute Deviations (MAD) from the median gradient.
        4. Updating a detection array by adding +1 for significant positive gradients and -1 for significant negative gradients in each segment.
        5. Iterating over multiple window sizes (`l`), updating the detection array at each step.
        6. Normalizing the detection array according to the selected segmentation mode.

    Two segmentation modes are available:

    **"two_sided" mode** (default, recommended):
        - Uses forward and backward segmentation to reduce bias and improve coverage.
        - Segments overlap, providing better temporal coverage of the time series.
        - Normalizes by dividing by a counter array that tracks how many segments cover each time point.
        - Applies edge correction by downweighting signals at boundaries where fewer segments overlap, reducing unreliable detections at the start and end of the time series.
        - Generally produces smoother and more reliable results, especially near the edges.

    **"original" mode** (following [Boulton+Lenton2019]_):
        - Uses centered, non-overlapping segments that are evenly distributed within the time series.
        - Normalizes by dividing by the total number of window sizes used (lmax - lmin + 1).
        - Simpler approach that matches the original algorithm implementation.

    Note: The detection is computed in float64 for any input dtype (e.g. float32
    netCDF data); the scores are returned in the input dtype.

    Note: NaN values are supported when they occur only as leading and/or trailing
    padding (e.g. GWL bins the model never reached). The finite core is analysed and
    scores are written back with zero outside that span. Internal gaps (NaN between
    finite values) still yield an all-zero detection series.

    Args:
        lmin: The minimum segment length (in time steps) used for detection. Controls the
            shortest resolvable timescale and must be at least 3. Defaults to 5.
        lmax: Optional maximum segment length (in time steps) used for detection. To resolve
            multiple shifts within the same grid cell, this should be chosen smaller than the
            minimum temporal separation between shifts of interest. If not specified, it
            defaults to one third of the length of the time dimension.
        timescale: Optional tuple specifying the minimum and maximum time window sizes for
            shift detection in the same physical units as the input time axis (e.g. years,
            days). When provided, this is converted internally to integer segment lengths
            (lmin, lmax) and overrides the explicit lmin/lmax settings. Either bound may be
            set to None to fall back to the defaults (5 time steps for the minimum,
            one third of the series length for the maximum).

            **Important:** When the input dataset uses cftime format, the timescale parameter
            must be specified in seconds, because cftime values are internally converted to
            seconds since the first time point. For example, to detect shifts at timescales of
            1-2 years with cftime data, use ``timescale=(365.25*24*3600, 2*365.25*24*3600)``
            (seconds). For numeric time data (not cftime), use the same units as your time axis.

            **Note:** The timescale conversion requires regular temporal spacing. If your time
            series has irregular spacing, a ValueError will be raised. In this case, use lmin/lmax
            parameters directly instead of timescale, or resample your data to regular spacing.
        segmentation: Segmentation method to use. "two_sided" (recommended) applies forward
            and backward segmentation with counter-based normalisation and edge correction,
            reducing positional bias at roughly twice the computational cost of the original
            algorithm. "original" reproduces the centred segmentation of the original
            ASDETECT implementation for backward compatibility. Defaults to "two_sided".
        ignore_nan_warnings: (Optional) If True, timeseries containing NaN values will be
            ignored, i.e. a detection time series of all zeros will be returned. If False,
            an error will be raised.
    """

    # minimum allowed segment length
    LMIN_MIN = 5

    def __init__(
        self,
        lmin: int = LMIN_MIN,
        lmax: Optional[int] = None,
        timescale: Optional[tuple[Optional[float], Optional[float]]] = None,
        segmentation: Literal["two_sided", "original"] | str = "two_sided",
        ignore_nan_warnings: bool = False,
    ):
        self.lmin = lmin
        self.lmax = lmax
        self.timescale = timescale
        self.segmentation: Literal["two_sided", "original"] | str = segmentation
        self.ignore_nan_warnings = ignore_nan_warnings

        if timescale is not None and not (
            isinstance(timescale, tuple)
            and len(timescale) == 2
            and (timescale[0] is not None or timescale[1] is not None)
            and (timescale[0] is None or isinstance(timescale[0], (float, int)))
            and (timescale[1] is None or isinstance(timescale[1], (float, int)))
            and (
                timescale[0] is None
                or timescale[1] is None
                or timescale[1] > timescale[0]
            )
        ):
            raise ValueError(
                f"Timescale must be a tuple of two numbers with the second number larger than the first, e.g. (20, 30). One of the numbers can be None. If the first is None, e.g. (None, 30), {self.LMIN_MIN} is used as default. If the second is None, e.g., (20, None), 1/3 the length of the time series is used as default."
            )

    def _get_segment_lengths(
        self,
        times_1d: NDArray[np.float64],
    ) -> tuple[int, int]:
        """Get the final lmin and lmax values, handling timescale conversion if needed.

        Args:
            times_1d: 1D array of times

        Returns:
            tuple: (lmin, lmax) as integers
        """

        lmax_max = len(times_1d) // 3
        lmin, lmax = self.lmin, self.lmax if self.lmax is not None else lmax_max

        dt = np.diff(times_1d)[0]
        if self.timescale is not None:
            if self.timescale[0] is not None:
                lmin = int(self.timescale[0] / dt)
            if self.timescale[1] is not None:
                lmax = int(self.timescale[1] / dt)

            logging.getLogger("TOAD").debug(
                f"for dt={dt:.2f} -> (lmin={lmin}, lmax={lmax})"
            )

            if lmin < self.LMIN_MIN and self.timescale[0] is not None:
                raise ValueError(
                    f"The temporal resolution is too low to detect shifts at timescales of {self.timescale[0]} (units of time). "
                    f"We recommend using a minimum timescale of {(self.LMIN_MIN * dt)} (units of time)."
                )

        if lmax > lmax_max:
            if not getattr(self, "_lmax_cap_warned", False):
                if self.timescale is not None and self.timescale[1] is not None:
                    logging.getLogger("TOAD").warning(
                        f"The time series is not long enough for detecting shifts at timescales of {self.timescale[1]} (units of time). "
                        f"A maximum upper bound of {(lmax_max * dt)} (units of time) has been imposed. "
                        f"This corresponds to 1/3 the length of the time series."
                    )
                else:
                    logging.getLogger("TOAD").warning(
                        f"lmax cannot be larger than 1/3 the length of the time series. Setting lmax to {lmax_max}."
                    )
                self._lmax_cap_warned = True
            lmax = lmax_max

        return lmin, lmax

    def fit_predict(
        self,
        values_1d: NDArray[np.float64],
        times_1d: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the detection time series for each grid cell in the 3D data array.

        Args:
            values_1d: 1D array of values
            times_1d: 1D array of times

        Returns:
            A 1D array of the same length as `values_1d`, where each value represents the abrupt shift score for a grid cell at a specific time. The score ranges from -1 to 1:
                - `1` indicates that all tested segment lengths detected a significant positive gradient (i.e. exceeding 3 MAD of the median gradient),
                - `-1` indicates that all tested segment lengths detected a significant negative gradient.
                - Values between -1 and 1 indicate the proportion of segment lengths detecting a significant gradient at that time point.
        """

        # Resolve locally. Writing the result back onto self would make an
        # automatic lmax stick and be reused on a series of a different length.
        lmin, lmax = self._get_segment_lengths(times_1d)

        return construct_detection_ts(
            values_1d=values_1d,
            times_1d=times_1d,
            lmin=lmin,
            lmax=lmax,
            segmentation=self.segmentation,
            ignore_nan_warnings=self.ignore_nan_warnings,
        )


# Helper functions =============================================================


# 1D time series analysis of abrupt shifts =====================================
@njit
def _leading_trailing_finite_bounds(
    values_1d: NDArray[np.float64],
) -> tuple[int, int]:
    """Inclusive index bounds of the finite core, or (-1, -1) if unusable.

    Returns (-1, -1) when the series is all non-finite or contains an internal gap.
    """
    n = len(values_1d)
    first = -1
    for i in range(n):
        if np.isfinite(values_1d[i]):
            first = i
            break
    if first < 0:
        return -1, -1

    last = -1
    for i in range(n - 1, -1, -1):
        if np.isfinite(values_1d[i]):
            last = i
            break

    for i in range(first, last + 1):
        if not np.isfinite(values_1d[i]):
            return -1, -1
    return first, last


def leading_trailing_nan_processable(values_1d) -> bool:
    """True if ``values_1d`` is usable (finite core, no internal NaN gaps)."""
    arr = np.asarray(values_1d, dtype=np.float64)
    first, _ = _leading_trailing_finite_bounds(arr)
    return first >= 0


@njit
def _construct_detection_ts_finite(
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    lmin: int,
    lmax: int,
    segmentation: Literal["two_sided", "original"] | str,
) -> NDArray[np.float64]:
    """Run ASDETECT on a finite 1D series (no NaNs)."""
    n_tot = len(values_1d)
    detection_ts = np.zeros_like(values_1d)

    if lmax < lmin:
        if lmin == 5 and lmax == int(n_tot / 3):
            raise ValueError(
                f"Time series is too short for ASDETECT: with n={n_tot}, default lmin={lmin}, and default lmax=int(n/3)={lmax}. lmin must be smaller than lmax; your time series must be at least {lmin * 3} steps long. "
                f"Either increase the length of your input time series, or decrease lmin."
            )
        raise ValueError("lmin must be smaller than lmax")

    if segmentation == "original":
        for length in range(lmin, lmax + 1):
            n_seg = int(n_tot / length)
            idx0 = (n_tot - n_seg * length) // 2
            seg_idces = (idx0 + np.arange(n_seg + 1, dtype=np.int32) * length).astype(
                np.int32
            )
            gradients = compute_gradients_from_segments(
                values_1d, times_1d, seg_idces[:-1], seg_idces[1:] - seg_idces[:-1]
            )

            grad_median = median(gradients)
            deviations = gradients - grad_median
            mask = np.abs(deviations) > 3 * mad_from_median(gradients, grad_median)

            signs = np.sign(deviations) * mask.astype(np.float64)
            for i in range(len(mask)):
                if mask[i]:
                    detection_ts[seg_idces[i] : seg_idces[i + 1]] += signs[i]

        detection_ts /= lmax - lmin + 1
    elif segmentation == "two_sided":
        counter = np.zeros_like(values_1d, dtype=np.int32)
        # Work arrays sized for the shortest segment length (the most segments),
        # allocated once per series instead of once per segment length.
        max_segs = 2 * ((n_tot - lmin) // lmin + 1)
        gradients = np.empty(max_segs, dtype=np.float64)
        ix0_arr = np.empty(max_segs, dtype=np.int32)
        work = np.empty(max_segs, dtype=np.float64)
        for length in range(lmin, lmax + 1):
            detection_ts, counter = update_detection_ts_two_sided(
                detection_ts,
                counter,
                values_1d,
                times_1d,
                length,
                gradients,
                ix0_arr,
                work,
            )

        detection_ts /= counter
        detection_ts *= counter / max(counter)
    else:
        raise ValueError(
            f"Segmentation method '{segmentation}' not recognized. Choose 'original' or 'two_sided'."
        )

    return detection_ts


@njit
def construct_detection_ts(
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    lmin: int = 5,
    lmax: Optional[int] = None,
    segmentation: Literal["two_sided", "original"] | str = "two_sided",
    ignore_nan_warnings: bool = False,
) -> NDArray[np.float64]:
    """Construct a detection time series (asdetect algorithm).

    Following [Boulton+Lenton2019]_, the time series (ts) is divided into
    segments of length l, for each of which the gradient is computed. Segments
    with gradients > 3 MAD of the gradients distribution are marked. Averaging
    over many segmentation choices (i.e. values of l) results in a detection
    time series that indicates the points of largest relative gradients.

    Two segmentation modes are available:
    - "two_sided": Uses forward and backward segmentation to reduce bias, with counter-based
      normalization. Applies edge correction to downweight signals at boundaries.
    - "original": Uses the original centered segmentation algorithm from [Boulton+Lenton2019]_.
      Segments are non-overlapping and centered within the time series.

    >> Args:
        values_1d:
            Time series, shape (n,)
        times_1d:
            Times, shape (n,), same length as values_1d
        lmin:
            Smallest segment length, default = 5
        lmax:
            Largest segment length, default = n/3
        segmentation:
            Segmentation method to use. Options are "original" (classic) and "two_sided" (removes bias + smoother with edge correction). Defaults to "two_sided".
        ignore_nan_warnings:
            Deprecated; retained for API compatibility. Leading/trailing NaNs are
            trimmed automatically. Internal NaN gaps still return all zeros.

    >> Returns:
        - Abrupt shift score time series, shape (n,), in the dtype of values_1d

    The detection runs in float64 whatever the input dtype. In float32, the
    segment sums lose enough precision in the least-squares slope (which
    subtracts nearly equal products) to flip segments at the 3 MAD threshold,
    most for values or times far from zero (e.g. temperatures in K, calendar
    years).
    """

    detection_ts = np.zeros_like(values_1d)
    first, last = _leading_trailing_finite_bounds(values_1d)
    if first < 0:
        return detection_ts

    core_values = values_1d[first : last + 1].astype(np.float64)
    core_times = times_1d[first : last + 1]
    n_core = len(core_values)

    lmax_eff = int(n_core / 3) if lmax is None else lmax
    lmax_max = n_core // 3
    if lmax_eff > lmax_max:
        lmax_eff = lmax_max
    if lmax_eff < lmin:
        return detection_ts

    core_detection = _construct_detection_ts_finite(
        core_values,
        core_times,
        lmin,
        lmax_eff,
        segmentation,
    )
    detection_ts[first : last + 1] = core_detection
    return detection_ts


@njit
def compute_gradients_from_segments(
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    seg_starts: NDArray[np.int32],
    seg_lengths: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compute gradients for segments defined by start indices and lengths.

    Loops over segments and computes the gradient using a linear fit (1st degree polynomial).

    Args:
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values
        seg_starts: 1D array of starting indices for each segment
        seg_lengths: 1D array of lengths for each segment (or 1-element array if all same length)

    Returns:
        1D array of gradients for each segment
    """
    n_segs = len(seg_starts)
    gradients = np.empty(n_segs)
    is_single = len(seg_lengths) == 1

    for i in range(n_segs):
        i1 = seg_starts[i]
        i2 = i1 + (seg_lengths[0] if is_single else seg_lengths[i])
        gradients[i] = linear_fit(times_1d[i1:i2], values_1d[i1:i2])

    return gradients


@njit
def update_detection_ts_two_sided(
    detection_ts: NDArray[np.float64],
    counter: NDArray[np.int32],
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    segment_length: int,
    gradients: NDArray[np.float64],
    ix0_arr: NDArray[np.int32],
    work: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Update the detection time series for a given segment length (two_sided mode).

    Args:
        detection_ts: Current detection time series
        counter: Current normalization counter
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values
        segment_length: Length of the segments to divide the time series into
        gradients: Work array for the segment gradients, at least as long as
            the number of forward plus backward segments
        ix0_arr: Work array for the segment start indices, same length
        work: Scratch array for the median and MAD, same length
    Returns:
        Updated detection time series and normalization counter
    """
    n_grads = get_gradients_double(
        values_1d, times_1d, segment_length, gradients, ix0_arr
    )

    work[:n_grads] = gradients[:n_grads]
    grad_median = _median_inplace(work[:n_grads])
    for i in range(n_grads):
        work[i] = np.abs(gradients[i] - grad_median)
    threshold = 3 * _median_inplace(work[:n_grads])

    for i in range(n_grads):
        deviation = gradients[i] - grad_median
        i1, i2 = ix0_arr[i], ix0_arr[i] + segment_length
        counter[i1:i2] += 1
        if np.abs(deviation) > threshold:
            detection_ts[i1:i2] += np.sign(deviation)

    return detection_ts, counter


@njit
def get_gradients_double(
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    segment_length: int,
    gradients: NDArray[np.float64],
    ix0_arr: NDArray[np.int32],
) -> int:
    """
    Compute the gradients by segmenting the time series in both forward and
    backward direction, to reduce bias.

    Writes forward then backward segments into the leading entries of
    ``gradients`` and ``ix0_arr``.

    Args:
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values
        segment_length: Length of the segments to divide the time series into.
        gradients: Output array for the gradient of each segment
        ix0_arr: Output array for the starting index of each segment
    Returns:
        Number of segments written (forward plus backward)
    """
    n_tot = len(values_1d)
    n_segs = (n_tot - segment_length) // segment_length + 1
    offset = n_tot - segment_length * n_segs

    for k in range(2 * n_segs):
        if k < n_segs:
            i1 = k * segment_length
        else:
            i1 = (k - n_segs) * segment_length + offset
        ix0_arr[k] = i1
        gradients[k] = linear_fit(
            times_1d[i1 : i1 + segment_length], values_1d[i1 : i1 + segment_length]
        )

    return 2 * n_segs


@njit
def linear_fit(x: NDArray[np.float64], y: NDArray[np.float64]) -> np.float64:
    """Fast linear fit (gradient only) using simple least squares formula.

    Optimized for deg=1 case. Returns only the gradient (slope) coefficient.
    Uses efficient formula: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)

    The sums are accumulated in one loop without temporary arrays. Each starts
    from a zero of the dtype ``np.sum`` would use and adds in the same order, so
    the result matches ``np.sum``-based code exactly.

    Args:
        x: 1D array of x-coordinates (independent variable)
        y: 1D array of y-coordinates (dependent variable)

    Returns:
        Gradient (slope) coefficient
    """
    n = len(x)
    if n == 0:
        return np.float64(0.0)
    # np.multiply on scalars takes the dtype numba gives the array product
    # (e.g. int64 * float32 -> float32), whereas ``*`` on scalars would promote
    # to float64. ``a - a`` is a zero of a's dtype (inputs are finite here).
    sum_x = x[0] - x[0]
    sum_y = y[0] - y[0]
    sum_xy = np.multiply(x[0], y[0]) - np.multiply(x[0], y[0])
    sum_x2 = np.multiply(x[0], x[0]) - np.multiply(x[0], x[0])
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += np.multiply(x[i], y[i])
        sum_x2 += np.multiply(x[i], x[i])
    denominator = n * sum_x2 - sum_x * sum_x
    return np.float64(
        (n * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0.0
    )


@njit
def polyfit(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    deg: int,
) -> NDArray[np.float64]:
    """Least squares polynomial fit.

    This function is a stripped-down version of numpy.polyfit, to make it
    compatible with numba. It computes the least squares polynomial fit for
    the given data points (x, y) of degree deg. The function returns the
    coefficients of the polynomial.

    Args:
        x: 1D array of x-coordinates (independent variable)
        y: 1D or 2D array of y-coordinates (dependent variable)
        deg: Degree of the polynomial to fit (0 <= deg <= 1 for linear fit, 2 for quadratic, etc.)

    Returns:
        Array of polynomial coefficients

    Raises:
        ValueError: If deg < 0
        TypeError: If x is not 1D, x is empty, y is not 1D/2D, or x and y have different lengths
    """

    order = int(deg) + 1
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0

    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    lhs = np.vander(x, order)
    scale = np.sqrt((lhs * lhs).sum(axis=0))
    lhs /= scale
    coefficients = lstsq(lhs, y, len(x) * np.finfo(x.dtype).eps)[0]  # type: ignore
    return (coefficients.T / scale).T  # type: ignore


@njit
def mad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numba-compatible median-absolute-deviation function.

    Computes the median absolute deviation of the input array x.

    Args:
        x: 1D array of values (e.g., gradients)

    Returns:
        The median absolute deviation of the input array x
    """
    return mad_from_median(x, median(x))


@njit
def mad_from_median(
    x: NDArray[np.float64], x_median: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute MAD given the median (avoids recomputing median).

    Args:
        x: 1D array of values
        x_median: Pre-computed median of x

    Returns:
        The median absolute deviation of x
    """
    # Compute absolute deviations explicitly for clarity
    abs_deviations = np.abs(x - x_median)
    return median(abs_deviations)


@njit
def median(x: NDArray[np.float64]) -> float:
    """Numba-compatible median function.

    Computes the median of the input array x (x is not modified).

    Args:
        x: 1D array of values (e.g., gradients)

    Returns:
        The median of the input array x
    """
    return _median_inplace(x.copy())


@njit
def _median_inplace(x: NDArray[np.float64]) -> float:
    """Median of x, reordering x in place.

    Quickselect for the upper middle element, then (for even length) the largest
    element below it. These are the elements a full sort would pick, so the
    result equals the sort-based median exactly, in O(n) instead of O(n log n)
    and without allocating.

    Args:
        x: 1D array of values, reordered in place

    Returns:
        The median of x
    """
    n = len(x)
    k = n // 2
    lo, hi = 0, n - 1
    while lo < hi:
        pivot = x[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while x[i] < pivot:
                i += 1
            while x[j] > pivot:
                j -= 1
            if i <= j:
                x[i], x[j] = x[j], x[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            break
    upper = x[k]
    if n % 2 == 1:
        return upper
    lower = x[0]
    for q in range(1, k):
        if x[q] > lower:
            lower = x[q]
    return 0.5 * (lower + upper)
