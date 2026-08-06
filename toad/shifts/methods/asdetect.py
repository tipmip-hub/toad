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

        # Convert timescale to lmin/lmax if needed (recomputed each call for correct behaviour when dt varies)
        if self.timescale is not None:
            self.lmin, self.lmax = self._get_segment_lengths(times_1d)

        return construct_detection_ts(
            values_1d=values_1d,
            times_1d=times_1d,
            lmin=self.lmin,
            lmax=self.lmax,
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
        for length in range(lmin, lmax + 1):
            detection_ts, counter = update_detection_ts_two_sided(
                detection_ts, counter, values_1d, times_1d, length
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
        - Abrupt shift score time series, shape (n,)
    """

    detection_ts = np.zeros_like(values_1d)
    first, last = _leading_trailing_finite_bounds(values_1d)
    if first < 0:
        return detection_ts

    core_values = values_1d[first : last + 1]
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
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Update the detection time series for a given segment length (two_sided mode).

    Args:
        detection_ts: Current detection time series
        counter: Current normalization counter
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values
        segment_length: Length of the segments to divide the time series into
    Returns:
        Updated detection time series and normalization counter
    """
    gradients, ix0_arr = get_gradients_double(values_1d, times_1d, segment_length)

    grad_median = median(gradients)
    deviations = gradients - grad_median
    mask = np.abs(deviations) > 3 * mad_from_median(gradients, grad_median)

    for i, shift_detected in enumerate(mask):
        i1, i2 = ix0_arr[i], ix0_arr[i] + segment_length
        counter[i1:i2] += 1
        if shift_detected:
            detection_ts[i1:i2] += np.sign(deviations[i])

    return detection_ts, counter


@njit
def get_gradients_double(
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    segment_length: int,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Compute the gradients by segmenting the time series in both forward and
    backward direction, to reduce bias.

    Args:
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values
        segment_length: Length of the segments to divide the time series into.
    Returns:
        gradients: 1D array of gradients for each segment
        ix0_arr: 1D array of starting indices for each segment
    """
    n_tot = len(values_1d)
    n_segs = (n_tot - segment_length) // segment_length + 1
    seg_len_arr = np.array([segment_length], dtype=np.int32)

    seg_starts_fwd = np.arange(n_segs, dtype=np.int32) * segment_length
    gradients_fwd = compute_gradients_from_segments(
        values_1d, times_1d, seg_starts_fwd, seg_len_arr
    )

    seg_starts_bwd = seg_starts_fwd + (n_tot - segment_length * n_segs)
    gradients_bwd = compute_gradients_from_segments(
        values_1d, times_1d, seg_starts_bwd, seg_len_arr
    )

    # Pre-allocate arrays instead of concatenating for better performance
    n_total = len(gradients_fwd) + len(gradients_bwd)
    gradients = np.empty(n_total, dtype=np.float64)
    ix0_arr = np.empty(n_total, dtype=np.int32)

    gradients[: len(gradients_fwd)] = gradients_fwd
    gradients[len(gradients_fwd) :] = gradients_bwd
    ix0_arr[: len(seg_starts_fwd)] = seg_starts_fwd
    ix0_arr[len(seg_starts_fwd) :] = seg_starts_bwd

    return gradients, ix0_arr


@njit
def linear_fit(x: NDArray[np.float64], y: NDArray[np.float64]) -> np.float64:
    """Fast linear fit (gradient only) using simple least squares formula.

    Optimized for deg=1 case. Returns only the gradient (slope) coefficient.
    Uses efficient formula: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)

    Args:
        x: 1D array of x-coordinates (independent variable)
        y: 1D array of y-coordinates (dependent variable)

    Returns:
        Gradient (slope) coefficient
    """
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
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
def median(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numba-compatible median function.

    Computes the median of the input array x.

    Args:
        x: 1D array of values (e.g., gradients)

    Returns:
        The median of the input array x
    """
    # np.sort already returns a copy, so no need for .copy()
    x_sorted = np.sort(x)
    n = len(x_sorted)
    mid = n // 2
    return np.array(
        0.5 * (x_sorted[mid - 1] + x_sorted[mid]) if n % 2 == 0 else x_sorted[mid]
    )
