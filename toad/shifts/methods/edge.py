"""Edge detection method for shifts detection.

Contains the edge detection algorithm with associated helper functions.

Created: May 19, 2026 (Sjoerd)
"""

import logging
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, stats

from .base import ShiftsMethod


class EDGE(ShiftsMethod):
    """Detect abrupt shifts in a time series using edge detection. This method is based on the edge
    detection algorithm by Canny (1986), adapted for abrupt shifts in climate data by Bathiany et al. (2020),
    and adapted by Terpstra et al. (2025). The implementation used here is the simplified 1D
    version of the original 3D algorithm.

    The algorithm works by:
        1. Smoothing the input time series using a Gaussian filter to reduce noise.
        2. Computing the gradient using the Sobel operator.
        3. Applying non-maximum suppression to thin the edges (i.e. keep only local maxima in the gradient)
        4. Applying thresholding to identify significantly high gradients.
        5. Computing abruptness at candidate edges (difference in segment means normalised by pooled std)

    Note: EDGE does not work with NaN values so it will return a detection time series of all zeros if the input time series contains NaN values.
    Note: The algorithm works only on data with enough variability to compute
        abruptness. With minimal variability or noise, this method will not be able to reliably detect
        any abrupt shifts.

    Note: All parameters related to datalength assume they have time step unit.

    Note: Unlike ASDETECT, EDGE abruptness scores are not normalised to [-1, 1]. Significance
        filtering is deferred to :meth:`toad.TOAD.compute_clusters` via ``shift_threshold``.
        For the common 4-sigma abruptness cut used in Bathiany et al. (2020), pass
        ``shift_threshold=4`` when clustering EDGE shifts.

    Args:
        lmin: Minimum segment length for the abruptness measure. If None, defaults
            to 10% of the length of the time series. Recommended to manually set based
            on timescale of interest and data resolution (e.g. for annual data with 10-year
            timescale of abrupt shifts, set lmin=15).
        lmax: Maximum segment length for the abruptness measure. The algorithm aims to use `lmax`,
            but if available data is less it will use between `lmax` and `lmin`. If None, defaults
            to 20% of the length of the time series. Recommended to manually set based on timescale
            of interest and data resolution (e.g. for annual data with 10-year timescale of abrupt
            shifts, set lmax=30).
        lcutoff: Length of data to exclude on both sides of the edge when calculating abruptness.
            If None, defaults to 2% of the length of the time series. Recommended to manually
            set based on timescale of interest and data resolution (e.g. for annual data with
            10-year timescale of abrupt shift, set lcutoff=3).
        alpha: Parameter to control how much the difference in standard deviations between the two
            segments around the edge reduces the pooled standard deviation used to calculate
            abruptness. A higher value increases the abruptness of edges in timeseries where
            variability changes significantly after a shift. Defaults to 0.4.
        smoothing_scale: Standard deviation for Gaussian kernel used to smooth the time series
            before calculating the gradient. Accepts an integer (sigma, in time steps), None to
            disable smoothing, the string "auto" to let the method choose a default (10% of the
            length of the time series). Recommended to manually set based on timescale of interest
            and data resolution (e.g. for annual data with 10-year timescale of abrupt shift,
            set smoothing_scale=10).
        gradient_threshold: Threshold for gradient edge detection. Either a float (absolute threshold)
            or "relative" (uses gradient_threshold_multiplier × max(gradient)). Defaults to "relative".
        gradient_threshold_multiplier: Multiplier for max(gradient) when using relative thresholding.
            Defaults to 0.5.
        ignore_nan_warnings: (Optional) If True, timeseries containing NaN values will be
            ignored, i.e. a detection time series of all zeros will be returned. If False,
            an error will be raised.
    """

    def __init__(
        self,
        lmin: Optional[int] = None,
        lmax: Optional[int] = None,
        lcutoff: Optional[int] = None,
        alpha: float = 0.4,
        smoothing_scale: Optional[Union[int, str]] = "auto",
        gradient_threshold: Optional[Union[float, str]] = "relative",
        gradient_threshold_multiplier: float = 0.5,
        ignore_nan_warnings: bool = False,
    ):
        # Some of these can be None. If None, they will be calculated during call of fit_predict(),
        # based on the data
        self.lmin = lmin
        self.lmax = lmax
        self.lcutoff = lcutoff
        self.alpha = alpha
        self.smoothing_scale = smoothing_scale
        self.gradient_threshold = gradient_threshold
        self.gradient_threshold_multiplier = gradient_threshold_multiplier
        self.ignore_nan_warnings = ignore_nan_warnings

        if lmin is not None and lmax is not None and lmin >= lmax:
            raise ValueError("lmin must be less than lmax.")
        if (lmin is None) != (lmax is None):
            raise ValueError(
                "Both lmin and lmax must be provided together, or both must be None."
            )

        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if (
            gradient_threshold is not None
            and gradient_threshold != "relative"
            and not isinstance(gradient_threshold, (int, float))
        ):
            raise ValueError("gradient_threshold must be a number, 'relative', or None")
        if isinstance(gradient_threshold, (int, float)) and gradient_threshold < 0:
            raise ValueError("gradient_threshold must be non-negative")
        if gradient_threshold_multiplier < 0:
            raise ValueError("gradient_threshold_multiplier must be non-negative")

    def fit_predict(
        self,
        values_1d: NDArray[np.float64],
        times_1d: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the detection time series for each grid cell in the 3D data array.

        Args:
            values_1d: 1D array of values
            times_1d: 1D array of times (accepted for API compatibility; abruptness uses
                time-step indices, as in the reference 1D EDGE implementation)

        Returns:
            A 1D array of the same length as ``values_1d``. Non-zero values are signed abruptness
            scores at candidate edges (0 elsewhere). Apply ``shift_threshold`` in
            :meth:`toad.TOAD.compute_clusters` to filter significant shifts; use
            ``shift_threshold=4`` for the usual 4-sigma abruptness cut.
        """
        n = len(values_1d)

        # Set default values based on the length of the time series
        if self.lmin is None or self.lmax is None:
            self.lmin = int(0.1 * n)
            self.lmax = int(0.2 * n)

            logging.getLogger("TOAD").debug(f"(lmin={self.lmin}, lmax={self.lmax})")
        if self.lcutoff is None:
            self.lcutoff = int(0.02 * len(values_1d))

            logging.getLogger("TOAD").debug(f"lcutoff={self.lcutoff}")

        # If None, no smoothing, if auto calculate default smoothing, if int, use that value.
        smoothing_scale = self.smoothing_scale
        if smoothing_scale == "auto":
            smoothing_scale = int(0.1 * n)
            logging.getLogger("TOAD").debug(f"smoothing_scale={smoothing_scale} (auto)")
        elif smoothing_scale is None:
            logging.getLogger("TOAD").debug("no smoothing applied")
        else:
            try:
                smoothing_scale = int(smoothing_scale)
            except (TypeError, ValueError):
                raise ValueError("smoothing_scale must be an int or 'auto' or None")
            if smoothing_scale < 0:
                raise ValueError("smoothing_scale must be nonnegative")

        # Sanity check on parameter values
        # lmax cannot be larger than (n / 2) - 2*lcutoff - 1
        if self.lmax > (n / 2) - 2 * self.lcutoff - 1:
            raise ValueError(
                f"The timeseries is not long enough to use lmax={self.lmax}. `lmax` cannot be larger than (n/2) - 2*lcutoff - 1"
            )

        if self.lmin <= 0 or self.lmax <= 0 or self.lcutoff <= 0:
            raise ValueError("lmin, lmax, and lcutoff cannot be zero or negative")

        return construct_detection_ts(
            values_1d=values_1d,
            lmin=self.lmin,
            lmax=self.lmax,
            lcutoff=self.lcutoff,
            alpha=self.alpha,
            smoothing_scale=smoothing_scale,
            gradient_threshold=self.gradient_threshold,
            gradient_threshold_multiplier=self.gradient_threshold_multiplier,
            ignore_nan_warnings=self.ignore_nan_warnings,
        )


# Helper functions =============================================================
def sobel_edge_detection(
    values_1d: NDArray[np.float64],
    smoothing_scale: Optional[int] = 10,
) -> NDArray[np.float64]:
    """Perform Sobel edge detection for a one-dimensional time series.

    Args:
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        smoothing_scale: Length of smoothing in Gaussian filter. If None (or 0), no smoothing is
            applied.

    Returns:
        Gradient of the (smoothed) ``values_1d``.
    """

    # Smooth using Gaussian kernel on the specified smoothing_scale
    if smoothing_scale is not None and smoothing_scale > 0:
        values_1d = ndimage.gaussian_filter1d(
            values_1d, smoothing_scale, mode="reflect"
        )

    sobel_x = np.array([-1, 0, 1])
    gradient = ndimage.convolve1d(values_1d, sobel_x, mode="reflect")
    return gradient


def thin_and_threshold_edges(
    gradient: NDArray[np.float64],
    threshold: Optional[Union[float, str]] = "relative",
    threshold_multiplier: float = 0.5,
) -> NDArray[np.float64]:
    """Apply non-maximum suppression and thresholding to a 1D gradient.

    Args:
        gradient: 1D array of gradients
        threshold: Value above which the gradient must lie. Either a float (absolute
            threshold) or ``"relative"`` (uses ``threshold_multiplier`` × max(|gradient|)).
        threshold_multiplier: Multiplier for max(|gradient|) when using relative thresholding.

    Returns:
        1D array of gradient magnitudes at thinned edge locations, 0 elsewhere.
    """
    thinned_edges = np.zeros_like(gradient)

    # Pad gradient with zeros on either end to detect edges at the series boundaries
    padded_gradient = np.pad(gradient, pad_width=1, mode="constant", constant_values=0)

    if threshold == "relative":
        used_threshold = threshold_multiplier * np.max(np.abs(padded_gradient))
    else:
        used_threshold = abs(threshold)

    # Find local maxima in the gradient and apply thresholding
    for i in range(1, len(padded_gradient) - 1):
        if (
            padded_gradient[i] > 0
            and padded_gradient[i] > padded_gradient[i - 1]
            and padded_gradient[i] >= padded_gradient[i + 1]
        ) or (
            padded_gradient[i] < 0
            and padded_gradient[i] < padded_gradient[i - 1]
            and padded_gradient[i] <= padded_gradient[i + 1]
        ):
            if abs(padded_gradient[i]) >= used_threshold:
                thinned_edges[i - 1] = abs(padded_gradient[i])

    return thinned_edges


def compute_abruptness_1D(
    edges_1d: NDArray[np.float64],
    values_1d: NDArray[np.float64],
    lcutoff: int,
    lmax: int,
    lmin: int,
    alpha: float = 0.4,
) -> NDArray[np.float64]:
    """Compute abruptness for 1D data.

    Args:
        edges_1d: 1D array indicating the presence of edges (non-zero values)
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        lcutoff: Timesteps excluded on each side of the edge before fitting segments
        lmax: Maximum segment length used for abruptness
        lmin: Minimum segment length required for a reliable fit
        alpha: Pooled-std adjustment for differing segment variances

    Returns:
        Signed abruptness at edge locations, 0 elsewhere.
    """

    time_inds = np.arange(len(values_1d), dtype=np.float64)
    abruptness = np.zeros(len(values_1d))

    # Calculate abruptness value for each time point in values_1d where an edge exists
    for index in range(len(values_1d)):
        if edges_1d[index] != 0:
            # Check if there is enough length before and after the index
            if (index - lcutoff >= 0) and (index + lcutoff + 1 <= len(values_1d)):
                # First, remove cutoff length
                chunk1_data = values_1d[0 : index - lcutoff]
                chunk2_data = values_1d[index + lcutoff + 1 :]
                chunk1_time = time_inds[0 : index - lcutoff]
                chunk2_time = time_inds[index + lcutoff + 1 :]

                if np.size(chunk1_data) > lmax:
                    chunk1_start = np.size(chunk1_data) - lmax
                else:
                    chunk1_start = 0
                if np.size(chunk2_data) > lmax:
                    chunk2_end = lmax
                else:
                    chunk2_end = np.size(chunk2_data)

                chunk1_data_short = chunk1_data[chunk1_start:]
                chunk2_data_short = chunk2_data[0:chunk2_end]
                chunk1_time_short = chunk1_time[chunk1_start:] - time_inds[index]
                chunk2_time_short = chunk2_time[0:chunk2_end] - time_inds[index]

                N1 = np.size(chunk1_data_short)
                N2 = np.size(chunk2_data_short)

                # Make sure line segments are long enough to compute a reliable regression line
                if not ((N1 < lmin) or (N2 < lmin)):
                    slope_chunk1, intercept_chunk1, r_value, p_value, std_err = (
                        stats.linregress(chunk1_time_short, chunk1_data_short)
                    )

                    slope_chunk2, intercept_chunk2, r_value, p_value, std_err = (
                        stats.linregress(chunk2_time_short, chunk2_data_short)
                    )

                    chunk1_std = np.nanstd(chunk1_data_short, ddof=1)
                    chunk2_std = np.nanstd(chunk2_data_short, ddof=1)
                    pooled_std = np.sqrt(
                        ((N1 - 1) * chunk1_std**2 + (N2 - 1) * chunk2_std**2)
                        / (N1 + N2 - 2)
                    )

                    std_diff = alpha * np.sqrt(
                        np.abs(
                            ((N1 - 1) * chunk1_std**2 - (N2 - 1) * chunk2_std**2)
                            / (N1 + N2 - 2)
                        )
                    )

                    # Make sure pooled_std is not negative
                    if std_diff < pooled_std:
                        pooled_std = pooled_std - std_diff

                    # Prevent division by zero, set abruptness to 0
                    if pooled_std == 0:
                        abruptness[index] = 0
                    else:
                        abruptness[index] = (
                            intercept_chunk2 - intercept_chunk1
                        ) / pooled_std

    return abruptness


# 1D time series analysis of abrupt shifts =====================================
def construct_detection_ts(
    values_1d: NDArray[np.float64],
    lmin: int,
    lmax: int,
    lcutoff: int,
    alpha: float,
    smoothing_scale: Optional[int],
    gradient_threshold: Optional[Union[float, str]] = "relative",
    gradient_threshold_multiplier: float = 0.5,
    ignore_nan_warnings: bool = False,
) -> NDArray[np.float64]:
    """Construct a detection time series (edge detection algorithm).

    Args:
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        lmin: Minimum segment length for abruptness
        lmax: Maximum segment length for abruptness
        lcutoff: Timesteps excluded on each side of the edge
        alpha: Pooled-std adjustment for differing segment variances
        smoothing_scale: Gaussian smoothing sigma in timesteps, or None to disable
        gradient_threshold: Absolute or ``"relative"`` gradient threshold
        gradient_threshold_multiplier: Multiplier for relative gradient thresholding
        ignore_nan_warnings: Passed for API parity with :class:`ASDETECT` (see note below)

    Returns:
        Abrupt shift score time series, shape (n,). Signed abruptness at candidate edges,
        0 elsewhere. Filter with ``shift_threshold`` in :meth:`toad.TOAD.compute_clusters`.

    Note:
        As for :class:`ASDETECT`, any NaN in ``values_1d`` yields an all-zero detection
        series. ``compute_shifts`` skips grid cells that contain NaNs before calling
        ``fit_predict``.
    """

    detection_ts = np.zeros_like(values_1d)

    # return zeros if timeseries contains nan values
    if np.isnan(values_1d).any():
        # User is warned of this in the TOAD.compute_shifts() method
        return detection_ts

    # Calculate the local gradient using the Sobel operator
    gradient = sobel_edge_detection(values_1d, smoothing_scale=smoothing_scale)

    # Apply non-maximum suppression and thresholding to identify significant edges
    edges_1d = thin_and_threshold_edges(
        gradient,
        threshold=gradient_threshold,
        threshold_multiplier=gradient_threshold_multiplier,
    )

    # Return signed abruptness at candidate edges (significance cut deferred to clustering)
    return compute_abruptness_1D(
        edges_1d, values_1d, lcutoff=lcutoff, lmax=lmax, lmin=lmin, alpha=alpha
    )
