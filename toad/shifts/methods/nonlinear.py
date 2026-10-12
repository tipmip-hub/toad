"""Nonlinear shift detection via recurrence measures (pyunicorn).

Builds on ASDETECT: sliding-window recurrence measures are computed first, then
ASDETECT is run on that coarse series and interpolated back to the input time axis.

Created: June 2025 (Fritz)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .asdetect import ASDETECT

if TYPE_CHECKING:
    from toad.core import TOAD

RecurrenceMeasureName = Literal["transitivity", "average_path_length"]


def _recurrence_network_cls():
    """Lazy import so toad.shifts loads without pyunicorn installed."""
    try:
        from pyunicorn.timeseries import RecurrenceNetwork
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        raise ImportError(
            "Recurrence / NONLINEAR tools require pyunicorn. "
            "Install with: pip install 'tipmip-toad[recurrence]' "
            "(or pip install pyunicorn)."
        ) from exc
    return RecurrenceNetwork


class NONLINEAR(ASDETECT):
    """Detect abrupt shifts using nonlinear recurrence measures, then ASDETECT.

    Use with the standard API::

        td.compute_shifts(var, NONLINEAR(window_size=40, window_step=5))

    ``fit_predict`` returns only the detection time series (DTS). For a
    grid-wide recurrence diagnostic (plotting), call
    :func:`add_recurrence_diagnostic` after ``compute_shifts``.

    Args:
        window_size: Sliding window length in time-index units. Small windows
            (roughly <~30 samples at ``recurrence_rate=0.05``) often yield NaN
            network measures in pyunicorn; prefer longer windows.
        window_step: Step between windows in time-index units.
        recurrence_rate: Recurrence rate for RQA/RNA measures.
        embedding_dim: Embedding dimension.
        embedding_tau: Embedding delay.
        recurrence_measure: ``"transitivity"`` or ``"average_path_length"``.
        lmin: Minimum ASDETECT segment length. Defaults to 5.
        lmax: Maximum ASDETECT segment length; defaults to one-third of the
            (windowed) series length when unset.
        timescale: Optional ASDETECT timescale tuple (see :class:`ASDETECT`).
        segmentation: ASDETECT segmentation mode.
        ignore_nan_warnings: Forwarded to :class:`ASDETECT`.
    """

    def __init__(
        self,
        window_size: float,
        window_step: float,
        recurrence_rate: float = 0.05,
        embedding_dim: int = 1,
        embedding_tau: int = 0,
        recurrence_measure: RecurrenceMeasureName | str = "transitivity",
        lmin: int = 5,
        lmax: Optional[int] = None,
        timescale: Optional[tuple[Optional[float], Optional[float]]] = None,
        segmentation: Literal["two_sided", "original"] | str = "two_sided",
        ignore_nan_warnings: bool = False,
    ):
        self.window_size = window_size
        self.window_step = window_step
        self.recurrence_rate = recurrence_rate
        self.recurrence_measure = recurrence_measure
        self.embedding_dim = embedding_dim
        self.embedding_tau = embedding_tau

        super().__init__(
            lmin=lmin,
            lmax=lmax,
            timescale=timescale,
            segmentation=segmentation,
            ignore_nan_warnings=ignore_nan_warnings,
        )

    def recurrence_1d(
        self,
        values_1d: np.ndarray,
        times_1d: np.ndarray,
    ) -> np.ndarray:
        """Interpolated recurrence measure on the input time grid (plot diagnostic)."""
        recurrence_measure, window_centers = compute_recurrence_measure_1d(
            values_1d=values_1d,
            times_1d=times_1d,
            window_size=self.window_size,
            window_step=self.window_step,
            recurrence_rate=self.recurrence_rate,
            embedding_dim=self.embedding_dim,
            embedding_tau=self.embedding_tau,
            recurrence_measure=self.recurrence_measure,
        )
        time_idx = np.arange(len(times_1d), dtype=float)
        return np.interp(time_idx, window_centers, recurrence_measure)

    def fit_predict(
        self,
        values_1d: np.ndarray,
        times_1d: np.ndarray,
    ) -> np.ndarray:
        """Detection time series (standard :class:`ShiftsMethod` contract)."""
        recurrence_measure, window_centers = compute_recurrence_measure_1d(
            values_1d=values_1d,
            times_1d=times_1d,
            window_size=self.window_size,
            window_step=self.window_step,
            recurrence_rate=self.recurrence_rate,
            embedding_dim=self.embedding_dim,
            embedding_tau=self.embedding_tau,
            recurrence_measure=self.recurrence_measure,
        )
        shifts = super().fit_predict(
            values_1d=recurrence_measure, times_1d=window_centers
        )
        time_idx = np.arange(len(times_1d), dtype=float)
        return np.interp(time_idx, window_centers, shifts, left=0.0, right=0.0)


def add_recurrence_diagnostic(
    td: "TOAD",
    var: str,
    method: NONLINEAR,
    output_label_suffix: str = "",
    overwrite: bool = True,
) -> str:
    """Store an interpolated recurrence field on ``td.data`` for plotting.

    Does not change :func:`~toad.shifts.compute_shifts` (DTS only). Writes
    ``{var}_{recurrence_measure}{suffix}``. Recomputes the recurrence measure
    (not reused from a preceding ``compute_shifts`` call).

    Args:
        td: TOAD object (``td.data`` is updated in place).
        var: Base variable name already present in ``td.data``.
        method: Configured :class:`NONLINEAR` instance.
        output_label_suffix: Same suffix style as ``compute_shifts``.
        overwrite: Drop an existing diagnostic variable of the same name.

    Returns:
        Name of the written diagnostic variable.
    """
    if not isinstance(method, NONLINEAR):
        raise TypeError("add_recurrence_diagnostic requires a NONLINEAR method")

    name = f"{var}_{method.recurrence_measure}{output_label_suffix}"
    if overwrite and name in td.data:
        td.data = td.data.drop_vars(name)

    rec = xr.apply_ufunc(
        lambda ts: method.recurrence_1d(ts, td.numeric_time_values),
        td.data[var],
        input_core_dims=[[td.time_dim]],
        output_core_dims=[[td.time_dim]],
        vectorize=True,
    ).transpose(*td.data[var].dims)
    td.data[name] = rec
    return name


def compute_recurrence_measure_1d(
    values_1d: NDArray,
    times_1d: NDArray,
    window_size: float,
    window_step: float,
    recurrence_rate: float,
    embedding_dim: int,
    embedding_tau: int,
    recurrence_measure: str,
) -> tuple[NDArray, NDArray]:
    """Sliding-window recurrence measure at window centres (not interpolated).

    Returns:
        measure: 1D array of length ``n_windows``.
        window_centers: Centre indices (or times if ``timings`` were used).
    """
    del times_1d  # reserved for irregular sampling via SlidingWindow.timings
    RecurrenceNetwork = _recurrence_network_cls()

    slider = SlidingWindow(
        timeseries=values_1d,
        window_size=window_size,
        window_step=window_step,
    )

    if recurrence_measure == "transitivity":
        for i in range(slider.n_windows):
            net = RecurrenceNetwork(
                slider.window(i),
                recurrence_rate=recurrence_rate,
                dim=embedding_dim,
                tau=embedding_tau,
                silence_level=3,
            )
            slider.results[0, i] = net.transitivity()
    elif recurrence_measure == "average_path_length":
        for i in range(slider.n_windows):
            net = RecurrenceNetwork(
                slider.window(i),
                recurrence_rate=recurrence_rate,
                dim=embedding_dim,
                tau=embedding_tau,
                silence_level=3,
            )
            slider.results[0, i] = net.average_path_length()
    else:
        raise ValueError(
            f"Calculation of '{recurrence_measure}' is not implemented (yet). "
            "Use 'transitivity' or 'average_path_length'."
        )

    return slider.results[0], slider.window_center


# ---------------------------------------------------------------------------
# SlidingWindow — intended for pyunicorn.timeseries; vendored until upstream.
# ---------------------------------------------------------------------------


class SlidingWindow:
    """Sliding-window helper for timeseries (index- or time-based)."""

    def __init__(
        self,
        timeseries: NDArray,
        window_size: float,
        window_step: float,
        min_t: int = 0,
        max_t: int = -1,
        timings: NDArray | None = None,
        n_recurrence_measures: int = 1,
    ):
        self.timeseries = timeseries
        self.window_size = window_size
        self.window_step = window_step
        if timings is not None:
            self.min_t = np.min(timings)
            self.max_t = np.max(timings)
        else:
            self.min_t = min_t
            self.max_t = len(self.timeseries) if max_t == -1 else max_t
        self.timings = timings
        self.n_recurrence_measures = n_recurrence_measures

        self.win = np.arange(
            self.min_t, self.max_t - self.window_size, self.window_step
        )
        self.n_windows = len(self.win)

        self.window_center = np.zeros(self.n_windows)
        for i in range(self.n_windows):
            y = self.window_indices(i)
            if self.timings is not None:
                self.window_center[i] = (
                    max(self.timings[y]) + min(self.timings[y])
                ) / 2
            else:
                self.window_center[i] = (max(y) + min(y)) / 2

        self.results = np.zeros((self.n_recurrence_measures, self.n_windows))

    def window(self, i_win: int):
        return self.timeseries[self.window_indices(i_win)]

    def window_indices(self, i_win: int):
        t_left = self.win[i_win]
        if self.timings is not None:
            y = np.where(
                (self.timings > t_left) & (self.timings < t_left + self.window_size)
            )[0]
        else:
            y = np.arange(t_left, t_left + self.window_size)
        return y

    def timing(self, i_win: int):
        return self.timings[self.window_indices(i_win)]
