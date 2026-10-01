"""
Utility functions and constants for TOAD.

This module provides helper functions and constants used throughout TOAD, including:
- Dimension handling and coordinate detection
- Attribute management and naming conventions
- Deprecation utilities
- Grid validation
"""

import functools
import re
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cftime
import numpy as np
import xarray as xr

__all__ = [
    "DEFAULT_SHIFT_THRESHOLD",
    "get_space_dims",
    "get_latlon_info",
    "detect_latlon_names",
    "is_regular_grid",
    "deprecated",
    "get_unique_variable_name",
    "_attrs",
    "convert_time_to_seconds",
    "convert_numeric_to_original_time",
]


@dataclass(frozen=True)
class _Attrs:
    """Constants for xarray attribute names and values used throughout TOAD."""

    # Attribute names
    VARIABLE_TYPE: str = "variable_type"
    BASE_VARIABLE: str = "base_variable"
    SHIFTS_VARIABLE: str = "shifts_variable"
    CLUSTER_IDS: str = "cluster_ids"
    SHIFT_THRESHOLD: str = "shift_threshold"
    SHIFT_SELECTION: str = "shift_selection"
    SHIFT_DIRECTION: str = "shift_direction"
    MIN_EVENT_MAGNITUDE: str = "min_event_magnitude"
    MIN_EVENT_MAGNITUDE_L: str = "min_event_magnitude_L"
    MIN_EVENT_MAGNITUDE_G: str = "min_event_magnitude_G"
    SCALER: str = "scaler"
    TIME_WEIGHT: str = "time_weight"
    N_DATA_POINTS: str = "n_data_points"
    METHOD_NAME: str = "method_name"
    RUNTIME_PREPROCESSING: str = "runtime_preprocessing"
    RUNTIME_CLUSTERING: str = "runtime_clustering"
    RUNTIME_SHIFTS_DETECTION: str = "runtime_shifts_detection"
    RUNTIME_TOTAL: str = "runtime_total"
    TOAD_VERSION: str = "toad_version"
    TIME_DIM: str = "time_dim"

    # optimization related attributes
    OPTIMIZATION: str = "optimization"
    OPT_OBJECTIVE: str = "opt_objective"
    OPT_BEST_SCORE: str = "opt_best_score"
    OPT_DIRECTION: str = "opt_direction"
    OPT_PARAMS: str = "opt_params"
    OPT_BEST_PARAMS: str = "opt_best_params"
    OPT_N_TRIALS: str = "opt_n_trials"

    # Attribute values
    TYPE_SHIFT: str = "shift"
    TYPE_CLUSTER: str = "cluster"
    TYPE_CLUSTER_SIGN: str = "cluster_sign"
    TYPE_CONSENSUS_CLUSTER: str = "consensus_cluster"
    TYPE_CONSENSUS_RATE: str = "consensus_rate"
    TYPE_CONSENSUS_SIGN: str = "consensus_sign"

    # Consensus metadata
    CONSENSUS_LABELS_VAR: str = "consensus_labels_var"
    CLUSTER_LABELS_VAR: str = "cluster_labels_var"
    CONSENSUS_RATE_SUFFIX: str = "_rate"
    CONSENSUS_SIGN_SUFFIX: str = "_sign"
    N_MODELS: str = "n_models"
    CLUSTER_VARS: str = "cluster_vars"


_attrs = _Attrs()

# Default parameter values used across clustering, stats, and plotting
DEFAULT_SHIFT_THRESHOLD: float = 0.5


def get_space_dims(xr_da: Union[xr.DataArray, xr.Dataset], tdim: str) -> list[str]:
    """Get spatial dimensions from an xarray DataArray or Dataset.

    Args:
        xr_da: Input DataArray or Dataset to get dimensions from
        tdim: Name of temporal dimension. All other dims are considered spatial.

    Returns:
        List of spatial dimension names as strings. If standard spatial dims
        (x/y, y/x) or (lon/lat, lat/lon) are found, returns only those.

    Raises:
        ValueError: If provided temporal dim is not in the dimensions
    """

    # get dims from first data variable (not from the dataset, as these are more prone to change order..)
    dims = xr_da[list(xr_da.data_vars)[0]].dims

    if tdim not in dims:
        raise ValueError(f"Provided temporal dim '{tdim}' is not in the dimensions!")

    # Check for standard spatial dim pairs
    for pair in [("x", "y"), ("lon", "lat")]:
        if all(dim in dims for dim in pair):
            return sorted(
                list(pair), key=lambda x: list(dims).index(x)
            )  # keep original order from xr_da

    # Fallback: use all non-temporal dims
    sdims = list(dims)
    sdims.remove(tdim)
    return [str(dim) for dim in sdims if "bnds" not in str(dim)]


def _reorder_space_dims(space_dims: list[str]) -> list[str]:
    """Reorder space dimensions to ensure lat comes before lon if both present.

    Args:
        space_dims: List of spatial dimension names

    Returns:
        Reordered list with lat before lon if both present, otherwise original list
    """
    if all(dim in space_dims for dim in ["lat", "lon"]):
        return [dim for dim in ["lat", "lon"] if dim in space_dims] + [
            dim for dim in space_dims if dim not in ["lat", "lon"]
        ]
    return space_dims


def get_latlon_info(
    data: xr.Dataset, space_dims: list[str]
) -> Tuple[Optional[str], Optional[str], bool, bool]:
    """Detect lat/lon names and grid type from dataset and space dimensions.

    Returns:
        (lat_name, lon_name, has_latlon, is_latlon_dims)
        - has_latlon: True if both lat and lon coordinates were found
        - is_latlon_dims: True if space_dims are exactly [lat_name, lon_name]
    """
    lat_name, lon_name = detect_latlon_names(data)
    has_latlon = lat_name is not None and lon_name is not None
    is_latlon_dims = has_latlon and (space_dims == [lat_name, lon_name])
    return lat_name, lon_name, has_latlon, is_latlon_dims


def detect_latlon_names(data: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
    """Detect latitude and longitude coordinate names in a dataset.

    Searches for common latitude/longitude names in coordinates first,
    then falls back to data variables.

    Args:
        data: xarray Dataset to search

    Returns:
        Tuple of (lat_name, lon_name). Either can be None if not found.
    """
    lat_candidates = ["lat", "latitude", "nav_lat"]
    lon_candidates = ["lon", "longitude", "nav_lon"]

    # Try coordinates first
    lat_name = next((n for n in lat_candidates if n in data.coords), None)
    if lat_name is None:
        # Fall back to data variables
        lat_name = next((n for n in lat_candidates if n in data.variables), None)

    lon_name = next((n for n in lon_candidates if n in data.coords), None)
    if lon_name is None:
        # Fall back to data variables
        lon_name = next((n for n in lon_candidates if n in data.variables), None)

    return lat_name, lon_name


def is_regular_grid(data: xr.Dataset) -> bool:
    """Check if a dataset has a regular grid.

    Args:
        data: xarray Dataset to check

    Returns:
        True if the grid is regular (1D lat/lon), False otherwise
    """
    # Get lat/lon coordinate names
    lat_name, lon_name = detect_latlon_names(data)

    # Check if lat/lon coordinates are present
    if lat_name is None or lon_name is None:
        # No lat/lon coordinates → likely Cartesian/projected → regular
        return True

    # Check if both lat/lon coordinates are 1D (regular grid)
    try:
        lat = data[
            lat_name
        ]  # Use data[] instead of coords[] to handle both coords and data vars
        lon = data[lon_name]

        # Regular grid: both lat and lon must be 1D
        return lat.ndim == 1 and lon.ndim == 1

    except KeyError:
        # lat_name or lon_name not found in dataset
        return False


def deprecated(message=None):
    """Mark functions as deprecated with @deprecated decorator"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_message = (
                message
                if message
                else f"{func.__name__} is deprecated and will be removed in a future version."
            )
            warnings.warn(warn_message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _all_functions(obj) -> list[str]:
    return [x for x in dir(obj) if not x.startswith("__") and callable(getattr(obj, x))]


def _is_equal_to(x, value):
    """Check if x equals value, whether x is a scalar or sequence."""
    if np.isscalar(x):
        return x == value
    return np.array_equal(x, [value])


def _contains_value(x, value):
    """Check if x contains value, whether x is a scalar or sequence."""
    if np.isscalar(x):
        return x == value
    return value in x


def get_unique_variable_name(desired_name: str, existing_vars, logger=None) -> str:
    """Generate a unique variable name by appending sequential numbers if needed.

    Args:
        desired_name: The desired variable name
        existing_vars: Container with existing variable names (Dataset, dict, list, set, etc.)
        logger: Optional logger for info messages

    Returns:
        Unique variable name (either desired_name or desired_name_N)

    Examples:
        >>> get_unique_variable_name("tas_cluster", ["tas_cluster"])
        "tas_cluster_1"
        >>> get_unique_variable_name("tas_cluster", ["tas_cluster", "tas_cluster_1"])
        "tas_cluster_2"
        >>> get_unique_variable_name("tas_cluster_5", ["tas_cluster_5", "tas_cluster_6"])
        "tas_cluster_7"
    """
    if desired_name not in existing_vars:
        return desired_name

    # Check if the name already has a number at the end
    match = re.search(r"_(\d+)$", desired_name)
    if match:
        # Extract the base name and current number
        name_without_num = desired_name[: match.start()]
        current_num = int(match.group(1))
        next_num = current_num + 1
    else:
        # No number at the end, start with _1
        name_without_num = desired_name
        next_num = 1

    # Find the next available number
    while f"{name_without_num}_{next_num}" in existing_vars:
        next_num += 1

    new_name = f"{name_without_num}_{next_num}"

    if logger:
        logger.debug(
            f"Variable {desired_name} already exists. Using {new_name} instead."
        )

    return new_name


def convert_time_to_seconds(time_array: xr.DataArray) -> np.ndarray:
    """Convert time dimension values to numeric seconds since the first time point.

    Args:
        time_array: xarray DataArray containing the time dimension to convert

    Returns:
        numpy.ndarray: Array of numeric time values in seconds relative to first time point

    Raises:
        ValueError: If time dimension values are not integers, floats, or datetime objects

    Examples:
        >>> times = xr.DataArray(pd.date_range('2000-01-01', periods=5))
        >>> convert_time_to_seconds(times)
        array([0., 86400., 172800., 259200., 345600.])
    """
    # Check if time dimension values are numeric (integers or floats)
    if not (
        np.issubdtype(time_array.dtype, np.integer)
        or np.issubdtype(time_array.dtype, np.floating)
    ):
        # Handle datetime values (both numpy datetime64 and cftime)
        try:
            if np.issubdtype(time_array.dtype, np.datetime64):
                # NumPy datetime64
                numeric_times = (
                    (time_array - time_array[0])
                    .astype("timedelta64[s]")
                    .astype(float)
                    .values
                )
            else:
                # cftime or other datetime objects
                time_values = time_array.values
                numeric_times = np.array(
                    [(t - time_values[0]).total_seconds() for t in time_values]
                )
        except (TypeError, AttributeError):
            raise ValueError(
                "time dimension must consist of integers, floats, or datetime objects."
            )
    else:
        # Time values are already numeric, use as-is
        numeric_times = time_array.values  # Add .values here too

    return numeric_times


def convert_numeric_to_original_time(
    numeric_result: float, numeric_times: np.ndarray, original_time_values: xr.DataArray
) -> Union[float, cftime.datetime, np.datetime64]:
    """Convert a numeric time result back to original format for user-facing results.

    Args:
        numeric_result: The numeric time value to convert
        numeric_times: Array of numeric time values used for calculations
        original_time_values: Array of original time values for conversion

    Returns:
        The interpolated original time value (can be between existing values)
    """
    # Check if original time values are already numeric (int/float)
    if np.issubdtype(original_time_values.dtype, np.integer) or np.issubdtype(
        original_time_values.dtype, np.floating
    ):
        # If original times are numeric, just return the numeric result
        return (
            float(numeric_result)
            if hasattr(numeric_result, "values")
            else numeric_result
        )
    else:
        # Extract scalar value from DataArray if needed
        if hasattr(numeric_result, "values"):
            seconds = float(numeric_result.values)  # type: ignore
        else:
            seconds = float(numeric_result)

        first_time = original_time_values.values[0]

        # Check if it's numpy datetime64
        if np.issubdtype(original_time_values.dtype, np.datetime64):
            # Use numpy timedelta64 for numpy datetime64
            new_time = first_time + np.timedelta64(int(seconds), "s")
            return new_time
        else:
            # Assume cftime or other datetime objects that work with datetime.timedelta
            from datetime import timedelta

            new_time = first_time + timedelta(seconds=seconds)
            return new_time
