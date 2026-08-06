"""
Shifts module for TOAD.

This module provides functionality for detecting abrupt shifts in climate data. The main function
`compute_shifts` takes a dataset and a method for detecting abrupt shifts and returns the detected
shifts as an xarray object.

Currently implemented methods:
- ASDETECT: Implementation of the [Boulton+Lenton2019]_
"""

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from joblib import Parallel, cpu_count, delayed
from tqdm_joblib import tqdm_joblib

from toad._version import __version__
from toad.utils import _attrs, get_unique_variable_name

from .methods.asdetect import ASDETECT, leading_trailing_nan_processable
from .methods.base import ShiftsMethod
from .methods.edge import EDGE

# Currently implemented methods:
# - ASDETECT: Implementation of the [Boulton+Lenton2019]_ algorithm for detecting abrupt shifts

# Expose all methods here
__all__ = ["ASDETECT", "EDGE", "compute_shifts", "ShiftsMethod"]

logger = logging.getLogger("TOAD")

# to avoid circular import we use TYPE_CHECKING for importing TOAD obj
if TYPE_CHECKING:
    from toad.core import TOAD


def compute_shifts(
    td: "TOAD",
    var: str,
    method: ShiftsMethod,
    output_label_suffix: str = "",
    overwrite: bool = False,
    run_parallel: bool = True,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> xr.Dataset:
    """Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

    Args:
        td: TOAD object
        var: Name of the variable in the dataset to analyze for abrupt shifts
        method: The abrupt shift detection algorithm to use. Choose from predefined method objects
            in toad.shifts or create your own following the base class in toad.shifts.methods.base
        output_label_suffix: A suffix to add to the output label. Defaults to "".
        overwrite: Whether to overwrite existing variable. Defaults to False.
        run_parallel: Whether to run the shift detection in parallel. Defaults to True.
        n_jobs: Number of jobs to run in parallel. Defaults to -1 (use all available cores).
        show_progress: Whether to show a progress bar during parallel processing. Defaults to True.

    Returns:
        Union[xr.Dataset, xr.DataArray]: If merge_input is True, returns an xarray.Dataset containing
            the original data and the detected shifts. If merge_input is False, returns an
            xarray.DataArray containing the detected shifts.

    Raises:
        ValueError: If data is invalid or required parameters are missing
    """

    """
    Overview of the shifts detection process:
    1. Input validation
    2. Apply the detector
    3. Rename the output variable
    4. Save details as attributes
    5. Merge the detected shifts with the original data or return the detected shifts

    The input data is not modified.
    """

    # check if var is in data
    data_array = td.data.get(var)
    if data_array is None:
        raise ValueError(f"variable {var} not found in dataset!")

    # check that data_array is not empty
    if data_array.size == 0:
        raise ValueError(f"data array for variable {var} is empty!")

    # check that data_array is 3-dimensional
    if data_array.ndim != 3:
        raise ValueError("data must be 3-dimensional: time/forcing x space x space")

    use_dask = data_array.chunks is not None
    if use_dask:
        # TODO implement dask backend
        raise RuntimeError(
            "Chunked data is not yet supported. Please call .compute() on the data first."
        )

    # Check if the output_label is already in the data
    output_label = f"{var}_dts{output_label_suffix}"
    if not overwrite:
        output_label = get_unique_variable_name(output_label, td.data, logger)
    elif overwrite and output_label in td.data:
        td.data = td.data.drop_vars(output_label)

    # Apply the detector
    logger.debug(f"Applying detector {method.__class__.__name__} to {var}")

    # Create a mask for non-constant, non-NaN cells
    # Create separate masks for constant values and NaN values
    constant_mask = data_array.min(dim=td.time_dim) == data_array.max(dim=td.time_dim)
    nan_mask = data_array.isnull().all(dim=td.time_dim)

    # Combine masks to get valid cells (not constant and not all NaN)
    valid_mask = ~(constant_mask | nan_mask)
    masked_data_array = data_array.where(valid_mask)

    # Cells with internal NaN gaps are skipped; leading/trailing NaN padding is OK.
    processable_mask = xr.apply_ufunc(
        leading_trailing_nan_processable,
        data_array,
        input_core_dims=[[td.time_dim]],
        vectorize=True,
        dask="allowed",
    )
    internal_nan_mask = (
        data_array.isnull().any(dim=td.time_dim) & valid_mask & ~processable_mask
    )
    n_internal_nan = int(internal_nan_mask.sum().item())
    if n_internal_nan > 0:
        logger.warning(
            f"{n_internal_nan} valid grid cells contain internal NaN gaps in their "
            "time series and will be skipped."
        )

    fully_valid_mask = valid_mask & processable_mask
    masked_data_array = data_array.where(fully_valid_mask)

    # Initialize output array with NaN values
    shifts = xr.full_like(data_array, fill_value=np.nan)

    # Apply detector only to fully valid cells (no all-NaNs, no constants, and no NaNs inside time series)
    if run_parallel and use_dask:
        raise RuntimeError(
            "Dask backend is not yet supported. Please use the Joblib backend instead."
        )
        # # Build chunking dict: time dimension must be -1 (single chunk), spatial dims can be chunked
        # # TODO chunking code should be moved to a helper function
        # chunk_dict: dict[str, int | str] = {td.time_dim: -1}
        # for space_dim in td.space_dims:
        #     chunk_dict[space_dim] = int(masked_data_array.sizes[td.space_dims[0]] / 4)
        # masked_data_array = masked_data_array.chunk(chunk_dict)

        # valid_shifts = (
        #     xr.apply_ufunc(
        #         method.fit_predict,
        #         masked_data_array,
        #         kwargs=dict(times_1d=td.numeric_time_values),
        #         input_core_dims=[[td.time_dim]],
        #         output_core_dims=[[td.time_dim]],
        #         vectorize=True,
        #         dask="parallelized",
        #         output_dtypes=[masked_data_array.dtype],
        #     )
        #     .transpose(*data_array.dims)
        #     .compute()
        # )
    elif run_parallel:
        valid_shifts = _process_with_joblib(
            method,  # only pass method object here
            da=masked_data_array,
            time_dim=td.time_dim,
            space_dims=td.space_dims,
            times_1d=td.numeric_time_values,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
    else:
        valid_shifts = xr.apply_ufunc(
            method.fit_predict,
            masked_data_array,
            kwargs=dict(times_1d=td.numeric_time_values),
            input_core_dims=[[td.time_dim]],
            output_core_dims=[[td.time_dim]],
            vectorize=True,
        ).transpose(*data_array.dims)

    # Update only the valid cells in the output array
    shifts = shifts.where(~fully_valid_mask, valid_shifts)

    # Rename the output variable
    shifts = shifts.rename(output_label)

    # Save method params
    method_params = {
        f"shifts_{param}": str(value)
        for param, value in dict(sorted(vars(method).items())).items()
        if value is not None and not param.startswith("_")
    }

    # Save details as attributes
    shifts.attrs.update(
        {
            _attrs.TIME_DIM: td.time_dim,
            _attrs.METHOD_NAME: method.__class__.__name__,
            _attrs.TOAD_VERSION: __version__,
            _attrs.BASE_VARIABLE: var,
            _attrs.VARIABLE_TYPE: _attrs.TYPE_SHIFT,
            **method_params,
        }
    )

    n_constants = int((constant_mask).sum().values)
    n_nans = int((nan_mask).sum().values)
    n_true = int(valid_mask.sum().values)
    skipped_ratio = float(100 * (1 - n_true / valid_mask.size))
    min_shift = float(shifts.min().values)
    mean_shift = float(shifts.mean().values)
    max_shift = float(shifts.max().values)

    logger.info(
        f"New shifts variable \033[1m{output_label}\033[0m: min/mean/max={min_shift:.3f}/{mean_shift:.3f}/{max_shift:.3f} using {n_true} grid cells. "
        f"Skipped {skipped_ratio:.1f}% grid cells: {int(n_nans)} NaN, {int(n_constants)} constant."
    )

    # 6. Merge the detected shifts with the original data
    return xr.merge([td.data, shifts], combine_attrs="override", compat="override")


def _clone_method(m: ShiftsMethod) -> ShiftsMethod:
    """
    Clone an instance of a ShiftsMethod subclass by copying its non-private, non-None attributes.

    This helper is needed to create a new instance with the same configuration/parameters as an
    existing ShiftsMethod object. It's used when running shift detection on multiple blocks of
    data in parallel, to ensure each block gets an independent, safe-to-modify copy of the method.
    """
    params = {
        k: v for k, v in vars(m).items() if v is not None and not k.startswith("_")
    }
    return m.__class__(**params)


def _process_with_joblib(
    method: ShiftsMethod,
    da: xr.DataArray,
    time_dim: str,
    space_dims: list[str],
    times_1d: np.ndarray,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> xr.DataArray:
    """
    Parallelise per-cell fit_predict(ts) with joblib/loky and return an xarray DataArray
    with the same dims as `da` (including time_dim).

    Assumes: method.fit_predict(ts, times_1d=...) -> 1D array of length len(time).
    """

    if da.chunks is not None:
        raise RuntimeError(
            "Joblib backend requires in-memory data. "
            "Chunked arrays should use the Dask backend."
        )

    # Ensure (cell, time) layout
    x = da.transpose(*space_dims, time_dim).stack(cell=space_dims)
    data = np.asarray(x.transpose("cell", time_dim).data)
    n_cell, n_time = data.shape

    def _run_block(start: int, stop: int):
        # Make local copy of the method to avoid state issues in the worker
        local_method = _clone_method(method)

        out = np.empty((stop - start, n_time), dtype=np.float32)
        for i, ts in enumerate(data[start:stop]):
            res = local_method.fit_predict(ts, times_1d=times_1d)
            if res.shape != (n_time,):
                raise ValueError(
                    "fit_predict must return a 1D array with length equal to the time dimension"
                )
            out[i] = res

        return start, out

    # compute block_cells
    # having more blocks/tasks than cores allows joblib to rebalance tasks if one is stalling. 4 is a good number.
    n_workers = cpu_count() if n_jobs == -1 else n_jobs
    n_workers = max(1, min(n_workers, n_cell))
    n_blocks = 4 * n_workers
    block_cells = int(np.ceil(n_cell / n_blocks))

    # guard rails against tiny and huge memory chunks
    block_cells = max(block_cells, 64)  # avoid tiny blocks
    block_cells = min(block_cells, 8192)  # avoid huge memory chunks

    # Partition cells into blocks to reduce scheduler overhead.
    blocks = [(i, min(i + block_cells, n_cell)) for i in range(0, n_cell, block_cells)]
    n_blocks = len(blocks)

    # Use tqdm_joblib for progress bar if enabled
    ctx = (
        tqdm_joblib(
            desc=f"Shift detection ({n_cell} grid cells in {n_blocks} blocks)",
            total=n_blocks,
        )
        if show_progress
        else nullcontext()
    )

    with ctx:  # conditional context
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
        )(delayed(_run_block)(s, e) for s, e in blocks)

    # Reassemble in correct order
    out = np.empty((n_cell, n_time), dtype=np.float32)
    for start, block_out in results:  # type: ignore
        out[start : start + block_out.shape[0], :] = block_out

    # Wrap back to xarray and restore original dims/order
    out_da = xr.DataArray(
        out,
        dims=("cell", time_dim),
        coords={
            "cell": x.coords["cell"],
            time_dim: x.coords[time_dim],
        },
        name=f"{da.name}_shifts" if da.name else "shifts",
        attrs=da.attrs,
    ).unstack("cell")

    # Match original dim order exactly
    return out_da.transpose(*da.dims)
