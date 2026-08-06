import logging
from typing import Callable, Optional, Union, cast

import numpy as np
import xarray as xr

logger = logging.getLogger("TOAD")

# Common CMIP / ensemble dimension names (kept when present on the dataset)
_ENSEMBLE_DIM_NAMES = frozenset(
    {"member_id", "realization", "ensemble", "number", "init"}
)
# CMIP6 DCPP catalog often carries this as length 1; multi-init must be subsetted by the user
_DCPP_INIT_YEAR = "dcpp_init_year"
# Vertical axes — TOAD uses 2 horizontal spatial dims only; multi-level data must be sliced first
_VERTICAL_DIM_NAMES = frozenset(
    {"lev", "plev", "depth", "olevel", "zlev", "pressure", "height"}
)


def clean_for_toad(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """Lightweight CMIP-style cleanup for typical 2D+time grids.

    Drops bounds variables, squeezes length-1 nuisance dimensions (including a
    singleton ``dcpp_init_year``), and checks that remaining **horizontal**
    structure is exactly **two** spatial dimensions. TOAD does not support 3D
    spatial fields (e.g. multiple vertical levels); subset those datasets before
    calling this.

    Raises:
        ValueError: If ``time_dim`` is missing, ``dcpp_init_year`` has length > 1,
            any vertical dimension has length > 1, or the inferred horizontal
            spatial dimension count is not 2. In those cases preprocess the
            dataset yourself.
    """
    if time_dim not in ds.dims:
        raise ValueError(
            f"time_dim {time_dim!r} not found in dataset dimensions {list(ds.dims)}"
        )

    if _DCPP_INIT_YEAR in ds.sizes and ds.sizes[_DCPP_INIT_YEAR] > 1:
        raise ValueError(
            f"{_DCPP_INIT_YEAR!r} has length {ds.sizes[_DCPP_INIT_YEAR]}; "
            "TOAD only supports a single initialization (or none). "
            "Subset DCPP-style data to one init year or preprocess manually."
        )

    dim_names = {str(d) for d in ds.sizes}
    ensemble_present = _ENSEMBLE_DIM_NAMES & dim_names

    keep_dims: set[str] = {time_dim} | ensemble_present

    # --- Drop bounds/bnds variables first so auxiliary dims become easier to drop ---
    bnds_vars = [
        v
        for v in list(ds.data_vars) + list(ds.coords)
        if "bnds" in str(v).lower() or "bounds" in str(v).lower()
    ]
    if bnds_vars:
        logger.info("Dropping bounds vars: %s", bnds_vars)
        ds = ds.drop_vars(bnds_vars, errors="ignore")

    # Static 2D grid geometry (e.g. CESM POP TLAT on y/x) is not a TOAD field.
    # Drop before inferring spatial dims so orphan y/x axes do not inflate the count.
    static_data_vars = [v for v in ds.data_vars if time_dim not in ds[v].dims]
    if static_data_vars:
        logger.info(
            "Dropping static data vars (no %r dim): %s", time_dim, static_data_vars
        )
        ds = ds.drop_vars(static_data_vars)

    data_var_dims = {d for v in ds.data_vars for d in ds[v].dims}
    for dim in list(ds.sizes):
        if dim in keep_dims or dim in data_var_dims:
            continue
        if ds.sizes[dim] == 1:
            logger.info("Squeezing size-1 dim: %s", dim)
            ds = ds.isel({dim: 0}, drop=True)
        else:
            logger.info(
                "Dropping unused dim (no data vars): %s (size %s)", dim, ds.sizes[dim]
            )
            ds = ds.drop_dims(cast(str, dim))

    known_nonspatial = (
        {time_dim} | _ENSEMBLE_DIM_NAMES | {_DCPP_INIT_YEAR} | _VERTICAL_DIM_NAMES
    )

    spatial_dims: set[str] = set()
    for dim in ds.sizes:
        if dim in keep_dims:
            continue
        if dim in known_nonspatial:
            if dim in _VERTICAL_DIM_NAMES and ds.sizes[dim] > 1:
                raise ValueError(
                    f"Vertical dimension {dim!r} has size {ds.sizes[dim]}; "
                    "TOAD expects 2 horizontal spatial dimensions only. "
                    "Slice to a single level (or surface) before loading."
                )
            continue
        if ds.sizes[dim] > 1:
            spatial_dims.add(str(dim))

    if len(spatial_dims) != 2:
        raise ValueError(
            "TOAD expects exactly 2 horizontal spatial dimensions; "
            f"found {len(spatial_dims)}: {sorted(spatial_dims)}. "
            "Subset curvilinear grids, zonal means, or other layouts manually."
        )

    logger.info("Detected spatial dims: %s", spatial_dims)
    keep_dims |= spatial_dims

    for dim in list(ds.sizes):
        if dim in keep_dims:
            continue
        if ds.sizes[dim] == 1:
            logger.info("Squeezing size-1 dim: %s", dim)
            ds = ds.isel({dim: 0}, drop=True)
        else:
            logger.info("Dropping auxiliary dim: %s (size %s)", dim, ds.sizes[dim])
            ds = ds.drop_dims(cast(str, dim))

    spatial_coord_names = {"lat", "lon", "latitude", "longitude"}
    orphan_coords = [
        c
        for c in ds.coords
        if c not in ds.sizes and c not in keep_dims and c not in spatial_coord_names
    ]
    if orphan_coords:
        logger.info("Dropping orphan coords: %s", orphan_coords)
        ds = ds.drop_vars(orphan_coords, errors="ignore")

    logger.info("Remaining dims: %s", dict(ds.sizes))
    return ds


class Preprocess:
    """
    Preprocessing methods for TOAD objects.

    Note: Docstrings here are short as this class is under heavy development
    """

    def __init__(self, toad):
        self.td = toad

    def preprocess(self, keep_only=None):
        """
        Preprocess the data. To be implemented.
        """

        raise NotImplementedError("Preprocessing is not yet implemented.")

        # Drop unnecessary variables
        if keep_only:
            self.data = self.data.drop_vars(
                [v for v in self.data.data_vars if v not in keep_only]
            )

        # apply XMIP preprocessing ...

        return self.data

    def preprocess_variable(
        self,
        var: str,
        filter_func: Callable,
        fill_value: Optional[Union[float, int]] = np.nan,
    ) -> None:
        """Apply preprocessing filter to a variable.

        Args:
            var: Variable name
            filter_func: Function that returns True for valid data points
            fill_value: Value to use for filtered out points
        """
        data = self.td.data[var].where(filter_func(self.td.data[var]), fill_value)
        self.td.data[var] = data

    def dimension_to_variables(
        self,
        var: str,
        dim: str,
        drop_original: bool = True,
        add_dim_to_name: bool = True,
    ):
        """
        Convert a dimension in a dataset to separate variables.

        Args:
            var: Name of variable to process
            dim: Name of dimension to convert to variables
            drop_original: Whether to remove the original variable after conversion. Defaults to True.

        Example:
            # Convert realization dimension to variables for 'thk'
            td.preprocess.dimension_to_variables(var='thk', dim='realization')
        """
        ds = self.td.data
        # Check if dimension exists
        if dim not in ds.dims and dim not in ds.coords:
            raise ValueError(
                f"Dimension '{dim}' not found in dataset. Available dimensions: {list(ds.dims.keys())}"
            )

        # Check if variable exists if specified
        if var not in ds.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. Available variables: {list(ds.data_vars.keys())}"
            )

        # Create new variables directly in the existing dataset
        new_var_names = []
        for val in ds[dim].values:
            data = ds[var].sel({dim: val}).drop_vars(dim)
            var_name = f"{var}_{dim}_{val}" if add_dim_to_name else f"{var}_{val}"
            self.td.data[var_name] = data
            self.td.data[var_name].attrs[dim] = val
            new_var_names.append(var_name)

        if drop_original:
            # Drop the original variable and dimension
            self.td.data = self.td.data.drop_vars(var).drop_dims(dim)

        logger.info(f"Converted dimension {dim} to variables: {new_var_names}")
