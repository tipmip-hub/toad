import logging
import os
import tempfile
from collections.abc import Callable
from typing import List, Literal, Optional, Union

import numpy as np
import optuna
import sklearn.cluster
import xarray as xr
from sklearn.base import ClusterMixin

from toad import (
    clustering,
    plotting,
    postprocessing,
    preprocessing,
    shifts,
)
from toad.clustering.optimizing import (
    default_opt_params,
)
from toad.postprocessing.stats import GeneralStats, SpaceStats, TimeStats
from toad.regridding.base import BaseRegridder
from toad.utils import (
    DEFAULT_SHIFT_THRESHOLD,
    _attrs,
    detect_latlon_names,
    get_space_dims,
)
from toad.utils.repr_html import (
    build_variable_hierarchy,
    load_toad_logo_html,
    render_consensus_variables_html,
    render_hierarchy_html,
)
from toad.utils.shift_selection_utils import _compute_episode_overlap_mask


class _StatsAccessor:
    """Callable accessor for Stats: use td.stats(var) for explicit var, or td.stats.time etc. for inferred var.

    time/space/general delegate to Stats (postprocessing.stats) for inferred var; explicit properties needed for IDE autocomplete.
    """

    def __init__(self, td: "TOAD") -> None:
        self._td = td

    def __call__(self, var: str | None = None) -> postprocessing.Stats:
        var = (
            var
            if var
            else str(self._td.get_clusters(self._td._get_base_var_if_none(None)).name)
        )
        if self._td._is_shift_variable(var):
            var = str(self._td.get_clusters(var).name)
        return postprocessing.Stats(self._td, var)

    @property
    def time(self) -> TimeStats:
        """Access time-related statistics for clusters (uses inferred variable)."""
        return self(None).time

    @property
    def space(self) -> SpaceStats:
        """Access space-related statistics for clusters (uses inferred variable)."""
        return self(None).space

    @property
    def general(self) -> GeneralStats:
        """Access general statistics for clusters (uses inferred variable)."""
        return self(None).general

    def __getattr__(self, name: str) -> object:
        return getattr(self(None), name)


class TOAD:
    """Main object for interacting with TOAD.

    TOAD (Tippping and Other Abrupt events Detector) is a framework for detecting and clustering spatio-temporal patterns in spatio-temporal data.

    Args:
        data: The input data. Can be either an xarray Dataset or a path to a netCDF file.
        time_dim: The name of the time dimension. Defaults to 'time'.
        log_level: The logging level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR',
            'CRITICAL'. Defaults to 'INFO'.
        engine: The engine to use to open the netCDF file. Defaults to 'netcdf4'.
        auto_clean: If True, run :func:`toad.preprocessing.clean_for_toad` after loading
            data (drops bounds, auxiliary dims, orphan coords). Defaults to False.
            Dimension names ``longitude``/``latitude`` are renamed to ``lon``/``lat`` first
            so cleaning and :attr:`space_dims` see the standard names.
        decode_times: How to decode CF time coordinates when ``data`` is a file path.
            ``None`` (default) uses :class:`xarray.coders.CFDatetimeCoder` with
            ``use_cftime=True``, which avoids xarray warnings for dates outside the
            ``numpy.datetime64`` range. Pass ``False`` for undecoded numeric time, or a
            :class:`xarray.coders.CFDatetimeCoder` instance for custom decoding.

    Raises:
        ValueError: If the input file path does not exist or if data dimensions are not 3D.
    """

    data: xr.Dataset
    path: str | None = None

    def __init__(
        self,
        data: xr.Dataset | str,
        time_dim: str = "time",
        log_level: str = "INFO",
        engine: str = "netcdf4",
        auto_clean: bool = False,
        decode_times: bool | xr.coders.CFDatetimeCoder | None = None,
    ):
        # load data from path if string
        if isinstance(data, str):
            if not os.path.exists(data):
                raise ValueError(f"File {data} does not exist.")
            open_decode_times = (
                xr.coders.CFDatetimeCoder(use_cftime=True)
                if decode_times is None
                else decode_times
            )
            self.data = xr.open_dataset(
                data, engine=engine, decode_times=open_decode_times
            )
            self.data.attrs["title"] = os.path.basename(data).split(".")[
                0
            ]  # store path as title for saving toad file later
            self.path = data  # store path
        elif isinstance(data, (xr.DataArray)):
            self.data = data.to_dataset()  # convert to dataset if data is a DataArray
        elif isinstance(data, (xr.Dataset)):
            self.data = data  # Original data

        # Initialize the logger for the TOAD object
        self.logger = logging.getLogger("TOAD")
        self.logger.propagate = False  # Prevent propagation to the root logger :: i.e. prevents dupliate messages
        self.set_log_level(log_level)

        # Rename longitude and latitude to lon and lat
        if "longitude" in self.data.dims:
            self.data = self.data.rename({"longitude": "lon"})
            self.logger.info("Renamed dimension longitude to lon")
        if "latitude" in self.data.dims:
            self.data = self.data.rename({"latitude": "lat"})
            self.logger.info("Renamed dimension latitude to lat")

        # Determine effective time dimension
        if time_dim in self.data.dims:
            self.time_dim = time_dim
        elif "time" in self.data.dims:
            self.time_dim = "time"
        elif "gwl" in self.data.dims:
            self.time_dim = "gwl"
            self.logger.info(
                "No 'time' dimension found, using 'gwl' as the time dimension."
            )
        else:
            raise ValueError(
                f"No suitable time dimension found in data. "
                f"Searched for {time_dim!r}, 'time', and 'gwl'."
            )

        if auto_clean:
            self.data = preprocessing.clean_for_toad(self.data, time_dim=self.time_dim)

        # Check that all variables have the same dimensions
        dims = [self.data[var].dims for var in self.data.data_vars]
        if len(set(dims)) > 1:
            dims_info = "\n".join(
                f"{var}: {self.data[var].dims}" for var in self.data.data_vars
            )
            raise ValueError(
                "All variables must have the same dimensions. Consider dropping variables not needed in TOAD.\n"
                f"Dimensions for each variable:\n{dims_info}"
            )

        lat, lon = detect_latlon_names(self.data)
        if (lat and lat not in self.data.dims) and (lon and lon not in self.data.dims):
            self.logger.info(
                "Found lat/lon coordinates (not dimensions). TOAD will use these for clustering and plotting instead of native dimensions. Drop lat/lon variables to use native coordinates."
            )

    def _is_time_numeric(self) -> bool:
        """Check if the time dimension contains numeric values (int/float) or datetime objects (cftime).

        Args:
            time_array: xarray DataArray containing the time dimension to check. Defaults to the time dimension of the TOAD object.

        Returns:
            bool: True if time dimension is numeric (int/float), False if datetime objects (cftime)
        """
        time_array = self.data[self.time_dim]
        return np.issubdtype(time_array.dtype, np.integer) or np.issubdtype(
            time_array.dtype, np.floating
        )

    @property
    def numeric_time_values(self):
        """Get numeric time values. Defined as property since this might change if user changes the time resolution.

        Returns:
            numpy.ndarray: Array of numeric time values in seconds relative to first time point
        """
        from toad.utils import convert_time_to_seconds

        # Store original time values for plotting
        numeric_time_values = self.data[self.time_dim].values  # convert to numpy array

        # Convert time dimension to numeric values if needed
        if not self._is_time_numeric():
            # Convert datetime objects to seconds since first time point
            numeric_time_values = convert_time_to_seconds(self.data[self.time_dim])

        if not np.issubdtype(
            numeric_time_values.dtype, np.integer
        ) and not np.issubdtype(numeric_time_values.dtype, np.floating):
            raise ValueError(
                "Failed to convert time dimension to numeric values. Convert manually."
            )

        return numeric_time_values

    def numeric_time_values_unit(self) -> str:
        """Get the unit of the numeric time values."""
        if self._is_time_numeric():
            # If original time values are numeric, use their original unit
            return self.data[self.time_dim].attrs.get("units", "")
        else:
            # If we converted cftime to numeric, the unit is "seconds"
            return "seconds"

    def _repr_html_(self):
        """Representation of the TOAD object in html with collapsible hierarchy."""
        hierarchy = build_variable_hierarchy(
            self.base_vars, self.shift_vars, self.cluster_vars, self.data
        )
        variable_table = render_hierarchy_html(hierarchy, self.data)
        consensus_table = render_consensus_variables_html(
            self.data, self.consensus_cluster_vars
        )
        logo_html = load_toad_logo_html()
        ds_repr = self.data._repr_html_()
        return f"""
        <div style='padding: 12px'>
            <h2 style='margin-bottom: 0px; display: flex; align-items: center;'>{logo_html}TOAD Object</h2>
            {variable_table}
            {consensus_table}
            <p style='font-size: 0.9em; margin: 16px 0;'>Hint: to access the xr.dataset call <code>td.data</code></p>
            {ds_repr}
        </div>
        """

    # # ======================================================================
    # #               Module functions
    # # ======================================================================
    @property
    def preprocess(self) -> preprocessing.Preprocess:
        """Access preprocessing methods."""
        return preprocessing.Preprocess(self)

    @property
    def stats(self) -> "_StatsAccessor":
        """Access statistics about clusters and their properties, such as time, space, and general metrics.

        Use as a property when you have a single base variable (var is inferred):
            >>> td.stats.time.start(cluster_id=0)

        Call with a variable name when you have multiple base variables or need to specify:
            >>> td.stats("temperature").time.start(cluster_id=0)
            >>> td.stats(var="temperature").space.mean(cluster_id=0)

        Returns:
            StatsAccessor: callable for explicit var, or use .time/.space/.general for inferred var.
        """
        return _StatsAccessor(self)

    @property
    def aggregate(self) -> postprocessing.Aggregation:
        """Access aggregation methods."""
        return postprocessing.Aggregation(self)

    @property
    def plot(self) -> plotting.Plotter:
        """Access plotting methods.

        Examples:
            >>> td.plot.overview()
            >>> td.plot.map()
            >>> td.plot.timeseries(cluster_ids=range(6))
        """
        return plotting.Plotter(self)

    # # ======================================================================
    # #               SET functions
    # # ======================================================================

    def set_log_level(self, level: str):
        """Sets the logging level for the TOAD logger.

        Sets the logging level and configures handlers for the TOAD logger instance.
        Available levels are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

        Examples:
            Used like this:
                >>> logger.debug("This is a debug message.")
                >>> logger.info("This is an info message.")
                >>> logger.warning("This is a warning message.")
                >>> logger.error("This is an error message.")
                >>> logger.critical("This is a critical message.")

            In sub-modules get logger like this:
                >>> logger = logging.getLogger("TOAD")

        Args:
            level: The logging level to set

        Raises:
            ValueError: If level is not one of the valid logging levels
        """
        level = level.upper()
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(
                "Invalid log level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'"
            )

        self.logger.setLevel(getattr(logging, level))

        # Only add a handler if there are no handlers yet (to avoid duplicate messages)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.debug(f"Logging level set to {level}")

    # # ======================================================================
    # #               COMPUTE functions
    # # ======================================================================

    def compute_shifts(
        self,
        var: str | None = None,
        method: shifts.ShiftsMethod = shifts.ASDETECT(),
        output_label_suffix: str = "",
        overwrite: bool = False,
        run_parallel: bool = True,
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        """Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

        Args:
            var: Name of the base variable to analyze for abrupt shifts. If None and only one base variable exists,
                that variable will be used automatically. If None and multiple base variables exist, raises a ValueError.
                Defaults to None.
            method: The abrupt shift detection algorithm to use. Choose from predefined method objects in `toad.shifts` (e.g., `ASDETECT`),
                or create your own by subclassing `ShiftsMethod` from `toad.shifts`. Defaults to `ASDETECT()`.
            output_label_suffix: A suffix to add to the output label. Defaults to `""`.
            overwrite: Whether to overwrite existing variable. Defaults to `False`.
            run_parallel: Whether to run the shift detection in parallel. Defaults to True.
            n_jobs: Number of jobs to run in parallel. Defaults to -1 (use all available cores).
            show_progress: Whether to show a progress bar during parallel processing. Defaults to True.

        Raises:
            ValueError: If data is invalid or required parameters are missing
        """

        self.data = shifts.compute_shifts(
            td=self,
            var=self._get_base_var_if_none(var),
            method=method,
            output_label_suffix=output_label_suffix,
            overwrite=overwrite,
            run_parallel=run_parallel,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )

    def compute_clusters(
        self,
        var: str | None = None,
        method: ClusterMixin | type = sklearn.cluster.HDBSCAN(
            allow_single_cluster=True
        ),
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
        shift_direction: Literal["both", "positive", "negative"] | str = "both",
        shift_selection: Literal["local", "global", "all"] | str = "local",
        min_event_magnitude: float | None = None,
        min_event_magnitude_L: int = 25,
        min_event_magnitude_G: int = 10,
        time_weight: float = 1,
        regridder: BaseRegridder | None = None,
        disable_regridder: bool = False,
        output_label_suffix: str = "",
        output_label: str | None = None,
        overwrite: bool = False,
        sort_by_size: bool = True,
        export_for_mma: str | None = None,
        mma_grid: Literal["healpix", "native"] = "healpix",
        # optimization related params
        optimize: bool = False,
        optimize_params: dict = default_opt_params,
        optimize_objective: Callable
        | Literal[
            "median_heaviside",
            "mean_heaviside",
            "mean_consistency",
            "mean_spatial_autocorrelation",
            "mean_nonlinearity",
            "combined_spatial_nonlinearity",
        ]
        | str = "combined_spatial_nonlinearity",
        optimize_n_trials: int = 50,
        optimize_direction: str = "maximize",
        optimize_log_level: int = optuna.logging.WARNING,
        optimize_progress_bar: bool = True,
    ):
        """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm.

        Args:
            var: Name of the shifts variable to cluster, or name of the base variable whose shifts
                should be clustered. If None, TOAD will attempt to infer which shifts to use.
                A ValueError is raised if the shifts variable cannot be uniquely determined.
            method: The clustering method to use. Choose methods from sklearn.cluster or create
                your by inheriting from sklearn.base.ClusterMixin. Defaults to HDBSCAN().
            shift_threshold: The minimum magnitude a shift must reach to be included in clustering. Raising this threshold filters out less significant shifts and helps focus clustering on the most meaningful events, while reducing it will include more subtle (and potentially noisier) shifts. Default is 0.5, which effectively excludes most noise when using ASDETECT.
            shift_direction: The sign of the shift. Options are "both", "positive", "negative".
                When "both", positive and negative shifts are clustered separately and merged
                into one output variable so that no cluster contains mixed signs. Defaults to "both".
            shift_selection: How shift values are selected for clustering. All options respect shift_threshold and shift_direction:
                "local": Finds peaks within individual shift episodes. Cluster only local maxima within each contiguous segment where abs(shift) > shift_threshold.
                "global": Finds the overall strongest shift per grid cell. Cluster only the single maximum shift value per grid cell where abs(shift) > shift_threshold.
                "all": Cluster all shift values that meet the threshold and direction criteria. Includes all data points above threshold, not just peaks.
                Defaults to "local".
            min_event_magnitude: Minimum absolute base-variable change per dts episode (physical
                units), measured as a peak-anchored plateau difference with outer half-width
                ``min_event_magnitude_L`` and guard ``min_event_magnitude_G``. Episodes below
                this threshold are excluded from clustering. Defaults to None.
            min_event_magnitude_L: Outer half-width (time steps) for plateau means. Defaults to 25.
            min_event_magnitude_G: Guard half-width (time steps) excluded around the peak.
                Requires ``L > G >= 0``. Defaults to 10.
            time_weight: Controls the relative influence of time in clustering. By default, time values are automatically scaled to match the standard deviation of the spatial coordinates. Increasing time_weight gives more emphasis to the temporal dimension, resulting in clusters that are tighter in time (shorter delays between abrupt events). Decreasing it emphasizes the spatial dimensions, allowing clusters to span a wider range of shift times. Defaults to 1.
            regridder: The regridding method to use from toad.clustering.regridding.
                Defaults to None. If None and coordinates are lat/lon, a HealPixRegridder will
                be created automatically.
            disable_regridder: Whether to disable the regridder. Defaults to False.
            output_label_suffix: A suffix to add to the output label. Defaults to "".
            overwrite: Whether to overwrite existing variable. Defaults to False.
            sort_by_size: Whether to reorder clusters by size. Defaults to True.
            export_for_mma: If set to a file path, exports cluster labels for MMA
                (multi-model aggregation). Format depends on mma_grid.
            mma_grid: Grid format for MMA export: "healpix" (from regridder.df_healpix
                when regridding was used) or "native". Defaults to "healpix".
            optimize: Whether to optimize the clustering parameters. Defaults to False.
            optimize_params: Parameters for the optimization. Defaults to clustering.default_opt_params.
            optimize_objective: The objective function to optimize. Defaults to combined_spatial_nonlinearity. Can be one of:
                - callable: Custom objective function taking (td, output_label) as arguments
                - "median_heaviside": Median heaviside score across clusters
                - "mean_heaviside": Mean heaviside score across clusters
                - "mean_consistency": Mean consistency score across clusters
                - "mean_spatial_autocorrelation": Mean spatial autocorrelation score
                - "mean_nonlinearity": Mean nonlinearity score across clusters
            optimize_n_trials: Number of trials to run for optimization. Defaults to 50.
            optimize_direction: The direction of the optimization. Defaults to "maximize".
            optimize_log_level: The log level for the optimization. Defaults to optuna.logging.WARNING.
            optimize_progress_bar: Whether to show the progress bar for the optimization. Defaults to True.

        Returns:
            None.

        Raises:
            ValueError: If data is invalid or required parameters are missing

        Notes:
            For global datasets, use toad.regridding.HealPixRegridder to ensure equal spacing
            between data points and prevent biased clustering at high latitudes.
        """

        self.data = clustering.compute_clusters(
            td=self,
            var=self._get_base_var_if_none(var),
            method=method,
            shift_threshold=shift_threshold,
            shift_selection=shift_selection,
            min_event_magnitude=min_event_magnitude,
            min_event_magnitude_L=min_event_magnitude_L,
            min_event_magnitude_G=min_event_magnitude_G,
            shift_direction=shift_direction,
            time_weight=time_weight,
            regridder=regridder,
            disable_regridder=disable_regridder,
            output_label_suffix=output_label_suffix,
            output_label=output_label,
            overwrite=overwrite,
            sort_by_size=sort_by_size,
            export_for_mma=export_for_mma,
            mma_grid=mma_grid,
            optimize=optimize,
            optimize_params=optimize_params,
            optimize_objective=optimize_objective,
            optimize_n_trials=optimize_n_trials,
            optimize_direction=optimize_direction,
            optimize_log_level=optimize_log_level,
            optimize_progress_bar=optimize_progress_bar,
        )

    def compute_consensus(
        self,
        cluster_vars: list[str] | None = None,
        *,
        min_consensus: float,
        temporal_tolerance: int,
        spatial_tolerance: int,
        stitch_meridian: bool | Literal["auto"] = "auto",
        show_progress: bool = True,
        output_label_suffix: str = "",
        output_label: str | None = None,
        overwrite: bool = False,
        min_cluster_area: int | None = 2,
    ) -> None:
        """Combine multiple clustering results into one per-voxel member-support consensus.

        This delegates to :meth:`toad.postprocessing.Aggregation.compute_consensus`; see that
        docstring for parameters and algorithm details.

        Args:
            cluster_vars: Input clustering variables to merge. Defaults to all ``td.cluster_vars``.
            min_consensus: Minimum fraction of input clusterings that must support each retained
                native event voxel after tolerance dilation (required).
            temporal_tolerance: Time tolerance used for support dilation and component labelling
                (required).
            spatial_tolerance: Spatial tolerance used for support dilation and component labelling
                (required).
            stitch_meridian: Whether to connect the first and last longitude column on native
                grids. ``\"auto\"`` (default) stitches when the grid spans nearly all
                longitudes; ``False`` disables stitching; ``True`` forces it.
            show_progress: Whether to show a progress bar.
            output_label_suffix: Suffix for the default ``cluster_consensus`` label name.
            output_label: Explicit name for the consensus labels variable.
            overwrite: If True, replace an existing variable with the same name; if False,
                append ``_1``, ``_2``, … when the name is taken.
            min_cluster_area: Minimum spatial footprint (distinct cells ever labelled) for a
                consensus cluster to be kept; smaller clusters become noise. Default ``2`` (see
                :meth:`toad.postprocessing.Aggregation.compute_consensus`). Use ``None`` to
                disable this post-filter.
        """

        self.aggregate.compute_consensus(
            cluster_vars=cluster_vars,
            min_consensus=min_consensus,
            stitch_meridian=stitch_meridian,
            show_progress=show_progress,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            output_label_suffix=output_label_suffix,
            output_label=output_label,
            overwrite=overwrite,
            min_cluster_area=min_cluster_area,
        )

    # # ======================================================================
    # #               netCDF functions
    # # ======================================================================

    def _write_netcdf_atomic(self, save_path: str) -> None:
        """Write via a temp file and atomic replace (safe if target is open for read)."""
        save_dir = os.path.dirname(os.path.abspath(save_path)) or "."
        fd, tmp_path = tempfile.mkstemp(
            suffix=".nc", prefix=".toad-save-", dir=save_dir
        )
        os.close(fd)
        try:
            self.data.to_netcdf(tmp_path)
            os.replace(tmp_path, save_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def save(
        self,
        suffix: Optional[str] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Save the TOAD object to a netCDF file.

        Args:
            suffix: Optional string to append to filename before extension
            path: Optional path to save file to. If not provided, uses self.path
            overwrite: If True, allow saving back to the original path without a
                suffix. All saves use a temporary file and atomic replace so an
                existing destination can stay open for reading (e.g. lazy load).

        Raises:
            ValueError: If neither path nor self.path is set
            ValueError: If using self.path without a suffix (to prevent overwriting)
        """
        if path is None and self.path is None:
            raise ValueError("Path to save TOAD dataset not set. Please provide path.")

        # Prevent overwriting when using self.path
        if path is None and self.path is not None and suffix is None and not overwrite:
            raise ValueError(
                "Please provide either a suffix to append to the original path, "
                "specify a new path, or set overwrite=True."
            )

        # Use user-provided path if specified, otherwise use self.path
        save_path = path if path is not None else self.path

        if save_path is None:
            raise ValueError("Path to save TOAD dataset not set. Please provide path.")

        if save_path == self.path:
            # Get original extension if using self.path (save_path equals self.path here)
            original_ext = save_path.rsplit(".", 1)[1] if "." in save_path else "nc"
        else:
            # For user-provided path without extension, default to .nc
            original_ext = save_path.rsplit(".", 1)[1] if "." in save_path else "nc"

        if suffix:
            # Split path into base and add suffix before extension
            base = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
            save_path = f"{base}_{suffix}.{original_ext}"
        elif "." not in save_path:
            # Add extension if path has none
            save_path = f"{save_path}.{original_ext}"

        # Apply compression =====
        try:
            # First clear any existing encoding
            for var in self.data.variables:
                self.data[var].encoding.clear()

            # Define compression settings
            compression_settings = {
                "zlib": True,
                "complevel": 4,
            }

            # Apply compression to both float and int data variables
            for var in self.data.data_vars:
                if np.issubdtype(self.data[var].dtype, np.number):
                    self.data[var].encoding.update(compression_settings)

                    if np.issubdtype(self.data[var].dtype, np.integer):
                        self.data[var].encoding.update(
                            {"_FillValue": None, "dtype": self.data[var].dtype}
                        )
        except Exception as e:
            self.logger.warning(
                f"Could not apply compression settings: {str(e)}. Proceeding with save without compression."
            )

        self._write_netcdf_atomic(save_path)
        self.logger.info(f"Saved TOAD dataset to {save_path}")

    # # ======================================================================
    # #               GET functions (postprocessing)
    # # ======================================================================

    @property
    def space_dims(self):
        return get_space_dims(self.data, self.time_dim)

    @property
    def base_vars(self) -> list[str]:
        """Gets the list of base variables in the dataset.

        Base variables are those that have not been derived from shift detection or
            clustering. A variable is considered a base variable if either:
                1. It has no 'variable_type' attribute, or
                2. Its 'variable_type' is neither 'shift', 'cluster', nor consensus-derived

        Returns:
            A list of strings containing the base variable names in the dataset.
        """
        _derived = (
            _attrs.TYPE_SHIFT,
            _attrs.TYPE_CLUSTER,
            _attrs.TYPE_CLUSTER_SIGN,
            _attrs.TYPE_CONSENSUS_CLUSTER,
            _attrs.TYPE_CONSENSUS_RATE,
            _attrs.TYPE_CONSENSUS_SIGN,
        )
        return [
            str(x)
            for x in list(self.data.data_vars.keys())
            if self.data[x].attrs.get(_attrs.VARIABLE_TYPE) not in _derived
        ]

    @property
    def shift_vars(self) -> list[str]:
        """Gets the list of shift variables in the dataset.

        Shift variables are those that have been derived from shift detection.
        A variable is considered a shift variable if it has a 'variable_type=_attrs.TYPE_SHIFT'
        attribute.

        Returns:
            A list of strings containing the shift variable names in the dataset.
        """
        return [
            str(x)
            for x in list(self.data.data_vars.keys())
            if self._is_shift_variable(x)
        ]

    @property
    def cluster_vars(self) -> list[str]:
        """Get the list of cluster variables in the dataset.

        Cluster variables are those that have been derived from clustering.
        A variable is considered a cluster variable if it has a 'variable_type="cluster"' attribute.

        Returns:
            list[str]: List of cluster variable names in the dataset
        """
        return [
            str(x)
            for x in list(self.data.data_vars.keys())
            if self._is_cluster_variable(x)
        ]

    @property
    def consensus_cluster_vars(self) -> list[str]:
        """Names of consensus label variables (``variable_type=consensus_cluster``)."""
        return [
            str(x)
            for x in list(self.data.data_vars.keys())
            if self.data[x].attrs.get(_attrs.VARIABLE_TYPE)
            == _attrs.TYPE_CONSENSUS_CLUSTER
        ]

    def _resolve_consensus_var(self, consensus_var: str | None) -> str:
        """Return the consensus labels variable name, or raise if ambiguous.

        If ``consensus_var`` is None, requires exactly one consensus label variable in
        :attr:`consensus_cluster_vars` (similar to how :meth:`get_clusters` resolves a
        single cluster variable for a base ``var``).
        """
        if consensus_var is not None:
            if consensus_var not in self.data:
                raise ValueError(f"Unknown data variable: {consensus_var!r}")
            if not self._is_consensus_cluster_variable(consensus_var):
                raise ValueError(
                    f"{consensus_var!r} is not a consensus label variable "
                    f"(expected attrs['{_attrs.VARIABLE_TYPE}'] == {_attrs.TYPE_CONSENSUS_CLUSTER!r})."
                )
            return consensus_var
        names = self.consensus_cluster_vars
        if len(names) == 0:
            raise ValueError(
                "No consensus variables in the dataset. Run compute_consensus() first or pass "
                "consensus_var=..."
            )
        if len(names) > 1:
            raise ValueError(
                f"Multiple consensus variables {names}. Pass consensus_var explicitly."
            )
        return names[0]

    @staticmethod
    def consensus_rate_var_name(consensus_var: str) -> str:
        """Companion variable name for member-support fractions (``{consensus_var}_rate``)."""
        return f"{consensus_var}{_attrs.CONSENSUS_RATE_SUFFIX}"

    @staticmethod
    def consensus_sign_var_name(consensus_var: str) -> str:
        """Companion sign grid name for a consensus labels variable."""
        return f"{consensus_var}{_attrs.CONSENSUS_SIGN_SUFFIX}"

    def _resolve_consensus_rate_var(self, consensus_var: str) -> str:
        """Return the rate companion variable for a consensus labels variable name."""
        name = self.consensus_rate_var_name(consensus_var)
        if name not in self.data:
            raise ValueError(
                f"No matching consensus rate variable {name!r} for {consensus_var!r}."
            )
        if not self._is_consensus_rate_variable(name):
            raise ValueError(
                f"{name!r} is not a consensus rate variable "
                f"(expected attrs['{_attrs.VARIABLE_TYPE}'] == {_attrs.TYPE_CONSENSUS_RATE!r})."
            )
        return name

    def remove_cluster(self, cluster_id: int, var: str | None = None):
        """Remove a cluster from the dataset.

        Args:
            cluster_id: The cluster ID to remove.
            var: The variable to remove the cluster from. If None, the cluster variable will be inferred automatically.
        """
        cluster_var_name = self.get_clusters(var).name
        if cluster_var_name is None:
            raise ValueError("Resolved cluster variable has no name.")
        cluster_var = str(cluster_var_name)
        original_attrs = dict(self.data[cluster_var].attrs)

        # Remove cluster from cluster variable
        self.data[cluster_var] = self.data[cluster_var].where(
            self.data[cluster_var] != cluster_id
        )

        # Update cluster ids attribute
        existing_ids = np.array(original_attrs.get(_attrs.CLUSTER_IDS, []))
        new_ids = existing_ids[existing_ids != cluster_id]
        updated_attrs = dict(original_attrs)
        updated_attrs[_attrs.CLUSTER_IDS] = new_ids
        self.data[cluster_var].attrs = updated_attrs

    def sort_clusters(
        self,
        var: str | None = None,
        *,
        sort_by: Literal[
            "size",
            "footprint_cumulative_area",
            "median_shift_magnitude",
            "median_shift_time",
            "start_shift_time",
        ] = "size",
        order: list[int] | None = None,
    ):
        """Sort cluster IDs by a given criterion (largest/earliest becomes ID 0).

        Keeps NaN values unchanged and preserves noise label ``-1`` if present.
        Useful after filtering/removing clusters to restore contiguous cluster IDs.

        Args:
            var: Base variable or cluster variable. If None, inferred automatically.
            sort_by: Criterion for sorting when order is None. Options:
                - "size" or "footprint_cumulative_area": by cluster cell count
                  (largest first). Equivalent.
                - "median_shift_magnitude": by median magnitude change (largest first).
                - "median_shift_time": by median time of shifts (earliest first).
                - "start_shift_time": by start time of cluster (earliest first).
            order: Manual order: list of current cluster IDs in the order they should
                become 0, 1, 2, ... When provided, sort_by is ignored. Must be a
                permutation of existing cluster IDs (each ID exactly once).
        """
        var = self._get_base_var_if_none(var)
        cluster_var = self.get_clusters(var).name
        if cluster_var is None:
            raise ValueError("Resolved cluster variable has no name.")
        cluster_var = str(cluster_var)
        original_attrs = dict(self.data[cluster_var].attrs)

        cluster_ids = self.get_cluster_ids(var=str(cluster_var), exclude_noise=True)
        if len(cluster_ids) == 0:
            return

        if order is not None:
            cluster_ids_set = set(cluster_ids)
            if set(order) != cluster_ids_set:
                raise ValueError(
                    "order must contain exactly the cluster IDs (each once). "
                    f"Got {order}, expected permutation of {sorted(cluster_ids_set)}."
                )
            sorted_ids = list(order)
        else:
            if sort_by in ("size", "footprint_cumulative_area"):
                sort_keys = {
                    cid: self.stats(var).space.footprint_cumulative_area(cid)
                    for cid in cluster_ids
                }
                reverse = True  # largest first
            elif sort_by == "median_shift_magnitude":
                base_var = self.data[cluster_var].attrs.get(_attrs.BASE_VARIABLE)
                if base_var is None or base_var not in self.data:
                    raise ValueError(
                        f"sort_by='median_shift_magnitude' requires a base variable. "
                        f"Cluster variable '{cluster_var}' has no BASE_VARIABLE attribute "
                        "or the base variable is missing from the dataset."
                    )
                sort_keys = {
                    cid: abs(
                        self.stats(var).time.value_change(cid, aggregation="median")
                    )
                    for cid in cluster_ids
                }
                # Treat NaN as smallest (sorts last when reverse=True)
                sort_keys = {
                    k: (v if np.isfinite(v) else -np.inf) for k, v in sort_keys.items()
                }
                reverse = True  # largest magnitude first
            elif sort_by == "median_shift_time":
                ts = self.stats(var).time
                sort_keys = {
                    cid: float(np.median(ts._get_cluster_numeric_times(cid)))
                    for cid in cluster_ids
                }
                reverse = False  # earliest first
            elif sort_by == "start_shift_time":
                ts = self.stats(var).time
                sort_keys = {
                    cid: float(np.min(ts._get_cluster_numeric_times(cid)))
                    for cid in cluster_ids
                }
                reverse = False  # earliest first
            else:
                raise ValueError(
                    "sort_by must be one of "
                    "'size', 'footprint_cumulative_area', 'median_shift_magnitude', "
                    "'median_shift_time', 'start_shift_time'"
                )
            sorted_ids = sorted(
                sort_keys.keys(), key=lambda c: sort_keys[c], reverse=reverse
            )
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
        cluster_da = self.data[cluster_var]

        # Build result from the original data to avoid remapping collisions.
        # TODO(dask): When dask support is added, the xr.where loop may need
        # vectorised remapping to avoid a large computation graph; .item() below
        # triggers eager compute for dask-backed arrays.
        sorted_clusters = cluster_da.copy()
        for old_id, new_id in old_to_new.items():
            sorted_clusters = xr.where(
                cluster_da == old_id, float(new_id), sorted_clusters
            )

        self.data[cluster_var] = sorted_clusters

        # Update cluster ids metadata.
        existing_ids = np.array(original_attrs.get(_attrs.CLUSTER_IDS, []))
        new_ids = np.array(list(range(len(old_to_new))), dtype=int)
        has_noise_label = bool((cluster_da == -1).any().item())
        if (-1 in existing_ids) or has_noise_label:
            new_ids = np.concatenate((np.array([-1], dtype=int), new_ids))
        updated_attrs = dict(original_attrs)
        updated_attrs[_attrs.CLUSTER_IDS] = np.unique(new_ids)
        self.data[cluster_var].attrs = updated_attrs

    def drop_clusters(self):
        """
        Remove all cluster variables from the dataset.

        This method drops all variables identified as cluster variables from the
        underlying data object, including companion cluster sign grids.
        """
        from toad.postprocessing.member_support_consensus import (
            sign_var_for_cluster_var,
        )

        to_drop = list(self.cluster_vars)
        for cvar in self.cluster_vars:
            sign_var = sign_var_for_cluster_var(cvar)
            if sign_var in self.data:
                to_drop.append(sign_var)
        self.data = self.data.drop_vars(list(dict.fromkeys(to_drop)))

    def drop_shifts(self):
        """
        Remove all shift variables from the dataset.

        This method drops all variables identified as shift variables from the
        underlying data object.
        """
        self.data = self.data.drop_vars(self.shift_vars)

    def drop_consensus_clusters(self):
        """
        Remove all consensus outputs from the dataset.

        Drops every variable with ``variable_type`` consensus label, rate, or sign.
        """
        to_drop = [
            str(x)
            for x in self.data.data_vars
            if self.data[x].attrs.get(_attrs.VARIABLE_TYPE)
            in (
                _attrs.TYPE_CONSENSUS_CLUSTER,
                _attrs.TYPE_CONSENSUS_RATE,
                _attrs.TYPE_CONSENSUS_SIGN,
            )
        ]
        if to_drop:
            self.data = self.data.drop_vars(to_drop)

    def shift_vars_for_var(self, var: str) -> list[str]:
        """Get the shift variables for a given variable.

        Args:
            var: The variable to get shift variables for. Can be either:
                - A base variable (e.g. 'temperature')
                - A cluster variable (e.g. 'temperature_cluster')
                Cannot be a shift variable.

        Returns:
            List of shift variables associated with the given variable:
                - For base variables: Returns all shift variables that have this as their base variable
                - For cluster variables: Returns the shift variable used to create this cluster

        Raises:
            ValueError: If var is a shift variable, or if no shift variables are found.
        """
        # If variable is a cluster variable, get the shift variable from attrs
        if self._is_cluster_variable(var):
            shift_variable = self.data[var].attrs.get(_attrs.SHIFTS_VARIABLE)
            if shift_variable:
                if shift_variable in self.shift_vars:
                    return [shift_variable]
                else:
                    raise ValueError(
                        f"Shift variable {shift_variable} not found in shift variables."
                    )
            else:
                raise ValueError(f"No shift variable found for cluster variable {var}.")
        # If variable is a shift variable, raise error
        if self._is_shift_variable(var):
            raise ValueError(
                "This is a shift variable. Use this function to get shift variable of a cluster or base variable."
            )
        # Else, must be a base variable, get all shift variables for that base variable
        else:
            return [
                str(x)
                for x in self.shift_vars
                if self.data[x].attrs.get(_attrs.BASE_VARIABLE) == var
            ]

    def cluster_vars_for_var(self, var: str) -> list[str]:
        """Get the cluster variables for a given variable.

        Args:
            var: The variable to get cluster variables for. Can be either:
                - A base variable (e.g. 'temperature')
                - A shift variable (e.g. 'temperature_dts')
                Cannot be a cluster variable.

        Returns:
            List of cluster variables associated with the given variable:
                - For base variables: Returns cluster variables that have this as their base variable
                - For shift variables: Returns cluster variables that were derived from this shift variable

        Raises:
            ValueError: If var is a cluster variable. This function can only get cluster variables
                for base or shift variables.
        """
        if self._is_cluster_variable(var):
            raise ValueError(
                "This is a cluster variable. Use this function to get cluster variables of a base or shift variable."
            )
        elif self._is_shift_variable(var):
            return [
                str(x)
                for x in self.cluster_vars
                if self.data[x].attrs.get(_attrs.SHIFTS_VARIABLE) == var
            ]
        else:
            return [
                str(x)
                for x in self.cluster_vars
                if self.data[x].attrs.get(_attrs.BASE_VARIABLE) == var
            ]

    def get_base_var(self, var: str | None = None) -> Optional[str]:
        """Get the base variable for a given variable.

        Args:
            var: Base variable name, cluster variable name, or shift variable name.
                If None, returns the single base variable when only one exists.
        """
        var = self._get_base_var_if_none(var)
        if var in self.base_vars:
            return var
        else:
            return self.data[var].attrs.get(_attrs.BASE_VARIABLE)

    def get_shifts(
        self, var: str | None = None, label_suffix: str = ""
    ) -> xr.DataArray:
        """Get shifts xr.DataArray for the specified variable.

        Args:
            var: Base variable name (e.g. 'temperature'), cluster variable name, or None to infer
                when only one base variable exists.
            label_suffix: If you added a suffix to the shifts variable, help the function find it.
                Defaults to "".

        Returns:
            The shifts xr.DataArray for the specified variable.

        Raises:
            ValueError: Failed to find valid shifts xr.DataArray for the given var.
        """
        var = self._get_base_var_if_none(var)

        # Check if the variable is a shifts variable
        if self._is_shift_variable(var):
            return self.data[var]

        shift_vars = self.shift_vars_for_var(var)

        # Filter by label_suffix if provided
        if label_suffix:
            shift_vars = [s for s in shift_vars if s.endswith(label_suffix)]

        if len(shift_vars) > 1:
            raise ValueError(
                f"Multiple shift variables exist for {var}: {shift_vars}. Please specify which one to use"
            )
        elif len(shift_vars) == 0:
            raise ValueError(
                f"No shifts variable found for {var}. Please first run compute_shifts()."
            )
        else:
            return self.data[shift_vars[0]]

    def get_clusters(self, var: str | None = None) -> xr.DataArray:
        """Get cluster xr.DataArray for the specified variable.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.

        Returns:
            The clusters xr.DataArray for the specified variable.

        Raises:
            ValueError: Failed to find valid cluster xr.DataArray for the given var. An
                xr.DataArray is only considered a cluster label if it contains _cluster in
                its name.
        """

        var = self._get_base_var_if_none(var)

        # Check if the variable is a cluster variable
        if self._is_cluster_variable(var):
            return self.data[var]

        cluster_vars = self.cluster_vars_for_var(var)
        if len(cluster_vars) > 1:
            raise ValueError(
                f"Multiple cluster variables exist for {var}: {cluster_vars}. Please specify which one to use"
            )
        elif len(cluster_vars) == 0:
            raise ValueError(
                f"No cluster variables found for {var}. Please first run compute_clusters()."
            )
        else:
            return self.data[cluster_vars[0]]

    def get_cluster_counts(self, var: str, exclude_noise: bool = True) -> dict:
        """Returns sorted dictionary with number of cells in both space and time for each cluster.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            exclude_noise: Whether to exclude noise points (cluster ID -1). Defaults to True.

        Returns:
            Dictionary mapping cluster IDs to their total cell counts, sorted by count in
            descending order.
        """
        counts = {}
        for cluster_id in self.get_cluster_ids(var, exclude_noise):
            count = self.get_cluster_mask(var, cluster_id).sum()
            counts[int(cluster_id)] = int(count)

        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def cluster_summary(
        self,
        var: str | None = None,
        cluster_ids: int | list[int] | range | None = None,
        *,
        extended: bool = False,
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
        exclude_noise: bool = True,
    ):
        """Per-cluster overview table for a clustering result.

        Convenience alias for :meth:`Stats.cluster_summary`. See that method for
        column definitions and examples.

        Args:
            var: Base variable, cluster variable, or None to infer when unambiguous.
            cluster_ids: Subset of clusters to include. Defaults to all non-noise clusters.
            extended: Include IQR bounds, shift amplitudes, pooled transition times,
                and clustering metadata.
            shift_threshold: Minimum shift magnitude for pooled transition-time columns.
            exclude_noise: Exclude noise (cluster ID -1) when ``cluster_ids`` is None.
        """
        return self.stats(var).cluster_summary(
            cluster_ids=cluster_ids,
            extended=extended,
            shift_threshold=shift_threshold,
            exclude_noise=exclude_noise,
        )

    def get_cluster_ids(
        self, var: str | None = None, exclude_noise: bool = True
    ) -> np.ndarray:
        """Return list of cluster ids sorted by total number of cells in each cluster.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            exclude_noise: Whether to exclude noise points (cluster ID -1). Defaults to True.

        Returns:
            List of cluster ids.
        """
        raw = self.get_clusters(var).attrs.get(_attrs.CLUSTER_IDS, [])
        cluster_ids = np.atleast_1d(np.asarray(raw, dtype=np.int64))
        if exclude_noise:
            cluster_ids = cluster_ids[cluster_ids != -1]
        return cluster_ids

    def get_cluster_mask(
        self,
        var: str | None = None,
        cluster_id: int | List[int] | range | None = None,
        numeric_times: bool = False,
    ) -> xr.DataArray:
        """Returns a 3D boolean mask (time x space x space) indicating which points belong to the specified cluster(s).

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            cluster_id: Cluster id(s) to apply the mask for.
            numeric_times: If True, returns mask with numeric time coordinates instead of original time format.
                Defaults to False.

        Returns:
            Mask for the cluster label.
        """

        clusters = self.get_clusters(var)

        all_cluster_ids = np.atleast_1d(
            np.asarray(clusters.attrs.get(_attrs.CLUSTER_IDS, []), dtype=np.int64)
        )
        if cluster_id is None:
            cluster_id = all_cluster_ids
        else:
            existing = set(all_cluster_ids.tolist())
            valid_cluster_ids = [
                int(id)
                for id in np.atleast_1d(cluster_id).flatten()
                if int(id) in existing
            ]
            if len(valid_cluster_ids) == 0:
                raise ValueError(
                    f"None of the specified clusters {cluster_id} for var {var} exists. Did you mean any of these: {all_cluster_ids}?"
                )
            cluster_id = valid_cluster_ids

        mask = clusters.isin(cluster_id)

        if numeric_times:
            # Replace time coordinates with numeric values
            mask = mask.assign_coords({self.time_dim: self.numeric_time_values})

        return mask

    def get_cluster_mask_spatial(
        self,
        var: str | None = None,
        cluster_id: int | list[int] | range | None = None,
    ) -> xr.DataArray:
        """Returns a 2D boolean mask indicating which grid cells belonged to the specified cluster at any point in time.

        I.e. a grid cell is True if it belonged to the specified cluster at any point in time during the entire timeseries.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name. If None, infers the variable.
            cluster_id: Cluster id(s) to apply the mask for. If None, uses all clusters.

        Returns:
            Mask for the cluster id.
        """
        return self.get_cluster_mask(var, cluster_id).any(dim=self.time_dim)

    def get_cluster_times(
        self,
        var: str | None = None,
        cluster_ids: int | list[int] | range | None = None,
        numeric: bool = True,
    ) -> np.ndarray:
        """Extract all time values when/where the cluster is present.

        Args:
            var: Base variable name or custom cluster variable name.
            cluster_ids: Single cluster ID, list of IDs, range, or None for all clusters.
            numeric: If True (default), return numeric time values (e.g. seconds).
                If False, return native time coordinate (e.g. datetime64 or cftime).

        Returns:
            Flattened array of time values for every (time, y, x) cell in the cluster.
        """
        var = self._get_base_var_if_none(var)
        mask = self.get_cluster_mask(var, cluster_ids)  # (time,y,x) bool
        time_values = (
            self.numeric_time_values if numeric else self.data[self.time_dim].values
        )
        t = xr.DataArray(
            time_values,
            dims=[self.time_dim],
            coords={self.time_dim: self.data[self.time_dim]},
        )
        t3 = t.broadcast_like(mask)  # (time,y,x)
        event_times = t3.where(mask).values
        if numeric:
            event_times = event_times[np.isfinite(event_times)]
        else:
            # NaN and NaT do not equal themselves; filters both
            event_times = event_times[event_times == event_times]
        return event_times

    def _get_base_var_if_none(self, var: str | None) -> str:
        """Get the default base variable if none specified, or return the provided variable.

        Helper method to handle cases where a variable is optional and should default to the
        single base variable if one exists, or raise an error if multiple exist.

        Args:
            var: Optional variable name. If None, will attempt to use the single base variable.

        Returns:
            The variable name to use - either the provided var or the default base variable.

        Raises:
            ValueError: If var is None and multiple base variables exist.
        """
        if var is None:
            if len(self.base_vars) > 1:
                raise ValueError(
                    f"Multiple base variables exist: {self.base_vars}. Please specify which one to use."
                )
            else:
                return self.base_vars[0]
        else:
            return var

    def _is_shift_variable(self, var: str) -> bool:
        """Check if a variable is a shift variable."""
        return self.data[var].attrs.get(_attrs.VARIABLE_TYPE) == _attrs.TYPE_SHIFT

    def _is_cluster_sign_variable(self, var: str) -> bool:
        """Check if a variable is a cluster sign companion grid."""
        return (
            self.data[var].attrs.get(_attrs.VARIABLE_TYPE) == _attrs.TYPE_CLUSTER_SIGN
        )

    def _is_cluster_variable(self, var: str) -> bool:
        """Check if a variable is a cluster variable."""
        return self.data[var].attrs.get(_attrs.VARIABLE_TYPE) == _attrs.TYPE_CLUSTER

    def _is_consensus_cluster_variable(self, var: str) -> bool:
        """Check if a variable is a consensus cluster label variable."""
        return (
            self.data[var].attrs.get(_attrs.VARIABLE_TYPE)
            == _attrs.TYPE_CONSENSUS_CLUSTER
        )

    def _is_consensus_rate_variable(self, var: str) -> bool:
        """Check if a variable is a consensus rate variable."""
        return (
            self.data[var].attrs.get(_attrs.VARIABLE_TYPE) == _attrs.TYPE_CONSENSUS_RATE
        )

    def _is_base_variable(self, var: str) -> bool:
        """Check if a variable is a base variable."""
        return self.data[var].attrs.get(_attrs.VARIABLE_TYPE) not in [
            _attrs.TYPE_SHIFT,
            _attrs.TYPE_CLUSTER,
            _attrs.TYPE_CLUSTER_SIGN,
            _attrs.TYPE_CONSENSUS_CLUSTER,
            _attrs.TYPE_CONSENSUS_RATE,
            _attrs.TYPE_CONSENSUS_SIGN,
        ]

    def _aggregate_spatial(
        self,
        data: xr.DataArray,
        method: str = "raw",
        percentile: Optional[float] = None,
    ) -> xr.DataArray:
        """Aggregate data across spatial dimensions.

        Args:
            data: Data to aggregate.
            method: Aggregation method:
                - "mean": Average across space
                - "median": Median across space
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
                - "max": Maximum across space
                - "min": Minimum across space
                - "raw": Return data for each grid cell separately (default).
            percentile: Percentile value between 0-1 when using percentile aggregation.

        Returns:
            Aggregated data. If method="raw", includes cell_xy dimension.
        """
        # Check if data already has cell_xy dimension (e.g., when cluster_id=None)
        if "cell_xy" in data.dims:
            agg_dim = "cell_xy"
        else:
            agg_dim = self.space_dims

        if method == "mean":
            return data.mean(dim=agg_dim)
        elif method == "median":
            return data.median(dim=agg_dim)
        elif method == "sum":
            return data.sum(dim=agg_dim)
        elif method == "std":
            return data.std(dim=agg_dim)
        elif method == "max":
            return data.max(dim=agg_dim)
        elif method == "min":
            return data.min(dim=agg_dim)
        elif method == "percentile":
            if percentile is None:
                raise ValueError(
                    "percentile argument required for percentile aggregation"
                )
            return data.quantile(percentile, dim=agg_dim)
        elif method == "raw":
            if "cell_xy" in data.dims:
                # Already in cell_xy format, just return as-is
                return data.dropna(dim="cell_xy", how="all")
            else:
                # Stack spatial dimensions
                result = data.stack(cell_xy=self.space_dims).transpose()
                return result.dropna(dim="cell_xy", how="all")
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _finalize_timeseries(
        self,
        data: xr.DataArray,
        aggregation: str,
        percentile: Optional[float],
        normalize: Optional[str],
    ) -> xr.DataArray:
        """Aggregate spatially, normalising per-cell trajectories first when needed.

        For ``normalize`` with statistical aggregations (median, min, max, etc.),
        normalisation must happen on raw cell trajectories before aggregation so that
        band plots (min/max, percentiles) share a consistent scale with the median.
        """
        if normalize and aggregation != "raw":
            raw = self._aggregate_spatial(data, "raw", percentile)
            if normalize == "max":
                norm_val = float(raw.max())
            elif normalize == "max_each":
                norm_val = raw.max(dim=self.time_dim)
            else:
                raise ValueError(f"Unknown normalization method: {normalize}")
            raw = self._normalize_timeseries(raw, norm_val, normalize)
            return self._aggregate_spatial(raw, aggregation, percentile)

        data = self._aggregate_spatial(data, aggregation, percentile)
        if normalize:
            if normalize == "max":
                data = self._normalize_timeseries(data, float(data.max()), normalize)
            elif normalize == "max_each":
                norm_val = (
                    data.max(dim=self.time_dim)
                    if "cell_xy" in data.dims
                    else float(data.max())
                )
                data = self._normalize_timeseries(data, norm_val, normalize)
            else:
                raise ValueError(f"Unknown normalization method: {normalize}")
        return data

    def _normalize_timeseries(
        self,
        data: xr.DataArray,
        scalar_or_scalars: Union[float, xr.DataArray],
        normalize: str,
    ) -> xr.DataArray:
        """Normalise timeseries by scalar or per-trajectory scalars."""
        if isinstance(scalar_or_scalars, xr.DataArray):
            scalars = scalar_or_scalars
            valid_mask = (scalars != 0) & np.isfinite(scalars)
            if not valid_mask.any():
                self.logger.error(
                    f"Failed to normalise by {normalize}: all scalars are zero or NaN"
                )
                return data
            divisor = scalars.where(valid_mask)
        else:
            scalar = float(scalar_or_scalars)
            if scalar == 0 or np.isnan(scalar) or scalar is None:
                self.logger.error(f"Failed to normalise by {normalize} = {scalar}")
                return data
            divisor = scalar
        normalized = data / divisor
        return normalized.where(np.isfinite(normalized))

    def get_timeseries(
        self,
        var: str | None = None,
        cluster_id: Optional[Union[int, List[int]]] = None,
        cluster_var: Optional[str] = None,
        aggregation: Literal[
            "raw", "mean", "sum", "std", "median", "percentile", "max", "min"
        ]
        | str = "raw",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["max", "max_each"]] | str = None,
        keep_full_timeseries: bool = True,
        timeseries_window: Literal["full", "cluster", "shift"] | None = None,
    ) -> xr.DataArray:
        """Get time series for cluster, optionally aggregated across space.

        If cluster_id is None, returns all data from the dataset in timeseries format.

        Args:
            var: Variable name to extract time series from, or None to infer when
                only one base variable exists. Can be a base variable
                (e.g., 'thk') or a cluster variable (e.g., 'thk_dts_cluster'). If a
                cluster variable is passed, the base variable is auto-inferred.
            cluster_var: Variable name to extract cluster ids from. Defaults to None,
                in which case it is inferred from var.
            cluster_id: Single cluster ID, list of cluster IDs, or None to return all data.
            aggregation: How to aggregate spatial data:
                - "mean": Average across space
                - "median": Median across space
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
                - "max": Maximum across space
                - "min": Minimum across space
                - "raw": Return data for each grid cell separately
            percentile: Percentile value between 0-1 when using percentile aggregation.
            normalize: How to normalize the data:
                - "max": Normalize by the maximum value
                - "max_each": Normalize each trajectory by its own maximum value
                - None: Do not normalize
            keep_full_timeseries: If True, returns full time series of cluster cells. If
                False, values outside cluster bounds will be nan. Ignored when cluster_id is None.
                Superseded by ``timeseries_window`` when that argument is set.
            timeseries_window: Which portion of each trajectory to retain:
                ``"full"`` — entire time axis;
                ``"cluster"`` — only timesteps within the cluster start/end window;
                ``"shift"`` — full dts episodes whose overlap with the cluster start/end
                window is non-empty when ``cluster_id`` is set; otherwise all detection
                episodes on the full axis.

        Returns:
            The time series data for the specified cluster(s), or all data if cluster_id is None.


        Note:
            If var is a cluster variable (e.g., 'thk_dts_cluster'), the base variable
            is automatically inferred from its attributes and used for data extraction,
            while the cluster variable is used for masking. This ensures you get actual
            data values rather than cluster labels.

        """
        var = self._get_base_var_if_none(var)

        if timeseries_window is None:
            timeseries_window = "full" if keep_full_timeseries else "cluster"

        # Smart inference: if var is a cluster variable, extract base variable for data
        # and use the cluster variable for masking
        if self._is_cluster_variable(var):
            inferred_cluster_var = var
            base_var = self.data[var].attrs.get(_attrs.BASE_VARIABLE)
            if base_var is None:
                raise ValueError(
                    f"Cluster variable '{var}' has no BASE_VARIABLE attribute. "
                    f"Cannot infer which data variable to use."
                )
            var = base_var  # Use base variable for data extraction
            cluster_var = cluster_var if cluster_var else inferred_cluster_var
        else:
            cluster_var = cluster_var if cluster_var else var

        # Handle case when cluster_id is None - return all data
        if cluster_id is None:
            data = self.data[var]
            if timeseries_window == "shift":
                shift_vars = self.shift_vars_for_var(var)
                if not shift_vars:
                    raise ValueError(
                        f"No shifts variable for {var!r}; cannot use timeseries_window='shift'."
                    )
                shifts_variable = shift_vars[0]
                dts = self.data[shifts_variable]
                threshold = float(
                    self.data[shifts_variable].attrs.get(
                        _attrs.SHIFT_THRESHOLD, DEFAULT_SHIFT_THRESHOLD
                    )
                )
                data = data.where(np.abs(dts) > threshold)
            # Stack spatial dimensions to get timeseries format (same as cluster data format)
            non_time_dims = [d for d in data.dims if d != self.time_dim]
            if len(non_time_dims) > 0:
                data = data.stack(cell_xy=non_time_dims).transpose(
                    "cell_xy", self.time_dim
                )
                data = data.dropna(dim="cell_xy", how="all")
            else:
                # Already 1D timeseries, expand to match format
                data = data.expand_dims("cell_xy")
        else:
            mask = self.get_cluster_mask_spatial(cluster_var, cluster_id)

            # Apply mask
            data = self.data[var].where(mask)

            if timeseries_window == "cluster":
                start_idx = self.stats(var).time.start_timestep(cluster_id)
                end_idx = self.stats(var).time.end_timestep(cluster_id)
                time_indices = np.arange(data.sizes[self.time_dim])
                mask_in_range = (time_indices >= start_idx) & (time_indices <= end_idx)
                data = data.where(
                    xr.DataArray(mask_in_range, dims=self.time_dim, name="time_mask")
                )
            elif timeseries_window == "shift":
                shift_vars = self.shift_vars_for_var(var)
                if not shift_vars:
                    raise ValueError(
                        f"No shifts variable for {var!r}; cannot use timeseries_window='shift'."
                    )
                shifts_variable = shift_vars[0]
                dts = self.data[shifts_variable]
                threshold = DEFAULT_SHIFT_THRESHOLD
                for source in (cluster_var, shifts_variable):
                    if source in self.data:
                        threshold = float(
                            self.data[source].attrs.get(
                                _attrs.SHIFT_THRESHOLD, threshold
                            )
                        )
                        break
                start_idx = int(self.stats(var).time.start_timestep(cluster_id))
                end_idx = int(self.stats(var).time.end_timestep(cluster_id))
                overlap_mask = _compute_episode_overlap_mask(
                    dts,
                    self.time_dim,
                    threshold,
                    start_idx,
                    end_idx,
                )
                data = data.where(overlap_mask)

        # Aggregate (and normalise) spatially
        data = self._finalize_timeseries(data, aggregation, percentile, normalize)

        return data

    # TODO remove in v1.1
    def get_cluster_timeseries(
        self,
        var: str,
        cluster_id: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ) -> xr.DataArray:
        """Deprecated alias for ``get_timeseries()``."""
        self.logger.warning(
            "The method `get_cluster_timeseries` is deprecated and will be removed in a future version. "
            "Please use `get_timeseries` instead."
        )
        return self.get_timeseries(var=var, cluster_id=cluster_id, **kwargs)

    # end of TOAD object


@xr.register_dataarray_accessor("toad")
class TOADAccessor:
    """Accessor for xarray DataArrays providing TOAD-specific functionality."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_timeseries(self, time_dim: str = "time"):
        """Convert spatial data to timeseries format by stacking spatial dimensions.

        Args:
            time_dim: Name of the time dimension. Defaults to "time".

        Returns:
            DataArray with dimensions [time, cell_xy] suitable for timeseries plotting.

        Examples:
            >>> data.toad.to_timeseries().plot.line(x="time", add_legend=False, color='k', alpha=0.1);
        """

        # Check if time_dim is in dims
        if time_dim not in self._obj.dims:
            raise ValueError(
                f"Time dimension '{time_dim}' not found in data. Please specify a time dimension using the time_dim argument."
            )

        # Get all dims except time dim
        non_time_dims = [d for d in self._obj.dims if d != time_dim]

        return (
            self._obj.stack(cell_xy=non_time_dims)
            .transpose("cell_xy", time_dim)
            .dropna(dim="cell_xy", how="all")
        )
