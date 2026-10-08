import inspect
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, List, Literal, Optional, Tuple, Union, cast, overload

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    ListedColormap,
    Normalize,
    to_hex,
    to_rgb,
    to_rgba,
)
from matplotlib.figure import FigureBase
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from toad.utils import (
    DEFAULT_SHIFT_THRESHOLD,
    _attrs,
    detect_latlon_names,
    is_regular_grid,
)

_VIOLIN_SIDE_SUPPORTED = "side" in inspect.signature(Axes.violinplot).parameters

__all__ = ["Plotter", "MapStyle", "add_shared_ylabel"]

logger = logging.getLogger("TOAD")

_projection_map = {
    "plate_carree": ccrs.PlateCarree(),
    "north_pole": ccrs.NorthPolarStereo(),
    "north_polar_stereo": ccrs.NorthPolarStereo(),
    "south_pole": ccrs.SouthPolarStereo(),
    "south_polar_stereo": ccrs.SouthPolarStereo(),
    "global": ccrs.Robinson(),
    "robinson": ccrs.Robinson(),
    "mollweide": ccrs.Mollweide(),
}


def get_projection(projection: str | ccrs.Projection) -> ccrs.Projection:
    """Get a cartopy projection object from a string name or return the projection.

    Args:
        projection: Either a string name of a projection (e.cg., "plate_carree", "north_pole")
            or a cartopy Projection object. Valid string names are: "plate_carree",
            "north_pole", "north_polar_stereo", "south_pole", "south_polar_stereo",
            "global", "robinson", "mollweide".

    Returns:
        A cartopy Projection object.

    Raises:
        ValueError: If projection is a string but not a valid projection name.
        TypeError: If projection is neither a string nor a Projection object.
    """
    if isinstance(projection, str):
        if projection not in _projection_map:
            raise ValueError(
                f"Invalid projection: {projection}. Please choose between {list(_projection_map.keys())} or provide a ccrs.Projection object."
            )
        return _projection_map[projection]
    elif isinstance(projection, ccrs.Projection):
        return projection
    else:
        raise TypeError(f"Invalid projection: {projection}")


def _plot_boolean_mask_contour(
    mask, *, linewidths: float, plot_params: dict[str, Any]
) -> None:
    """Draw a boolean-mask outline, avoiding Cartopy contour projection bugs.

    PlateCarree GeoAxes need ``transform_first=True`` with 2D gridded coordinates
    (SciTools/cartopy#2176). Other axes keep the standard xarray contour path.
    """
    ax = plot_params["ax"]
    transform = plot_params.get("transform")
    projection = getattr(ax, "projection", None)

    if isinstance(projection, ccrs.PlateCarree) and isinstance(
        transform, ccrs.PlateCarree
    ):
        x_dim = plot_params["x"]
        y_dim = plot_params["y"]
        lon2, lat2 = np.meshgrid(
            mask.coords[x_dim].values,
            mask.coords[y_dim].values,
        )
        cmap = plot_params["cmap"]
        if isinstance(cmap, ListedColormap) and cmap.colors:
            color = cast(Any, cmap.colors[0])
        else:
            color = "black"
        try:
            ax.contour(
                lon2,
                lat2,
                mask.values.astype(float),
                levels=[0.5],
                transform=transform,
                transform_first=True,
                colors=[color],
                linewidths=linewidths,
                zorder=plot_params.get("zorder", 2),
            )
        except Exception as exc:
            logger.warning("Skipping cluster contour on PlateCarree map: %s", exc)
        return

    mask.plot.contour(
        levels=1,
        linewidths=linewidths,
        **plot_params,
    )


default_cmap = "tab20b"
default_cmap_other = ListedColormap(plt.cm.Greys_r(np.linspace(0.25, 0.75, 256)))  # type: ignore


def _maybe_tight_layout(
    fig: FigureBase | None,
    *,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    """Call ``tight_layout`` when *fig* is a matplotlib figure-like container."""
    if fig is None:
        return
    tight_layout = getattr(fig, "tight_layout", None)
    if tight_layout is None:
        return
    if rect is not None:
        tight_layout(rect=rect)
    else:
        tight_layout()


def add_shared_ylabel(
    fig: FigureBase,
    axes: Sequence[Axes],
    label: str,
    *,
    pad: float = 0.03,
    fontsize: float | None = None,
) -> None:
    """Place a single rotated y-axis label beside a column of subplots.

    The label is centred vertically on the union of *axes* bounding boxes in
    figure coordinates. Call after layout is final (uses ``fig.canvas.draw()``).

    Args:
        fig: Figure containing *axes*.
        axes: Subplots sharing the same y quantity (typically one column).
        label: Label text.
        pad: Horizontal offset left of the leftmost axes (figure fraction).
        fontsize: Label font size; defaults to ``axes.labelsize``.
    """
    if not axes or not label:
        return

    canvas = getattr(fig, "canvas", None)
    if canvas is not None:
        canvas.draw()

    positions = [ax.get_position() for ax in axes]
    top = max(p.y1 for p in positions)
    bottom = min(p.y0 for p in positions)
    left = min(p.x0 for p in positions)
    fig.text(
        left - pad,
        (top + bottom) / 2,
        label,
        va="center",
        ha="right",
        rotation="vertical",
        fontsize=fontsize or plt.rcParams.get("axes.labelsize", 10),
    )


def _discrete_colors_from_cmap(cmap: Union[str, ListedColormap], n: int) -> list[Any]:
    """Sample ``n`` colours from a colormap (same rule as :meth:`Plotter.consensus_map`)."""
    if n <= 0:
        return []
    if isinstance(cmap, str):
        base_cmap = plt.get_cmap(cmap)
        return [base_cmap(i) for i in np.linspace(0.0, 1.0, n)]
    cmap_colors: list = cmap.colors  # type: ignore
    if len(cmap_colors) < n:
        cmap_colors = cmap_colors * (n // len(cmap_colors) + 1)
    return cmap_colors[:n]


_CMIP_MEMBER_IN_CLUSTER_VAR = re.compile(r"(r\d+i\d+p\d+f\d+)")
_SIMPLE_MEMBER_IN_CLUSTER_VAR = re.compile(r"_r(\d+)_")


def _member_id_from_cluster_var(cluster_var: str) -> str:
    """Extract CMIP-style member id (e.g. ``r1i1p1f1``) from an input cluster variable name."""
    m = _CMIP_MEMBER_IN_CLUSTER_VAR.search(cluster_var)
    if m:
        return m.group(1)
    m = _SIMPLE_MEMBER_IN_CLUSTER_VAR.search(cluster_var)
    if m:
        return f"r{m.group(1)}"
    return cluster_var


def _realisation_from_cluster_var(cluster_var: str) -> str:
    """Extract realisation index (e.g. ``r1``) from an input cluster variable name."""
    member = _member_id_from_cluster_var(cluster_var)
    m = re.match(r"r(\d+)", member)
    if m:
        return f"r{m.group(1)}"
    return member


def _input_cluster_legend_label(
    cluster_var: str,
    *,
    n_cells: int,
    label_style: Literal["cluster_var", "member_id", "realisation"] = "cluster_var",
    include_n_cells: bool = True,
) -> str:
    if label_style == "member_id":
        base = _member_id_from_cluster_var(cluster_var)
    elif label_style == "realisation":
        base = _realisation_from_cluster_var(cluster_var)
    else:
        base = cluster_var
    if include_n_cells:
        return f"({n_cells}) {base}"
    return base


def _add_horizontal_left_map_colorbar(
    fig: Any,
    ax: Axes,
    mappable: Any,
    label: str,
    *,
    width_frac: float,
    pad: float,
    aspect: float,
    ticks: list[Any] | None = None,
) -> Any:
    """Horizontal colorbar under *ax*, left-aligned; label just right of the bar."""
    fig.canvas.draw()
    ax_pos = ax.get_position()
    bar_w = ax_pos.width * width_frac
    bar_h = max(bar_w / aspect, 0.015)
    y0 = ax_pos.y0 - pad * ax_pos.height - bar_h

    cax = fig.add_axes([ax_pos.x0, y0, bar_w, bar_h])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cax.set_xlabel("")
    cb.set_label("")
    if ticks is not None:
        cb.set_ticks(ticks)

    cb.ax.text(
        1.05,
        0.5,
        label,
        transform=cb.ax.transAxes,
        ha="left",
        va="center",
        fontsize=plt.rcParams.get("axes.labelsize", 10),
        clip_on=False,
    )
    return cb


def _legend_shrink_to_fit_axes(
    ax: Axes,
    *,
    loc: str = "best",
    fontsize_max: Optional[float] = None,
    fontsize_min: float = 4.0,
    step: float = 0.5,
    pad_px: float = 2.0,
) -> Any:
    """Draw a legend and reduce *fontsize* until its bbox lies inside the axes (display coords).

    Returns the final :class:`~matplotlib.legend.Legend`, or *None* if there are no
    labelled artists.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    fig = ax.get_figure()
    if fig is None:
        return ax.legend(handles, labels, loc=loc)

    if fontsize_max is None:
        fs0 = plt.rcParams["legend.fontsize"]
        if isinstance(fs0, (int, float, np.integer, np.floating)):
            hi = float(fs0)
        else:
            hi = float(FontProperties(size=fs0).get_size_in_points())
    else:
        hi = float(fontsize_max)

    lo = float(fontsize_min)
    if hi < lo:
        hi, lo = lo, hi

    old = ax.get_legend()
    if old is not None:
        old.remove()

    def _fits(legend: Any) -> bool:
        fig.canvas.draw()
        leg_bb = legend.get_window_extent()
        ax_bb = ax.get_window_extent()
        tol = pad_px
        return (
            leg_bb.x0 >= ax_bb.x0 - tol
            and leg_bb.x1 <= ax_bb.x1 + tol
            and leg_bb.y0 >= ax_bb.y0 - tol
            and leg_bb.y1 <= ax_bb.y1 + tol
        )

    leg: Any = None
    fs = hi
    while fs >= lo:
        leg = ax.legend(handles, labels, loc=loc, fontsize=fs)
        if _fits(leg):
            return leg
        if leg is not None:
            leg.remove()
        fs -= step

    return ax.legend(handles, labels, loc=loc, fontsize=lo)


@dataclass
class MapStyle:
    """Configuration for map styling parameters.

    This dataclass contains all the configuration options for styling maps
    with Plotter, including coastline, grid, and projection settings.
    """

    resolution: Literal["110m", "50m", "10m"] | str = "110m"
    coastline_linewidth: float = 0.5
    border_linewidth: float = 0.25
    grid_labels: bool = False
    grid_lines: bool = True
    grid_style: str = "--"
    grid_width: float = 0.5
    grid_color: str = "gray"
    grid_alpha: float = 0.5
    borders: bool = True
    projection: Optional[str | ccrs.Projection] = (
        None  # if lat/lon PlateCarree is used by default
    )
    extent: Tuple[float | int, float | int, float | int, float | int] | None = None
    map_frame: bool = True
    continent_shading: bool = False
    continent_shading_color: str = "#E9E9E9"
    ocean_shading: bool = False
    ocean_shading_color: str = "#E9E9E9"

    # Cluster map visualization options
    plot_contour: bool = True
    plot_fill: bool = True
    add_labels: bool = True
    contour_linewidth: float = 1.5
    other_legend_pos: Optional[Tuple[float, float]] = None
    other_legend: bool = True
    cluster_alpha: float = 0.75
    other_cluster_alpha: float = 0.5


def _normalize_map_style(
    map_style: Optional[Union[MapStyle, dict]] = None,
) -> MapStyle:
    """Normalize map_style to MapStyle.

    Args:
        map_style: Either a MapStyle object, a dict with MapStyle fields, or None.

    Returns:
        MapStyle object. If map_style is None, returns default MapStyle().
        If map_style is a dict, creates MapStyle from it (missing keys use defaults).
    """
    if map_style is None:
        return MapStyle()
    elif isinstance(map_style, MapStyle):
        return map_style
    elif isinstance(map_style, dict):
        # Create MapStyle from dict, missing keys will use defaults
        return MapStyle(**map_style)
    else:
        raise TypeError(
            f"map_style must be MapStyle, dict, or None, got {type(map_style)}"
        )


class Plotter:
    """Plotting utilities for TOAD objects.

    The Plotter class provides methods for creating publication-ready visualizations
    of TOAD data, including maps, timeseries, and statistical plots.

    Args:
        td: TOAD object containing the data to plot
    """

    def __init__(self, td):
        from toad import TOAD

        self.td: TOAD = td

    # Overloads are used for type hinting
    @overload
    def map(
        self,
        nrows: Literal[1] = 1,
        ncols: Literal[1] = 1,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        subplot_spec: Any = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[FigureBase, Axes]: ...

    @overload
    def map(
        self,
        nrows: int,
        ncols: int = 1,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        subplot_spec: Any = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[FigureBase, np.ndarray]: ...

    def map(
        self,
        nrows: int = 1,
        ncols: int = 1,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        subplot_spec: Optional[Any] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[FigureBase, Union[Axes, np.ndarray]]:
        """Create map plots with standard features.

        Args:
            nrows: Number of rows in subplot grid
            ncols: Number of columns in subplot grid
            figsize: Figure size (width, height) in inches. If None, not set (matplotlib default).
            height_ratios: List of height ratios for subplots
            width_ratios: List of width ratios for subplots
            subplot_spec: A gridspec subplot spec to place the map in
            map_style: Map style configuration. Can be a MapStyle object, a dict with
                     MapStyle fields, or None (uses defaults). If dict, missing keys use defaults.
                     The projection is set via map_style.projection.

        Returns:
            Tuple of (figure, axes). If nrows=1 and ncols=1, returns a single Axes.
            Otherwise, returns a numpy array of axes.
        """
        # Normalize map_style to MapStyle
        config = _normalize_map_style(map_style)

        # Check if data has lat/lon coordinates
        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        # Determine if we should use a projection
        projection_obj = None

        if config.projection is None:
            # Default to PlateCarree for lat/lon if projection not specified
            if has_latlon:
                projection_obj = get_projection("plate_carree")
        else:
            projection_obj = get_projection(config.projection)

        if subplot_spec is not None:
            # Create map in existing figure using subplot_spec
            fig = plt.gcf()
            ax = fig.add_subplot(subplot_spec, projection=projection_obj)
            axs = ax
        else:
            # Create new figure with subplots
            gridspec_kw = {}
            if height_ratios:
                gridspec_kw["height_ratios"] = height_ratios
            if width_ratios:
                gridspec_kw["width_ratios"] = width_ratios

            subplot_kw = {"projection": projection_obj}

            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=figsize,
                subplot_kw=subplot_kw,
                gridspec_kw=gridspec_kw if gridspec_kw else None,
            )

        # Ensure axs is always an array for consistent iteration
        axs_array = np.array(axs, ndmin=2)

        # Add map features and set extent for all axes (only if using projection)
        for ax in axs_array.flat:
            if hasattr(ax, "projection"):
                _add_map_features(ax, config)

                # Set extent if not specified
                if config.extent is None:
                    if ax.projection == ccrs.SouthPolarStereo():
                        ax.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
                    elif ax.projection == ccrs.NorthPolarStereo():
                        ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
                else:
                    ax.set_extent(config.extent, crs=ccrs.PlateCarree())

            # toggle frame on/off
            ax.set_frame_on(config.map_frame)

        # Return single axis or array
        if axs_array.size == 1:
            return fig, axs_array[0, 0]
        else:
            return fig, np.squeeze(axs_array)

    @overload
    def cluster_map(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        map_cmap_other: Optional[Union[str, Colormap]] = default_cmap_other,
        include_all_clusters: bool = True,
        subplots: Literal[False] = False,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        **kwargs: Any,
    ) -> Tuple[FigureBase | None, Optional[Axes]]: ...

    @overload
    def cluster_map(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        map_cmap_other: Optional[Union[str, Colormap]] = default_cmap_other,
        include_all_clusters: bool = True,
        subplots: Literal[True],
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        **kwargs: Any,
    ) -> Tuple[FigureBase | None, Optional[np.ndarray]]: ...

    def cluster_map(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        map_cmap_other: Optional[Union[str, Colormap]] = default_cmap_other,
        include_all_clusters: bool = True,
        subplots: bool = False,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        **kwargs: Any,
    ) -> Tuple[FigureBase | None, Optional[Union[Axes, np.ndarray]]]:
        """Plot one or multiple clusters on a map.

        Args:
            var: Base variable name (e.g. 'temperature', will look for
                        'temperature_cluster') or custom cluster variable name. If None, TOAD will attempt to infer which variable to use.
                A ValueError is raised if the variable cannot be uniquely determined.
            cluster_ids: Single cluster ID or list of cluster IDs to plot.
                         Defaults to range(9) (clusters 0-8) if not provided.
            map_style: Map style configuration. Can be a MapStyle object, a dict with
                     MapStyle fields, or None (uses defaults). If dict, missing keys use defaults.
                     Controls projection, grid, borders, and cluster visualization options
                     (plot_contour, plot_fill, add_labels, contour_linewidth).
            ax: Matplotlib axes to plot on. Creates new figure if None. Cannot be used with subplots=True.
            color: Color for cluster visualization. Can be:
                - A single color (str, hex, RGB tuple) to use for all clusters.
                - A list of colors to use for each cluster. Overrides cmap.
            cmap: Colormap for multiple clusters. Used only if color is None.
            map_cmap_other: Colormap for remaining clusters. Can be:
                - A string (e.g., "jet", "cividis") to use a built-in colormap.
                - A matplotlib colormap object.
            include_all_clusters: If True, plot all clusters on the map. If False, only plot selected clusters.
                Defaults to True.
            subplots: If True, plot each cluster on its own subplot. Defaults to False.
            figsize: Figure size when subplots=True. Defaults to (12, 3 * nrows).
            ncols: Number of columns in subplot grid when subplots=True. Defaults to 3.
            **kwargs: Additional arguments passed to xarray.plot methods
                      (e.g., `plot`, `plot.contour`).

        Returns:
            Tuple of (figure, axes). Figure is None if ax was provided.
            When subplots=True, axes is a numpy array of axes.

        Raises:
            ValueError: If no clusters found for given variable, or if ax is provided with subplots=True.
            TypeError: If `cluster_ids` is not an int, list, ndarray, range, or None,
                       or if `cmap` is not a string or ListedColormap.
        """
        # Normalize map_style to MapStyle
        config = _normalize_map_style(map_style)

        # Get cluster visualization options from map_style
        plot_contour = config.plot_contour
        plot_fill = config.plot_fill
        add_labels = config.add_labels
        contour_linewidth = config.contour_linewidth

        if not (plot_fill or plot_contour):
            raise ValueError("plot_fill and plot_contour cannot both be False")

        # plot_contour is not supported on irregular grids
        if (plot_contour and not plot_fill) and not is_regular_grid(self.td.data):
            raise ValueError(
                "plot_contour is not supported on irregular grids. Use plot_fill=True instead."
            )

        var = self.td._get_base_var_if_none(var)
        # get_clusters raises ValueError if no clusters found, so no need to check for None
        clusters_obj = self.td.get_clusters(var)

        # Check for incompatible parameters
        if subplots and ax is not None:
            raise ValueError(
                "Cannot use ax parameter with subplots=True. Set ax=None when using subplots."
            )

        # Plot all clusters (except -1) if no clusters passed
        all_cluster_ids = clusters_obj.cluster_ids
        cluster_ids = (
            cluster_ids
            if cluster_ids is not None
            else all_cluster_ids[all_cluster_ids != -1]
        )

        # Check that we have a valid clusters value
        if not isinstance(cluster_ids, (int, list, np.ndarray, range)):
            raise TypeError("clusters must be int, list, np.ndarray, range, or None")

        # Convert single cluster_id to list for consistent handling
        if isinstance(cluster_ids, int):
            single_plot = True
            cluster_ids = [cluster_ids]
        else:
            single_plot = False
            cluster_ids = list(cluster_ids)  # Convert to list for consistent indexing

        # Filter out cluster IDs that don't exist
        valid_cluster_ids = [id for id in cluster_ids if id in all_cluster_ids]
        if len(valid_cluster_ids) == 0:
            # raise ValueError(f"No valid clusters found in clusters for variable {var}")
            logger.warning(f"No valid clusters found in clusters for variable {var}")
            return None, None

        # Setup subplots if requested
        if subplots:
            n_clusters = len(valid_cluster_ids)
            nrows = int(np.ceil(n_clusters / ncols))
            if figsize is None:
                figsize = (12, 3 * nrows)
            fig, axs = self.map(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                map_style=config,
            )
            # Ensure axs is always an array for consistent iteration
            axs_array: Optional[np.ndarray] = np.array(axs, ndmin=2)
        else:
            if ax is None:
                fig, ax = self.map(figsize=figsize, map_style=config)
            else:
                fig = None
            axs_array = None

        # Create color list for each cluster (based on valid_cluster_ids)
        n_valid = len(valid_cluster_ids)
        if color is not None:
            # If color is a list, use it directly (one color per cluster)
            if (
                isinstance(color, (list, tuple))
                and len(color) > 1
                and not all(isinstance(c, (int, float)) for c in color)
            ):
                color_list = color
                if len(color_list) < n_valid:
                    # Repeat colors if needed
                    color_list = color_list * (n_valid // len(color_list) + 1)
                color_list = color_list[
                    :n_valid
                ]  # Trim to match valid_cluster_ids length
            else:
                # Single color for all clusters
                color_list = [color] * n_valid
        else:
            color_list = _discrete_colors_from_cmap(cmap, n_valid)

        # Create a ListedColormap for each cluster
        cmap_list = [ListedColormap([c]) for c in color_list]

        # Initialize plot_params for use after the loop (for remaining clusters)
        plot_params: dict[str, Any] = {}

        for i, id in enumerate(valid_cluster_ids):
            # Select the appropriate axis for this cluster
            if subplots:
                # Calculate subplot index
                row = i // ncols
                col = i % ncols
                current_ax: Axes = axs_array[row, col]  # type: ignore
            else:
                # ax is guaranteed to be set at this point (created if None)
                if ax is None:
                    raise ValueError("ax should be set when subplots=False")
                current_ax = ax

            # Get the colormap for this cluster
            cluster_cmap = cmap_list[i]

            # Get mask for clustered cells (including -1 if specified in cluster_ids)
            mask = self.td.get_cluster_mask_spatial(var, id)

            # prepare common plot parameters
            plot_params = {
                "ax": current_ax,
                "cmap": cluster_cmap,
                "add_colorbar": False,
                "alpha": config.cluster_alpha,
                **kwargs,
            }

            plot_params, use_pcolormesh = self._prepare_map_plot_params(
                current_ax, plot_params
            )

            # Z-order: each cluster gets fill and contour in same layer so later clusters
            # correctly stack on top. Contour has higher default z-order than pcolormesh,
            # so without this, Cluster 0's contour would appear above Cluster 5's fill.
            base_z = 2 * i
            plot_params["zorder"] = base_z

            if plot_fill:
                # Don't plot values outside mask: FALSE -> np.nan
                # Use pcolormesh explicitly for regular axes to ensure proper coordinate handling
                if use_pcolormesh:
                    mask.where(mask, np.nan).plot.pcolormesh(
                        **plot_params,
                    )
                else:
                    mask.where(mask, np.nan).plot(
                        **plot_params,
                    )

            # contour plots don't work for irregular grids
            if plot_contour and is_regular_grid(self.td.data):
                # Make contour color darker (use color_list directly to avoid type issues)
                contour_color = cast(Any, color_list[i])
                color_rgba = to_rgba(contour_color)
                darker_color = (
                    color_rgba[0] * 0.8,
                    color_rgba[1] * 0.8,
                    color_rgba[2] * 0.8,
                    color_rgba[3],
                )
                plot_params["cmap"] = ListedColormap([darker_color])
                plot_params["zorder"] = base_z + 1  # contour just above its own fill

                _plot_boolean_mask_contour(
                    mask,
                    linewidths=contour_linewidth,
                    plot_params=plot_params,
                )

            if add_labels:
                # returns space_dims[0, 1], so y, x or lon, lat
                # Uses the point furthest from the cluster edge for robust labeling
                y, x = self.td.stats(var).space.central_point_for_labeling(id)
                if np.isnan(x) or np.isnan(y):
                    # Get median coordinates as fallback
                    y, x = self.td.stats(var).space.footprint_median(id)

                if not (np.isnan(x) or np.isnan(y)):
                    # Use color_list directly to avoid type issues with cluster_cmap.colors
                    # Cast to Any since color_list[i] can be str or tuple (from colormap)
                    cluster_color = cast(Any, color_list[i])
                    _cluster_annotate(
                        current_ax,
                        x,
                        y,
                        id,
                        cluster_color,  # type: ignore[arg-type]
                        transform=plot_params.get("transform"),
                    )  # type: ignore
                else:
                    print(
                        f"Warning: Could not find valid label position for cluster {id}"
                    )

            # Set title for subplots or single plot
            if subplots:
                current_ax.set_title(f"{var}_cluster {id}")
            elif single_plot:
                current_ax.set_title(f"{var}_cluster {id}")

        # Plot remaining clusters (only when not using subplots, include_all_clusters is True, and there are valid clusters to plot)
        if (
            map_cmap_other
            and not subplots
            and include_all_clusters
            and len(valid_cluster_ids) > 0
        ):
            # ax is guaranteed to be set at this point when subplots=False
            if ax is None:
                raise ValueError("ax should be set when subplots=False")
            remaining_cluster_ids = [  # get unplotted clusters ids (except -1)
                int(id)
                for id in all_cluster_ids
                if id not in valid_cluster_ids and id != -1
            ]
            if len(remaining_cluster_ids) > 0:
                mask = self.td.get_cluster_mask(var, remaining_cluster_ids)
                cl = self.td.get_clusters(var).where(mask)

                plot_params["cmap"] = map_cmap_other
                plot_params["alpha"] = config.other_cluster_alpha
                plot_params["ax"] = ax  # Use the single ax for remaining clusters
                cl.max(dim=self.td.time_dim).plot(
                    **plot_params,
                )  # type: ignore

                # Pass the colormap to the legend function
                if config.other_legend:
                    _add_gradient_legend(
                        ax,
                        remaining_cluster_ids[0],
                        remaining_cluster_ids[-1],
                        legend_pos=config.other_legend_pos,
                        var=var,
                        alpha=config.other_cluster_alpha,
                        cmap=plt.get_cmap(map_cmap_other)
                        if isinstance(map_cmap_other, str)
                        else map_cmap_other,
                    )

        # Return appropriate axes based on subplots setting
        if subplots:
            if axs_array is None:
                raise ValueError("axs_array should be set when subplots=True")
            if axs_array.size > 1:
                return fig, np.squeeze(axs_array)  # type: ignore
            else:
                return fig, axs_array[0, 0]  # type: ignore
        else:
            if ax is None:
                raise ValueError("ax should be set when subplots=False")
        return fig, ax

    def _timeseries_var_ylabel(self, var_name: str) -> str:
        """Y-axis label from a data variable's metadata."""
        try:
            da = self.td.data[var_name]
        except (KeyError, AttributeError, TypeError):
            return var_name
        name = da.attrs.get("long_name") or var_name
        units = str(da.attrs.get("units", "")).strip()
        if units:
            return f"{name} ({units})"
        return str(name)

    def _time_axis_ylabel(self) -> str:
        """Y-axis label from the dataset time coordinate (name and units)."""
        td = self.td
        try:
            t = td.data.coords[td.time_dim]
        except (KeyError, AttributeError, TypeError):
            return str(getattr(td, "time_dim", "time"))
        name = t.attrs.get("long_name")
        if not name:
            name = str(getattr(t, "name", None) or td.time_dim)
        units = str(t.attrs.get("units", "")).strip()
        if units:
            return f"{name} ({units})"
        return name

    def _transition_time_ylabel(self) -> str:
        """Y-axis label for transition-time samples (matches :attr:`TOAD.numeric_time_values`).

        Uses the time coordinate ``long_name`` (or dimension name) and
        :meth:`TOAD.numeric_time_values_unit`, which reflects datetime-to-seconds conversion
        when applicable. If the axis range still looks wrong, the coordinate values on
        ``td.data`` may not match the metadata—pass ``ylabel=`` on the plot or fix coords.
        """
        td = self.td
        try:
            t = td.data.coords[td.time_dim]
        except (KeyError, AttributeError, TypeError):
            return str(getattr(td, "time_dim", "time"))
        name = t.attrs.get("long_name")
        if not name:
            name = str(getattr(t, "name", None) or td.time_dim)
        units = str(td.numeric_time_values_unit()).strip()
        if units:
            return f"{name} ({units})"
        return name

    def consensus_map(
        self,
        consensus_var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        subplots: bool = False,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        colorbar_shrink: float = 0.38,
        colorbar_pad: float = 0.025,
        colorbar_aspect: float = 28.0,
        colorbar_orientation: Literal["horizontal", "vertical"] = "horizontal",
        colorbar_location: str | None = None,
        colorbar_label: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[FigureBase | None, Optional[Union[Axes, np.ndarray]]]:
        """Plot consensus cluster footprints on a map (same layout ideas as :meth:`cluster_map`).

        Uses ``(consensus == id).any(time)`` as a 2D mask per consensus cluster id. Labels
        are disabled for consensus maps (centroid logic is not tied to :class:`Stats`).
        When ``subplots=False``, a horizontal colorbar below the map shows the numeric range
        of the plotted consensus cluster ids (one colour per id).

        Args:
            consensus_var: Name of the consensus labels variable on ``td.data``. If None and
                exactly one consensus variable exists, it is used.
            cluster_ids: Consensus cluster id(s) to plot. Defaults to ``range(9)``; missing
                ids are skipped with a warning.
            ax, color, cmap, subplots, ncols, figsize, map_style, **kwargs: Same role as
                :meth:`cluster_map` (without ``map_cmap_other`` / ``include_all_clusters``).
            colorbar_shrink: Horizontal colorbar length as a fraction of the parent axes span
                (matplotlib ``shrink``; only when ``subplots=False``). Smaller is shorter.
            colorbar_pad: Space between the map and the colorbar (axes fraction).
            colorbar_aspect: Width/height ratio of the colorbar strip (larger = shorter bar).
            colorbar_orientation: ``\"horizontal\"`` (default, below map) or ``\"vertical\"``.
            colorbar_location: For horizontal bars, ``\"left\"`` left-aligns the bar under the
                map with the label to its right; default centres the bar. For vertical bars,
                passed to matplotlib ``location`` (e.g. ``\"left\"``).
            colorbar_label: Label under the discrete cluster-id colorbar. Default is
                ``\"consensus cluster id\"`` (not the variable name).

        Returns:
            ``(fig, ax)`` or ``(fig, axs)`` when ``subplots=True``.
        """
        config = _normalize_map_style(map_style)
        config = replace(config, add_labels=False)

        plot_contour = config.plot_contour
        plot_fill = config.plot_fill
        contour_linewidth = config.contour_linewidth

        if subplots and ax is not None:
            raise ValueError(
                "Cannot use ax parameter with subplots=True. Set ax=None when using subplots."
            )

        if not (plot_fill or plot_contour):
            raise ValueError("plot_fill and plot_contour cannot both be False")
        if (plot_contour and not plot_fill) and not is_regular_grid(self.td.data):
            raise ValueError(
                "plot_contour is not supported on irregular grids. Use plot_fill=True instead."
            )

        consensus_var = self.td._resolve_consensus_var(consensus_var)
        da = self.td.data[consensus_var]
        time_dim = self.td.time_dim
        if time_dim not in da.dims:
            raise ValueError(
                f"Consensus variable {consensus_var!r} has no time dimension {time_dim!r}."
            )

        raw_ids = da.attrs.get(_attrs.CLUSTER_IDS)
        if raw_ids is None:
            v = np.asarray(da.values, dtype=np.float64)
            all_cluster_ids = np.unique(v[np.isfinite(v) & (v >= 0)]).astype(np.int64)
        else:
            all_cluster_ids = np.asarray(raw_ids, dtype=np.int64)
            all_cluster_ids = all_cluster_ids[all_cluster_ids >= 0]

        cluster_ids = (
            cluster_ids if cluster_ids is not None else np.atleast_1d(all_cluster_ids)
        )
        if not isinstance(cluster_ids, (int, list, np.ndarray, range)):
            raise TypeError("cluster_ids must be int, list, np.ndarray, range, or None")

        if isinstance(cluster_ids, int):
            single_plot = True
            cluster_ids = [cluster_ids]
        else:
            single_plot = False
            cluster_ids = list(cluster_ids)

        valid_cluster_ids = [
            int(i) for i in cluster_ids if int(i) in set(all_cluster_ids)
        ]
        if len(valid_cluster_ids) == 0:
            logger.warning(
                f"No valid consensus cluster ids in {cluster_ids} for {consensus_var!r}"
            )
            return None, None

        if subplots:
            n_clusters = len(valid_cluster_ids)
            nrows = int(np.ceil(n_clusters / ncols))
            if figsize is None:
                figsize = (12, 3 * nrows)
            fig, axs = self.map(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                map_style=config,
            )
            axs_array: Optional[np.ndarray] = np.array(axs, ndmin=2)
        else:
            if ax is None:
                fig, ax = self.map(figsize=figsize, map_style=config)
            else:
                fig = None
            axs_array = None

        # colors
        n_valid = len(valid_cluster_ids)
        if color is not None:
            if (
                isinstance(color, (list, tuple))
                and len(color) > 1
                and not all(isinstance(c, (int, float)) for c in color)
            ):
                color_list = color
                if len(color_list) < n_valid:
                    color_list = color_list * (n_valid // len(color_list) + 1)
                color_list = color_list[:n_valid]
            else:
                color_list = [color] * n_valid
        else:
            color_list = _discrete_colors_from_cmap(cmap, n_valid)

        cmap_list = [ListedColormap([c]) for c in color_list]

        for i, cid in enumerate(valid_cluster_ids):
            if subplots:
                row = i // ncols
                col = i % ncols
                current_ax: Axes = axs_array[row, col]  # type: ignore
            else:
                if ax is None:
                    raise ValueError("ax should be set when subplots=False")
                current_ax = ax

            cluster_cmap = cmap_list[i]
            mask = (da == cid).any(dim=time_dim)
            plot_params = {
                "ax": current_ax,
                "cmap": cluster_cmap,
                "add_colorbar": False,
                "alpha": config.cluster_alpha,
                **kwargs,
            }
            plot_params, use_pcolormesh = self._prepare_map_plot_params(
                current_ax, plot_params
            )
            base_z = 2 * i
            plot_params["zorder"] = base_z
            if plot_fill:
                if use_pcolormesh:
                    mask.where(mask, np.nan).plot.pcolormesh(**plot_params)
                else:
                    mask.where(mask, np.nan).plot(**plot_params)
            if plot_contour and is_regular_grid(self.td.data):
                contour_color = cast(Any, color_list[i])
                color_rgba = to_rgba(contour_color)
                darker_color = (
                    color_rgba[0] * 0.8,
                    color_rgba[1] * 0.8,
                    color_rgba[2] * 0.8,
                    color_rgba[3],
                )
                plot_params["cmap"] = ListedColormap([darker_color])
                plot_params["zorder"] = base_z + 1
                _plot_boolean_mask_contour(
                    mask,
                    linewidths=contour_linewidth,
                    plot_params=plot_params,
                )

            if subplots:
                current_ax.set_title(f"{consensus_var} {cid}")
            elif single_plot:
                current_ax.set_title(f"{consensus_var} {cid}")

        if not subplots and ax is not None and len(valid_cluster_ids) > 0:
            _add_consensus_cluster_discrete_colorbar(
                ax.get_figure(),
                ax,
                sorted_ids=np.sort(np.asarray(valid_cluster_ids, dtype=np.int64)),
                color_list=list(color_list),
                label=colorbar_label
                if colorbar_label is not None
                else "Consensus cluster id",
                shrink=colorbar_shrink,
                pad=colorbar_pad,
                aspect=colorbar_aspect,
                orientation=colorbar_orientation,
                location=colorbar_location,
            )

        if subplots:
            if axs_array is None:
                raise ValueError("axs_array should be set when subplots=True")
            if axs_array.size > 1:
                return fig, np.squeeze(axs_array)  # type: ignore
            return fig, axs_array[0, 0]  # type: ignore
        if ax is None:
            raise ValueError("ax should be set when subplots=False")
        return fig, ax

    def consensus_rate_map(
        self,
        consensus_var: str | None = None,
        *,
        time_reduce: Literal["max", "mean"] = "max",
        ax: Optional[Axes] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        cmap: Optional[Union[str, Colormap]] = "cividis",
        vmin: float = 0.0,
        vmax: float = 1.0,
        add_colorbar: bool = True,
        colorbar_orientation: Literal["horizontal", "vertical"] | None = None,
        colorbar_location: str | None = None,
        colorbar_shrink: float | None = None,
        colorbar_pad: float | None = None,
        colorbar_aspect: float | None = None,
        colorbar_label: str | None = None,
        cbar_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Tuple[FigureBase | None, Axes]:
        """Map member-support consensus rate from a consensus run (time-collapsed).

        Uses the companion field ``{consensus_var}_rate``: at each spacetime
        cell this is (supporting inputs) / (total inputs) on native event voxels,
        including voxels below the consensus threshold. The map collapses time with
        ``max`` (default) or ``mean`` over finite values, then masks cells where the
        result is zero (no input ever detected an event there).

        Args:
            consensus_var: Consensus labels variable; inferred when unique.
            time_reduce: ``\"max\"`` (peak agreement at any time) or ``\"mean\"`` over
                timesteps with data.
            ax: Axes to draw on; if None, a new figure is created via :meth:`map`.
            map_style: Passed to :meth:`map` when *ax* is None.
            cmap: Colormap for the consensus rate field.
            vmin, vmax: Color scale bounds; default ``[0, 1]``.
            add_colorbar: Whether to draw a colorbar (default True).
            colorbar_orientation: ``\"horizontal\"`` or ``\"vertical\"``. If None,
                xarray uses its default (vertical on the right).
            colorbar_location: For horizontal bars, ``\"left\"`` left-aligns the bar under the
                map with the label to its right. Otherwise passed to matplotlib ``location``.
            colorbar_shrink, colorbar_pad, colorbar_aspect: Passed to the colorbar
                (matplotlib ``shrink``, ``pad``, ``aspect``). Omitted when None.
            colorbar_label: Colorbar label; default ``\"Consensus rate\"``.
            cbar_kwargs: Extra keyword arguments merged into the colorbar kwargs
                (after ``colorbar_*`` parameters).
            **kwargs: Extra arguments forwarded to :meth:`xarray.DataArray.plot` or
                ``plot.pcolormesh`` (except ``add_colorbar`` and ``cbar_kwargs``).

        Returns:
            ``(fig, ax)``; figure is None if *ax* was provided.
        """
        consensus_var = self.td._resolve_consensus_var(consensus_var)
        rate_var = self.td._resolve_consensus_rate_var(consensus_var)
        rate = self.td.data[rate_var]

        time_dim = self.td.time_dim
        space_dims = tuple(self.td.space_dims)
        if time_dim in rate.dims:
            if time_reduce == "max":
                field = rate.max(dim=time_dim, skipna=True)
            elif time_reduce == "mean":
                field = rate.mean(dim=time_dim, skipna=True)
            else:
                raise ValueError(
                    f"`time_reduce` must be 'max' or 'mean', got {time_reduce!r}."
                )
        else:
            field = rate

        for d in space_dims:
            if d not in field.dims:
                field = field.expand_dims({d: 1})
        field = field.transpose(*space_dims)
        field = field.where(field > 0)

        config = _normalize_map_style(map_style)
        if ax is None:
            fig, ax = self.map(map_style=config)
        else:
            fig = None

        title = "Consensus rate"

        cbar_label = colorbar_label if colorbar_label is not None else "Consensus rate"

        add_colorbar = kwargs.pop("add_colorbar", add_colorbar)
        orient = colorbar_orientation or "horizontal"
        loc = colorbar_location

        shrink = colorbar_shrink if colorbar_shrink is not None else 0.38
        pad = colorbar_pad if colorbar_pad is not None else 0.04
        aspect = colorbar_aspect if colorbar_aspect is not None else 28.0

        manual_h_left = add_colorbar and orient == "horizontal" and loc == "left"

        merged_cbar_kwargs: dict[str, Any] = {}
        if add_colorbar and not manual_h_left:
            merged_cbar_kwargs["label"] = cbar_label
            if colorbar_orientation is not None:
                merged_cbar_kwargs["orientation"] = colorbar_orientation
            if loc is not None and loc != "left":
                merged_cbar_kwargs["location"] = loc
            merged_cbar_kwargs["shrink"] = shrink
            merged_cbar_kwargs["pad"] = pad
            merged_cbar_kwargs["aspect"] = aspect
        if cbar_kwargs:
            merged_cbar_kwargs.update(cbar_kwargs)
        if "cbar_kwargs" in kwargs:
            merged_cbar_kwargs.update(kwargs.pop("cbar_kwargs"))

        plot_params: dict[str, Any] = {
            "ax": ax,
            "add_colorbar": add_colorbar and not manual_h_left,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            **kwargs,
        }
        if add_colorbar and not manual_h_left:
            plot_params["cbar_kwargs"] = merged_cbar_kwargs
        plot_params, use_pcolormesh = self._prepare_map_plot_params(ax, plot_params)
        if use_pcolormesh:
            field.plot.pcolormesh(**plot_params)
        else:
            field.plot(**plot_params)

        if manual_h_left:
            plot_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=plot_cmap)
            sm.set_array([])
            _add_horizontal_left_map_colorbar(
                ax.figure,
                ax,
                sm,
                cbar_label,
                width_frac=shrink,
                pad=pad,
                aspect=aspect,
            )

        ax.set_title(title)
        return ax.figure if fig is None else fig, ax

    @staticmethod
    def _concat_finite_from_shift_dists(d: dict[int, np.ndarray]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for v in d.values():
            a = np.asarray(v, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size:
                parts.append(a)
        return np.concatenate(parts, dtype=np.float64) if parts else np.array([])

    def _distributions_to_violin_tuples(
        self,
        dists: dict[int, np.ndarray],
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]],
        *,
        cmap: Union[str, ListedColormap],
        show_sum: bool,
        show_total: bool,
        total_color: str,
        pad_empty_for_violin: bool,
    ) -> tuple[list[np.ndarray], list[str], list[str]]:
        """Build violin datasets/labels from ``{id: 1D times}`` (shared by consensus and label-field plots)."""
        if not dists:
            raise ValueError("No transition-time samples available for violin plot.")

        dists_full = dict(dists)
        times_total_all = self._concat_finite_from_shift_dists(dists_full)
        if cluster_ids is not None:
            if isinstance(cluster_ids, int):
                wanted = {cluster_ids}
            else:
                wanted = {int(x) for x in cluster_ids}
            dists = {k: v for k, v in dists.items() if k in wanted}

        ids_sorted = sorted(dists.keys())
        datasets = [np.asarray(dists[cid], dtype=np.float64) for cid in ids_sorted]
        datasets = [d[np.isfinite(d)] for d in datasets]
        ids_sorted = [cid for cid, d in zip(ids_sorted, datasets) if d.size > 0]
        datasets = [d for d in datasets if d.size > 0]
        if not ids_sorted:
            raise ValueError("No finite samples left after filtering.")

        times_sum_plotted = (
            np.concatenate(datasets, dtype=np.float64) if datasets else np.array([])
        )

        # Match :meth:`consensus_map` / `_discrete_colors_from_cmap` so violin hues agree with map +
        # colorbar when ``cmap`` is the same string or ListedColormap (avoid `_get_cmap_seq`,
        # which indexes cmap.N discretely and diverges from normalized sampling).
        disc = _discrete_colors_from_cmap(cmap, len(ids_sorted))
        colors = [to_hex(to_rgba(c)) for c in disc]

        xticklabels = [str(i) for i in ids_sorted]

        if pad_empty_for_violin:
            dataset: list[np.ndarray] = [
                arr if len(arr) > 0 else np.array([np.nan]) for arr in datasets
            ]
        else:
            dataset = list(datasets)
        if show_sum and times_sum_plotted.size > 0:
            dataset.append(times_sum_plotted)
            colors.append(total_color)
            xticklabels.append("sum")
        if show_total and times_total_all.size > 0:
            dataset.append(times_total_all)
            colors.append(total_color)
            xticklabels.append("total")
        return dataset, xticklabels, colors

    def _consensus_transition_time_groups(
        self,
        consensus_var: str | None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]],
        *,
        cmap: Union[str, ListedColormap],
        show_sum: bool,
        show_total: bool,
        total_color: str,
        pad_empty_for_violin: bool = False,
        source_input_cluster_var: str | None = None,
    ) -> Tuple[str, list[np.ndarray], list[str], list[str]]:
        """Shared transition-time samples per consensus cluster (and optional pooled groups).

        Used by :meth:`consensus_shift_times_violins`.
        """
        consensus_var = self.td._resolve_consensus_var(consensus_var)
        da = self.td.data[consensus_var]
        dists = self.td.aggregate.consensus_shift_time_distributions(
            da,
            source_input_cluster_var=source_input_cluster_var,
        )
        dataset, xticklabels, colors = self._distributions_to_violin_tuples(
            dists,
            cluster_ids,
            cmap=cmap,
            show_sum=show_sum,
            show_total=show_total,
            total_color=total_color,
            pad_empty_for_violin=pad_empty_for_violin,
        )
        return consensus_var, dataset, xticklabels, colors

    def _plot_transition_time_violin_axes(
        self,
        dataset: list[np.ndarray],
        xticklabels: list[str],
        colors: list[str],
        *,
        xlabel: str,
        show_scatter: bool = True,
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        width: float = 0.75,
        bw_method: float = 0.18,
        show_legend: bool = True,
        ylabel: Optional[str] = None,
        kde_side: Literal["left", "right"] = "right",
        point_size: float = 15.0,
        point_alpha: float = 0.75,
        jitter_half_span: Optional[float] = None,
        seed: Optional[int] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> tuple[FigureBase | None, Axes]:
        """Render violin (+ optional scatter) for transition-time sample lists; shared by consensus/label plotters."""
        n_groups = len(dataset)
        positions = np.arange(n_groups, dtype=float)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (max(10, 1.2 * n_groups), 6))
        else:
            fig = ax.get_figure()

        vp_kwargs: dict[str, Any] = {
            "showmeans": False,
            "showextrema": False,
            "showmedians": False,
            "quantiles": None,
            "bw_method": bw_method,
        }
        vp_kwargs.update(kwargs)
        if show_scatter:
            vp_kwargs["side"] = "low" if kde_side == "left" else "high"

        parts = ax.violinplot(
            dataset,
            positions=positions,
            widths=width,
            **vp_kwargs,
        )
        if show_scatter:
            _style_violin_bodies_iqr_median(
                ax, parts, dataset, positions, colors, clip_to_body=False
            )
            all_parts: list[np.ndarray] = []
            for d in dataset:
                a = np.asarray(d, dtype=np.float64)
                a = a[np.isfinite(a)]
                if a.size:
                    all_parts.append(a)
            if all_parts:
                all_y = np.concatenate(all_parts)
                y_span = float(np.ptp(all_y))
                pad = 0.05 * (y_span if y_span > 0 else 1.0)
                ax.set_ylim(float(all_y.min()) - pad, float(all_y.max()) + pad)

            x_margin = width / 2.0 + 0.05
            ax.set_xlim(-0.5 - x_margin, (n_groups - 1) + 0.5 + x_margin)

            if jitter_half_span is None:
                _jhalf = width / 8.0
            else:
                _jhalf = float(jitter_half_span)

            rng = np.random.default_rng(seed)

            for i, arr in enumerate(dataset):
                arr = np.asarray(arr, dtype=np.float64)
                mask = np.isfinite(arr)
                if not np.any(mask):
                    continue
                pos = float(positions[i])
                if kde_side == "left":
                    jitter_xc = pos + width / 4.0
                else:
                    jitter_xc = pos - width / 4.0
                n_pt = int(np.sum(mask))
                x_jitter = _jitter_strip_x(n_pt, jitter_xc, _jhalf, rng)
                ax.scatter(
                    x_jitter,
                    arr[mask],
                    s=point_size,
                    c=colors[i % len(colors)],
                    alpha=point_alpha,
                    edgecolors="#333333",
                    linewidths=0.35,
                    zorder=5,
                )
        else:
            _style_violin_bodies_iqr_median(ax, parts, dataset, positions, colors)

        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ylabel is not None else self._transition_time_ylabel())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.grid(True, axis="y", color="#d9d9d9", linestyle="-", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        if show_legend:
            if show_scatter:
                ax.legend(
                    handles=[
                        Line2D(
                            [0],
                            [0],
                            linestyle="None",
                            marker="o",
                            markerfacecolor=colors[0] if colors else "#888888",
                            markeredgecolor="#333333",
                            markersize=6,
                            markeredgewidth=0.35,
                            alpha=point_alpha,
                            label="Shift times",
                        ),
                        Line2D(
                            [0],
                            [0],
                            linestyle="None",
                            marker="o",
                            markerfacecolor=_VIOLIN_CONSENSUS_MEDIAN_FACE,
                            markeredgecolor=_VIOLIN_CONSENSUS_MEDIAN_EDGE,
                            markersize=6,
                            markeredgewidth=0.6,
                            label="Median",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color=_VIOLIN_CONSENSUS_IQR_COLOR,
                            linestyle="-",
                            linewidth=_VIOLIN_CONSENSUS_IQR_LW,
                            solid_capstyle="round",
                            label="25th–75th percentile",
                        ),
                    ],
                    loc="lower right",
                    bbox_to_anchor=(1.0, 1.0),
                    bbox_transform=ax.transAxes,
                    borderaxespad=0.0,
                    frameon=False,
                    fontsize=9,
                    ncols=3,
                )
            else:
                ax.legend(
                    handles=[
                        Line2D(
                            [0],
                            [0],
                            linestyle="None",
                            marker="o",
                            markerfacecolor=_VIOLIN_CONSENSUS_MEDIAN_FACE,
                            markeredgecolor=_VIOLIN_CONSENSUS_MEDIAN_EDGE,
                            markersize=6,
                            markeredgewidth=0.6,
                            label="Median",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color=_VIOLIN_CONSENSUS_IQR_COLOR,
                            linestyle="-",
                            linewidth=_VIOLIN_CONSENSUS_IQR_LW,
                            solid_capstyle="round",
                            label="25th–75th percentile",
                        ),
                    ],
                    loc="lower right",
                    bbox_to_anchor=(1.0, 1.0),
                    bbox_transform=ax.transAxes,
                    borderaxespad=0.0,
                    frameon=False,
                    fontsize=9,
                    ncols=2,
                )
            if tight_layout:
                _maybe_tight_layout(fig, rect=(0.0, 0.0, 1.0, 0.92))
        elif tight_layout:
            _maybe_tight_layout(fig)
        return fig, ax

    def consensus_shift_times_violins(
        self,
        consensus_var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        show_scatter: bool = True,
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        source_input_cluster_var: str | None = None,
        width: float = 0.75,
        bw_method: float = 0.18,
        show_sum: bool = False,
        show_total: bool = True,
        total_color: str = "#666666",
        show_legend: bool = True,
        ylabel: Optional[str] = None,
        kde_side: Literal["left", "right"] = "right",
        point_size: float = 15.0,
        point_alpha: float = 0.75,
        jitter_half_span: Optional[float] = None,
        seed: Optional[int] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> Tuple[FigureBase | None, Axes]:
        """Transition-time samples per consensus cluster: half-violin + scatter or full violins.

        Visualises the same pooled :func:`~toad.utils.cluster_consensus_utils.consensus_shift_time_distribution`
        samples that feed ``pooled_median_shift_time`` in :meth:`toad.postprocessing.Aggregation.consensus_summary`
        (per-cluster columns), plus optional pooled groups (``sum`` / ``total``).
        Complement :meth:`consensus_shift_times_medians` for median-of-medians from the table.

        Uses :meth:`toad.postprocessing.Aggregation.consensus_shift_time_distributions`.
        For a **single** input map’s events only, pass ``source_input_cluster_var`` to match
        that column in the long table (per-input timing within consensus groups).

        * ``show_scatter=True`` (default): half KDE on one side of each tick, fixed-width
          jittered sample points on the other; IQR and median are not clipped so they straddle
          the spine. Requires matplotlib ≥ 3.10 (``Axes.violinplot(..., side=...)``).
        * ``show_scatter=False``: standard symmetric violins with IQR bar and median clipped
          to each body (see :func:`_style_violin_bodies_iqr_median`).

        Optional pooled columns (``sum`` / ``total``) use ``total_color``; when both are True,
        ``sum`` is drawn first, then ``total``.

        Args:
            consensus_var: Consensus labels variable; inferred if there is exactly one.
            cluster_ids: Subset of consensus cluster ids. If None, plots all ids with samples.
            show_scatter: If True (default), half-violin + jittered scatter; if False, full violins.
            ax: Axes to draw on; creates a new figure if None.
            figsize: Figure size when creating a new figure.
            cmap: Colormap used to pick face colours per cluster.
            source_input_cluster_var: If set, only transition times that came from this input
                clustering (``cluster_var`` name in the long table) are used.
            width: Violin width passed to ``Axes.violinplot`` (scatter mode uses half of this per side).
            bw_method: KDE bandwidth factor passed to ``Axes.violinplot``.
            show_sum, show_total, total_color: Pooled columns (same as before).
            show_legend: If True (default), add a legend (three entries with scatter, two without).
            ylabel: Y-axis label; default :meth:`_transition_time_ylabel`.
            kde_side: For ``show_scatter=True`` only: ``\"left\"`` puts KDE left of the tick,
                jitter right; ``\"right\"`` swaps.
            point_size, point_alpha: Scatter markers when ``show_scatter=True`` (ignored otherwise).
            jitter_half_span: Half-width of horizontal jitter in *x* data units; default ``width / 8``.
            seed: RNG seed for jitter when ``show_scatter=True``.
            tight_layout: If True (default), call ``fig.tight_layout`` before returning. Set False
                when drawing into a composite figure (e.g. :meth:`consensus_overview`).
            **kwargs: Extra arguments forwarded to ``Axes.violinplot``.
        """
        if show_scatter and not _VIOLIN_SIDE_SUPPORTED:
            raise RuntimeError(
                "consensus_shift_times_violins(..., show_scatter=True) requires matplotlib>=3.10 "
                "(violinplot side=... for half violins)."
            )

        _cv, dataset, xticklabels, colors = self._consensus_transition_time_groups(
            consensus_var,
            cluster_ids,
            cmap=cmap,
            show_sum=show_sum,
            show_total=show_total,
            total_color=total_color,
            pad_empty_for_violin=True,
            source_input_cluster_var=source_input_cluster_var,
        )

        return self._plot_transition_time_violin_axes(
            dataset,
            xticklabels,
            colors,
            xlabel="Consensus cluster id",
            show_scatter=show_scatter,
            ax=ax,
            figsize=figsize,
            width=width,
            bw_method=bw_method,
            show_legend=show_legend,
            ylabel=ylabel,
            kde_side=kde_side,
            point_size=point_size,
            point_alpha=point_alpha,
            jitter_half_span=jitter_half_span,
            seed=seed,
            tight_layout=tight_layout,
            **kwargs,
        )

    def cluster_shift_times_violins(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        show_scatter: bool = True,
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        width: float = 0.75,
        bw_method: float = 0.18,
        show_sum: bool = False,
        show_total: bool = True,
        total_color: str = "#666666",
        show_legend: bool = True,
        ylabel: Optional[str] = None,
        kde_side: Literal["left", "right"] = "right",
        point_size: float = 15.0,
        point_alpha: float = 0.75,
        jitter_half_span: Optional[float] = None,
        seed: Optional[int] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> Tuple[FigureBase | None, Axes]:
        """Transition-time event samples for a **single** 3D cluster label field (not consensus).

        Uses the same per-voxel time convention as :meth:`consensus_shift_times_violins` but
        for one clustering result (native cluster ids in that map). Resolves *var* with
        :meth:`TOAD.get_clusters` (base or cluster name).

        For consensus timing restricted to one input model’s voxels, prefer
        :meth:`consensus_shift_times_violins` with ``source_input_cluster_var=...`` instead.
        """
        if show_scatter and not _VIOLIN_SIDE_SUPPORTED:
            raise RuntimeError(
                "cluster_shift_times_violins(..., show_scatter=True) requires matplotlib>=3.10."
            )

        var = self.td._get_base_var_if_none(var)
        cname = str(self.td.get_clusters(var).name)
        dists = self.td.aggregate.label_shift_time_distributions(cname)
        dataset, xticklabels, colors = self._distributions_to_violin_tuples(
            dists,
            cluster_ids,
            cmap=cmap,
            show_sum=show_sum,
            show_total=show_total,
            total_color=total_color,
            pad_empty_for_violin=True,
        )
        return self._plot_transition_time_violin_axes(
            dataset,
            xticklabels,
            colors,
            xlabel="cluster id",
            show_scatter=show_scatter,
            ax=ax,
            figsize=figsize,
            width=width,
            bw_method=bw_method,
            show_legend=show_legend,
            ylabel=ylabel,
            kde_side=kde_side,
            point_size=point_size,
            point_alpha=point_alpha,
            jitter_half_span=jitter_half_span,
            seed=seed,
            tight_layout=tight_layout,
            **kwargs,
        )

    def consensus_shift_times_medians(
        self,
        consensus_var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        *,
        spread: Literal["iqr", "std"] = "std",
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        model_point_size: float = 18.0,
        model_point_alpha: float = 0.5,
        model_point_color: str = "#1a1a1a",
        spread_line_lw: float = 4.0,
        spread_line_color: str = "#0d0d0d",
        errorbar_elinewidth: float = 1.75,
        errorbar_capsize: float = 3.5,
        errorbar_capthick: float = 1.2,
        summary_marker: Literal["D", "o"] = "D",
        summary_marker_size: float = 65.0,
        summary_cluster_cmap: Optional[Union[str, ListedColormap]] = default_cmap,
        categorical_cluster_axis: Optional[bool] = None,
        jitter_half_span: float = 0.18,
        seed: Optional[int] = None,
        show_legend: bool = True,
        ylabel: Optional[str] = None,
        tight_layout: bool = True,
    ) -> Tuple[FigureBase | None, Axes]:
        """Per-input median transition time vs median-of-medians summary (from :meth:`consensus_summary`).

        One point per input ``cluster_var`` at that map’s spatial median transition time in the cluster;
        the large marker is ``median_median_shift_time``. Inter-model spread defaults to
        **symmetric error bars** at ``median_median_shift_time ± std_median_shift_time`` (table
        columns) when those table values are finite. If not, the same IQR line as
        ``spread=\"iqr\"`` (quartiles of per-input medians) is used. With ``spread=\"iqr\"``,
        spread is always that vertical segment between the 25th and 75th percentiles of model
        medians.

        Args:
            consensus_var: Consensus labels variable; inferred if unique.
            cluster_ids: Subset of consensus cluster ids; default plots every id present in the
                shift-time dataset.
            spread: ``\"std\"`` (default) — ``errorbar`` from table when possible, else IQR of
                per-input medians; ``\"iqr\"`` — always a thick vertical line between quartiles of
                model medians.
            ax, figsize: Axis or new figure size.
            model_point_size, model_point_alpha, model_point_color: Per-model scatter.
            spread_line_lw, spread_line_color: IQR / fallback line width and colour
                (``spread=\"iqr\"`` or std fallback when table std is missing).
            errorbar_elinewidth, errorbar_capsize, errorbar_capthick: ``errorbar`` styling
                (``spread=\"std\"`` only).
            summary_marker, summary_marker_size: Summary median marker.
            summary_cluster_cmap: Fill colour of each summary diamond from this colormap (same
                sampling as :meth:`consensus_map`). Defaults to :data:`default_cmap` (``tab20b``);
                pass ``None`` for white faces with a black edge.
            categorical_cluster_axis: If True, x positions are ``0..n-1`` with tick labels showing
                cluster ids (useful for sparse ids like 1,4,6). If False, x matches cluster id
                numerically. If ``None``, use categorical mode when cluster ids are not a contiguous
                block (so there are no large empty gaps between ticks).
            jitter_half_span: Half-width of horizontal jitter for model points (x data units).
            seed: RNG seed for jitter.
            show_legend: Whether to add a short legend.
            ylabel: Y-axis label; default :meth:`_transition_time_ylabel`.
            tight_layout: If False, skip ``Figure.tight_layout`` (for embedding in a multi-panel
                figure; call :meth:`consensus_overview` instead).
        """
        consensus_var = self.td._resolve_consensus_var(consensus_var)
        da = self.td.data[consensus_var]
        dist_ds, _ = self.td.aggregate.consensus_shift_time_distribution(
            da,
        )
        if (
            len(dist_ds.data_vars) == 0
            or "spatial_median_transition_time" not in dist_ds
        ):
            raise ValueError(
                "No per-model spatial median shift times available; check consensus labels "
                "and input cluster variables."
            )

        sm = dist_ds["spatial_median_transition_time"]
        ids_all = np.asarray(sm.coords["consensus_cluster_id"].values, dtype=np.int64)
        if cluster_ids is not None:
            if isinstance(cluster_ids, int):
                wanted = {int(cluster_ids)}
            else:
                wanted = {int(x) for x in cluster_ids}
            plot_ids = np.array(
                [i for i in ids_all if int(i) in wanted], dtype=np.int64
            )
        else:
            plot_ids = ids_all.copy()
        if plot_ids.size == 0:
            raise ValueError("No consensus cluster ids left after filtering.")

        plot_ids.sort()

        def _ids_contiguous(p: np.ndarray) -> bool:
            if p.size <= 1:
                return True
            return bool(p.size == int(p[-1] - p[0] + 1) and np.all(np.diff(p) == 1))

        if categorical_cluster_axis is None:
            categorical_x = not _ids_contiguous(plot_ids)
        else:
            categorical_x = categorical_cluster_axis

        summary_face_colors: list[Any] | None = None
        if summary_cluster_cmap is not None:
            summary_face_colors = _discrete_colors_from_cmap(
                summary_cluster_cmap, len(plot_ids)
            )

        summary_df = self.td.aggregate.consensus_summary(consensus_var)
        summary_by = summary_df.set_index("cluster_id")

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize or (max(8, 1.1 * len(plot_ids)), 5.5)
            )
        else:
            fig = ax.get_figure()

        rng = np.random.default_rng(seed)

        for j, cid in enumerate(plot_ids):
            cid_i = int(cid)
            try:
                row = summary_by.loc[cid_i]
            except KeyError:
                continue
            median_tab = float(row["median_median_shift_time"])
            std_tab = float(row["std_median_shift_time"])

            vals = np.asarray(
                sm.sel(consensus_cluster_id=float(cid_i)).values,
                dtype=np.float64,
            ).ravel()
            fin = vals[np.isfinite(vals)]
            if fin.size == 0:
                continue

            x_base = float(j) if categorical_x else float(cid_i)
            for yv in fin:
                xj = x_base + rng.uniform(-jitter_half_span, jitter_half_span)
                ax.scatter(
                    xj,
                    yv,
                    s=model_point_size,
                    c=model_point_color,
                    alpha=model_point_alpha,
                    edgecolors="#000000",
                    linewidths=0.25,
                    zorder=5,
                )

            if spread == "std" and np.isfinite(std_tab) and np.isfinite(median_tab):
                ax.errorbar(
                    x_base,
                    median_tab,
                    yerr=float(std_tab),
                    fmt="none",
                    ecolor=spread_line_color,
                    elinewidth=errorbar_elinewidth,
                    capsize=errorbar_capsize,
                    capthick=errorbar_capthick,
                    zorder=3,
                )
            elif spread == "iqr" or (
                spread == "std"
                and (not np.isfinite(std_tab) or not np.isfinite(median_tab))
            ):
                # IQR of per-input medians (fallback when std mode cannot use table std/median)
                q1, q3 = np.percentile(fin, [25.0, 75.0])
                y_lo, y_hi = float(q1), float(q3)
                ax.plot(
                    [x_base, x_base],
                    [y_lo, y_hi],
                    color=spread_line_color,
                    linestyle="-",
                    linewidth=spread_line_lw,
                    solid_capstyle="round",
                    zorder=3,
                )

            if np.isfinite(median_tab):
                face = (
                    summary_face_colors[j]
                    if summary_face_colors is not None
                    else _VIOLIN_CONSENSUS_MEDIAN_FACE
                )
                ax.scatter(
                    [x_base],
                    [median_tab],
                    s=summary_marker_size,
                    c=face,
                    edgecolors="#000000",
                    linewidths=1.2,
                    marker=summary_marker,
                    zorder=8,
                )

        y_all: list[float] = []
        for cid in plot_ids:
            cid_i = int(cid)
            try:
                row = summary_by.loc[cid_i]
            except KeyError:
                continue
            vals = np.asarray(
                sm.sel(consensus_cluster_id=float(cid_i)).values,
                dtype=np.float64,
            ).ravel()
            fin = vals[np.isfinite(vals)]
            if fin.size:
                y_all.extend(fin.tolist())
            mt = float(row["median_median_shift_time"])
            if np.isfinite(mt):
                y_all.append(mt)
            if spread == "std":
                st = float(row["std_median_shift_time"])
                if np.isfinite(st) and np.isfinite(mt):
                    y_all.extend([mt - st, mt + st])
            elif fin.size:
                q1, q3 = np.percentile(fin, [25.0, 75.0])
                y_all.extend([float(q1), float(q3)])
        if y_all:
            ya = np.asarray(y_all, dtype=np.float64)
            y_span = float(np.ptp(ya))
            pad = 0.05 * (y_span if y_span > 0 else 1.0)
            ax.set_ylim(float(ya.min()) - pad, float(ya.max()) + pad)

        if categorical_x:
            ax.set_xticks(np.arange(len(plot_ids), dtype=float))
            ax.set_xticklabels([str(int(x)) for x in plot_ids])
        else:
            ax.set_xticks(plot_ids.astype(float))
            ax.set_xticklabels([str(int(x)) for x in plot_ids])
        ax.set_xlabel("Consensus cluster id")
        ax.set_ylabel(ylabel if ylabel is not None else self._transition_time_ylabel())
        # ax.set_title(
        #     f"Model median shift times — {consensus_var}",
        #     loc="left",
        # )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.grid(True, axis="y", color="#d9d9d9", linestyle="-", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        x_margin = jitter_half_span + 0.35
        if categorical_x:
            n_cat = max(len(plot_ids), 1)
            ax.set_xlim(-0.5 - x_margin, float(n_cat - 1) + 0.5 + x_margin)
        else:
            ax.set_xlim(
                float(plot_ids.min()) - x_margin,
                float(plot_ids.max()) + x_margin,
            )

        if show_legend:
            leg: list[Any] = [
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="o",
                    markerfacecolor=model_point_color,
                    markeredgecolor="#000000",
                    markersize=4.5,
                    markeredgewidth=0.3,
                    alpha=model_point_alpha,
                    label="Model median",
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=summary_marker,
                    markerfacecolor="none",
                    markeredgecolor="#000000",
                    markersize=8,
                    markeredgewidth=1.0,
                    label="Median across models",
                ),
            ]
            if spread == "iqr":
                spread_label = "Inter-model IQR"
                leg.append(
                    Line2D(
                        [0],
                        [0],
                        color=spread_line_color,
                        linestyle="-",
                        linewidth=spread_line_lw,
                        solid_capstyle="round",
                        label=spread_label,
                    ),
                )
            else:
                spread_label = "SD of model medians (±1 SD)"
                leg.append(
                    Line2D(
                        [0],
                        [0],
                        color=spread_line_color,
                        linestyle="-",
                        linewidth=max(errorbar_elinewidth, 1.5),
                        solid_capstyle="round",
                        marker="",
                        label=spread_label,
                    ),
                )
            ax.legend(
                handles=leg,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderaxespad=0.0,
                frameon=False,
                fontsize=9,
            )
            if tight_layout:
                _maybe_tight_layout(fig, rect=(0.0, 0.0, 1.0, 0.92))
        elif tight_layout:
            _maybe_tight_layout(fig)

        return fig, ax

    def consensus_overview(
        self,
        consensus_var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        *,
        kind: Literal["medians", "violins"] = "medians",
        spread: Literal["iqr", "std"] = "std",
        figsize: Optional[Tuple[float, float]] = None,
        width_ratios: Tuple[float, float] = (1.25, 1.0),
        wspace: float = 0.28,
        cmap: Union[str, ListedColormap] = default_cmap,
        colorbar_shrink: float = 0.38,
        colorbar_pad: float = 0.025,
        colorbar_aspect: float = 28.0,
        colorbar_label: Optional[str] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        show_legend: bool = True,
        ylabel: Optional[str] = None,
        seed: Optional[int] = None,
        show_sum: bool = False,
        show_total: bool = True,
        total_color: str = "#666666",
        bw_method: float = 0.18,
        **kwargs: Any,
    ) -> Tuple[FigureBase, Any, Axes]:
        """Two-panel figure: consensus map (left) and shift-time view (right).

        Left: :meth:`consensus_map` on the same ``cluster_ids`` as the right panel (defaults to
        every id with data for the chosen kind). Right: :meth:`consensus_shift_times_medians` when
        ``kind=\"medians\"`` (per-input medians and median-of-medians) or
        :meth:`consensus_shift_times_violins` when ``kind=\"violins\"`` (pooled sample violins).

        Args:
            consensus_var: Consensus labels variable; inferred if unique.
            cluster_ids: Subset of consensus cluster ids. Same filtering as the corresponding
                shift-time plot; default is every id present in the shift data for ``kind``.
            kind: ``\"medians\"`` (default) or ``\"violins\"`` for the right-hand panel.
            spread: Median-plot inter-model spread (``\"iqr\"`` or ``\"std\"``). Ignored when
                ``kind=\"violins\"``.
            show_legend, ylabel, seed: Forwarded to the shift-time panel.
            show_sum, show_total, total_color: Used when ``kind=\"violins\"`` only; passed to
                :meth:`consensus_shift_times_violins` (pooled columns).
            bw_method: Violin KDE bandwidth (``Axes.violinplot``); ``kind=\"violins\"`` only.
            figsize: Overall figure size; default ``(12, 5.2)``.
            width_ratios: ``GridSpec`` column width ratios (map, right panel).
            wspace: Spacing between panels (``Figure.subplots_adjust``).
            cmap, colorbar_shrink, colorbar_pad, colorbar_aspect, colorbar_label, map_style:
                Map panel (see :meth:`consensus_map` for ``colorbar_label``). The medians panel
                also uses the same ``cmap`` for summary diamond fill unless you pass
                ``summary_cluster_cmap`` in ``**kwargs`` (``None`` disables colouring). The
                violins panel uses ``cmap`` for per-cluster colours.
            **kwargs: Extra args for the active shift-time function (medians- or violins-specific;
                e.g. ``model_point_size`` / ``errorbar_capsize``, or ``show_scatter`` / ``width``).
                ``ax`` and ``figsize`` are ignored.

        Returns:
            ``(figure, map_axes, shift_axes)``. The map axes may be a cartopy ``GeoAxes``; the third
            element is the right-hand (medians or violins) axes.
        """
        if kind not in ("medians", "violins"):
            raise ValueError(
                f"consensus_overview: unknown kind {kind!r}; expected 'medians' or 'violins'."
            )

        consensus_var_resolved = self.td._resolve_consensus_var(consensus_var)
        da = self.td.data[consensus_var_resolved]
        if kind == "medians":
            dist_ds, _ = self.td.aggregate.consensus_shift_time_distribution(
                da,
            )
            if (
                len(dist_ds.data_vars) == 0
                or "spatial_median_transition_time" not in dist_ds
            ):
                raise ValueError(
                    "No per-model spatial median shift times available; check consensus labels "
                    "and input cluster variables."
                )
            sm = dist_ds["spatial_median_transition_time"]
            ids_all = np.asarray(
                sm.coords["consensus_cluster_id"].values, dtype=np.int64
            )
        else:
            dists = self.td.aggregate.consensus_shift_time_distributions(
                da,
            )
            if not dists:
                raise ValueError(
                    "No transition-time samples for violin plot; check consensus labels "
                    "and input cluster variables."
                )
            ids_all = np.array(sorted(dists.keys()), dtype=np.int64)

        if cluster_ids is not None:
            if isinstance(cluster_ids, int):
                wanted = {int(cluster_ids)}
            else:
                wanted = {int(x) for x in cluster_ids}
            plot_ids = np.array(
                [i for i in ids_all if int(i) in wanted], dtype=np.int64
            )
        else:
            plot_ids = ids_all.copy()
        if plot_ids.size == 0:
            raise ValueError("No consensus cluster ids left after filtering.")
        plot_ids.sort()

        _figsize = figsize if figsize is not None else (12.0, 5.2)
        config = _normalize_map_style(map_style)
        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None
        if config.projection is None:
            projection_obj = get_projection("plate_carree") if has_latlon else None
        else:
            projection_obj = get_projection(config.projection)
        if projection_obj is None:
            raise ValueError(
                "consensus_overview needs a geographic map on the "
                "left: ensure the dataset has latitude/longitude coordinates, or pass "
                "map_style with a projection (see :meth:`map`)."
            )

        fig = plt.figure(figsize=_figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios))
        ax_map = cast(GeoAxes, fig.add_subplot(gs[0, 0], projection=projection_obj))
        ax_right = fig.add_subplot(gs[0, 1])

        _add_map_features(ax_map, config)
        if config.extent is None:
            if ax_map.projection == ccrs.SouthPolarStereo():
                ax_map.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
            elif ax_map.projection == ccrs.NorthPolarStereo():
                ax_map.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
        else:
            ax_map.set_extent(config.extent, crs=ccrs.PlateCarree())
        ax_map.set_frame_on(config.map_frame)

        map_ids = [int(x) for x in plot_ids.tolist()]
        map_out = self.consensus_map(
            consensus_var,
            cluster_ids=map_ids,
            ax=ax_map,
            cmap=cmap,
            colorbar_shrink=colorbar_shrink,
            colorbar_pad=colorbar_pad,
            colorbar_aspect=colorbar_aspect,
            colorbar_label=colorbar_label,
            map_style=map_style,
        )
        if map_out[1] is None:
            plt.close(fig)
            raise ValueError(
                "consensus_map produced no axes; check consensus cluster ids against the dataset."
            )

        if kind == "medians":
            median_kw = dict(kwargs)
            median_kw.pop("ax", None)
            median_kw.pop("figsize", None)
            if "summary_cluster_cmap" not in median_kw:
                median_kw["summary_cluster_cmap"] = cmap

            self.consensus_shift_times_medians(
                consensus_var=consensus_var,
                cluster_ids=cluster_ids,
                spread=spread,
                ax=ax_right,
                show_legend=show_legend,
                ylabel=ylabel,
                seed=seed,
                tight_layout=False,
                **median_kw,
            )
        else:
            violin_kw = dict(kwargs)
            violin_kw.pop("ax", None)
            violin_kw.pop("figsize", None)
            for _k in ("show_sum", "show_total", "total_color", "bw_method"):
                violin_kw.pop(_k, None)
            violin_cmap = violin_kw.pop("cmap", cmap)
            self.consensus_shift_times_violins(
                consensus_var=consensus_var,
                cluster_ids=cluster_ids,
                ax=ax_right,
                cmap=violin_cmap,
                show_legend=show_legend,
                ylabel=ylabel,
                seed=seed,
                show_sum=show_sum,
                show_total=show_total,
                total_color=total_color,
                bw_method=bw_method,
                tight_layout=False,
                **violin_kw,
            )

        fig.subplots_adjust(wspace=wspace)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94 if show_legend else 1.0))

        return fig, ax_map, ax_right

    def _plot_consensus_raw_shift_indicators(
        self,
        target_ax: Axes,
        ts: Any,
        cname: str,
        consensus_da: Any,
        consensus_cluster_id: int,
        *,
        color: Any,
        alpha: float,
        shift_indicator_size: float,
    ) -> None:
        """Dots where each cell lies in a native cluster window (same idea as :meth:`timeseries`).

        Per input ``cluster_var`` we use its ``BASE_VARIABLE`` with
        ``get_clusters(...).where(get_cluster_mask(..., union of ids on footprint))``, stacked on
        the same extraction mask as ``ts``. Only for raw ``cell_xy`` trajectories.
        """
        time_dim = self.td.time_dim
        if "cell_xy" not in ts.dims or int(ts.sizes.get("cell_xy", 0)) == 0:
            return
        base_var = ts.attrs.get("base_var")
        if base_var is None:
            return
        mask2d = self.td.aggregate.consensus_extraction_mask_2d(
            consensus_da, consensus_cluster_id, cname
        )
        tslice: slice | None = None
        if not ts.attrs.get("keep_full_timeseries", True):
            cm = consensus_da == consensus_cluster_id
            active = np.flatnonzero(cm.any(dim=tuple(self.td.space_dims)).values)
            if active.size:
                tslice = slice(int(active[0]), int(active[-1]) + 1)

        lab_region = self.td.data[cname].where(mask2d)
        if tslice is not None:
            lab_region = lab_region.isel({time_dim: tslice})

        vals = np.asarray(lab_region.values, dtype=np.float64).ravel()
        native_ids = np.unique(vals[np.isfinite(vals) & (vals >= 0)]).astype(int)
        allowed = set(np.asarray(self.td.get_cluster_ids(base_var), dtype=int).tolist())
        valid_native = [int(i) for i in native_ids.tolist() if int(i) in allowed]
        if not valid_native:
            return

        try:
            cl_da = self.td.get_clusters(base_var)
        except ValueError:
            return

        cl_masked = cl_da.where(self.td.get_cluster_mask(base_var, valid_native)).where(
            mask2d
        )
        if tslice is not None:
            cl_masked = cl_masked.isel({time_dim: tslice})

        det_stacked = self.td._aggregate_spatial(cl_masked, "raw")
        try:
            det_stacked = det_stacked.reindex_like(ts)
        except ValueError:
            pass
        if det_stacked.sizes.get("cell_xy") != ts.sizes.get("cell_xy"):
            return

        n_cell = int(ts.sizes["cell_xy"])
        for j in range(n_cell):
            ts_cell = ts.isel(cell_xy=j)
            det_cell = det_stacked.isel(cell_xy=j)
            tcoord = ts_cell.coords[time_dim]
            valid = np.isfinite(np.asarray(det_cell.values, dtype=float))
            if not np.any(valid):
                continue
            xvals = np.asarray(tcoord.values)[valid]
            yvals = np.asarray(ts_cell.values, dtype=float)[valid]
            if xvals.size == 0:
                continue
            target_ax.plot(
                xvals,
                yvals,
                marker="o",
                linestyle="none",
                color=color,
                alpha=alpha,
                markersize=shift_indicator_size,
                zorder=5,
            )

    def consensus_timeseries(
        self,
        cluster_id: int,
        consensus_var: str | None = None,
        *,
        var: str | None = None,
        cluster_vars: Optional[List[str]] = None,
        aggregation: Literal[
            "raw", "mean", "sum", "std", "median", "percentile", "max", "min"
        ]
        | str = "raw",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["max", "max_each"]] | str = None,
        keep_full_timeseries: bool = True,
        ax: Optional[Axes] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        alpha: float = 1.0,
        linewidth: float = 1.0,
        add_legend: bool = True,
        legend_input_label: Literal[
            "cluster_var", "member_id", "realisation"
        ] = "cluster_var",
        legend_include_n_cells: bool = True,
        legend_autosize: bool = True,
        legend_fontsize_max: Optional[float] = None,
        legend_fontsize_min: float = 4.0,
        subplots: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        show_ylabels: bool = False,
        plot_shift_indicator: bool = False,
        shift_indicator_size: float = 5.0,
    ) -> Tuple[FigureBase | None, Optional[Union[Axes, np.ndarray]]]:
        """Overlay per-input-cluster timeseries for one consensus cluster (no map).

        Wraps :meth:`toad.postprocessing.Aggregation.consensus_cluster_timeseries`. When
        ``subplots=False``, all input cluster trajectories are drawn on one axis; when
        ``subplots=True``, one axis per input ``cluster_var`` (shared time axis). Legend
        entries and subplot titles are prefixed with ``(N)``, where ``N`` is the number of
        grid cells in the 2D extraction mask
        (:meth:`toad.postprocessing.Aggregation.consensus_extraction_mask_2d`).

        Args:
            cluster_id: Consensus cluster id to extract.
            consensus_var: Consensus labels variable; inferred if unique.
            var, cluster_vars, aggregation, percentile, normalize, keep_full_timeseries:
                Forwarded to ``consensus_cluster_timeseries``.
            ax: Single axis; used only when ``subplots=False``. Ignored when ``subplots=True``.
            cmap: Colormap for line colors (tab-like sampling).
            alpha, linewidth, add_legend: Line styling. With ``aggregation=\"raw\"``, one legend
                entry per input ``cluster_var`` (all per-cell lines share colour and label).
            legend_input_label: ``\"cluster_var\"`` (default) uses the full input variable name;
                ``\"member_id\"`` uses a CMIP-style member id parsed from the name (e.g.
                ``r1i1p1f1`` from ``mlotst_r1i1p1f1_dts_cluster``); ``\"realisation\"`` uses only
                the realisation index (e.g. ``r1``, ``r2``).
            legend_include_n_cells: If True (default), prefix each legend entry with ``(N)`` where
                ``N`` is the number of grid cells in the extraction mask.
            legend_autosize: If True (default) and ``subplots=False``, shrink legend font size until
                the legend box fits inside the axes (helps many/long labels).
            legend_fontsize_max, legend_fontsize_min: Bounds for autosizing (points); max defaults to
                :rc:`legend.fontsize`.
            subplots: If True, one subplot per contributing input clustering variable.
            figsize: Used when creating a figure.
            show_ylabels: If True, set a default y label on each subplot when ``subplots=True``.
            plot_shift_indicator: If True (and ``aggregation=\"raw\"`` with per-cell ``cell_xy``
                data), overlay dots at timesteps where each cell falls inside a **native** cluster
                window for that input ``cluster_var`` (union of native ids on the consensus
                extraction footprint; uses the same masking idea as :meth:`timeseries`).
                Ignored for aggregated trajectories (single line per input).
            shift_indicator_size: Marker size in points for shift dots.
        """
        consensus_var = self.td._resolve_consensus_var(consensus_var)
        da = self.td.data[consensus_var]
        try:
            series_by_input = self.td.aggregate.consensus_cluster_timeseries(
                da,
                cluster_id,
                var=var,
                cluster_vars=cluster_vars,
                aggregation=aggregation,
                percentile=percentile,
                normalize=normalize,
                keep_full_timeseries=keep_full_timeseries,
            )
        except ValueError as e:
            raise ValueError(str(e)) from e

        if not series_by_input:
            raise ValueError(
                "No input cluster timeseries returned; check consensus_cluster_id and masks."
            )

        if plot_shift_indicator and aggregation != "raw":
            logger.warning(
                "consensus_timeseries: plot_shift_indicator only applies when aggregation='raw' "
                "(per-cell trajectories); ignoring."
            )

        n_series = len(series_by_input)
        colors = _discrete_colors_from_cmap(cmap, n_series)

        time_dim = self.td.time_dim
        n_cells_by_input = {
            cname: int(
                self.td.aggregate.consensus_extraction_mask_2d(da, cluster_id, cname)
                .sum()
                .item()
            )
            for cname in series_by_input
        }

        def _label_for_input(cname: str) -> str:
            return _input_cluster_legend_label(
                cname,
                n_cells=n_cells_by_input.get(cname, 0),
                label_style=legend_input_label,
                include_n_cells=legend_include_n_cells,
            )

        def _plot_cluster_ts_lines(
            target_ax: Axes,
            ts: Any,
            color: Any,
            *,
            legend_label: str | None,
        ) -> None:
            """Plot ``ts`` on ``target_ax``. For ``aggregation == \"raw\"`` with a ``cell_xy``
            dimension, draw one line per cell; only the first line carries ``legend_label`` so
            the legend has one entry per input ``cluster_var``.
            """
            if (
                aggregation == "raw"
                and "cell_xy" in ts.dims
                and ts.sizes.get("cell_xy", 0) > 0
            ):
                n_cell = int(ts.sizes["cell_xy"])
                for j in range(n_cell):
                    lbl = (
                        legend_label
                        if j == 0 and legend_label is not None
                        else "_nolegend_"
                    )
                    ts.isel(cell_xy=j).plot.line(
                        x=time_dim,
                        ax=target_ax,
                        add_legend=False,
                        alpha=alpha,
                        lw=linewidth,
                        label=lbl,
                        color=color,
                    )
            else:
                ts.plot.line(
                    x=time_dim,
                    ax=target_ax,
                    add_legend=False,
                    alpha=alpha,
                    lw=linewidth,
                    label=legend_label,
                    color=color,
                )

        if subplots:
            fig, ax_arr = plt.subplots(
                n_series, 1, sharex=True, figsize=figsize or (8, 2 * n_series)
            )
            ax_arr = np.atleast_1d(ax_arr)
            for i, (cname, ts) in enumerate(series_by_input.items()):
                _plot_cluster_ts_lines(
                    ax_arr[i],
                    ts,
                    colors[i % len(colors)],
                    legend_label=None,
                )
                if plot_shift_indicator and aggregation == "raw":
                    self._plot_consensus_raw_shift_indicators(
                        ax_arr[i],
                        ts,
                        cname,
                        da,
                        cluster_id,
                        color=colors[i % len(colors)],
                        alpha=alpha,
                        shift_indicator_size=shift_indicator_size,
                    )
                ax_arr[i].set_title(_label_for_input(cname))
                ax_arr[i].set_ylabel("" if not show_ylabels else str(ts.name or cname))
            fig.suptitle(
                f"{consensus_var} cluster {cluster_id} — per input clustering",
                y=1.02,
            )
            fig.tight_layout()
            return fig, ax_arr
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize or (8, 4))
            else:
                fig = ax.get_figure()
            for i, (cname, ts) in enumerate(series_by_input.items()):
                _plot_cluster_ts_lines(
                    ax,
                    ts,
                    colors[i % len(colors)],
                    legend_label=_label_for_input(cname),
                )
                if plot_shift_indicator and aggregation == "raw":
                    self._plot_consensus_raw_shift_indicators(
                        ax,
                        ts,
                        cname,
                        da,
                        cluster_id,
                        color=colors[i % len(colors)],
                        alpha=alpha,
                        shift_indicator_size=shift_indicator_size,
                    )
            if add_legend:
                if legend_autosize:
                    _legend_shrink_to_fit_axes(
                        ax,
                        loc="best",
                        fontsize_max=legend_fontsize_max,
                        fontsize_min=legend_fontsize_min,
                    )
                else:
                    ax.legend(loc="best")
            ax.set_title(f"{consensus_var} cluster {cluster_id}")
            return fig, ax

    def max_shift_map(
        self,
        var: str | None = None,
        *,
        cluster_ids: int | list[int] | range | None = None,
        ax: Optional[Axes] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        cmap: Optional[Union[str, Colormap]] = "RdBu_r",
        cbar_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Plot a map showing the value in the time dimension where the absolute value of the shift is maximal, keeping sign.

        Args:
            var: Name of the variable for which to compute the maximum shift.
                If None, TOAD will attempt to infer which variable to use. A ValueError is raised
                if the variable cannot be uniquely determined.
            cluster_ids: Optional integer or list of integers specifying which cluster IDs to analyze.
                If None, analyzes all clusters. If specified, only analyzes grid cells belonging
                to the given cluster(s).
            ax: Matplotlib axes to plot on. Creates new figure if None.
            map_style: Configuration for the map style.
                Can be a MapStyle instance or a dictionary containing style settings. Defaults to None.
            cmap: Colormap to use for the plot. Can be a string name of a colormap
                recognized by matplotlib, or an actual Colormap object. Defaults to 'RdBu_r'.
            cbar_kwargs: Optional dictionary of keyword arguments to pass to the colorbar.
                If not provided, defaults to {'label': 'Maximum shift magnitude'}.

        Returns:
            Tuple[FigureBase | None, matplotlib.axes.Axes]:
                The created matplotlib Figure (None if ax was provided) and Axes objects.

        Notes:
            For each location, this plots the value along the time axis whose absolute value is maximal. Locations with all-NaN
            values will be masked out.
        """
        # Infer variable if not provided
        var = self.td._get_base_var_if_none(var)

        # Normalize map_style to MapStyle
        config = _normalize_map_style(map_style)

        # Create map if ax not provided
        if ax is None:
            fig, ax = self.map(map_style=config)
        else:
            fig = None

        shifts = self.td.get_shifts(var)

        # set vmin, vmax to (-1, 1) if values are generally in that range
        if np.all(np.abs(shifts) <= 1):
            vmax = 1
            vmin = -1
        else:
            vmax = None
            vmin = None

        # Handle cbar_kwargs and allow user override
        default_cbar_kwargs = {"label": "Maximum shift magnitude"}
        if cbar_kwargs is not None:
            merged_cbar_kwargs = {**default_cbar_kwargs, **cbar_kwargs}
        else:
            merged_cbar_kwargs = default_cbar_kwargs

        # Prepare plot parameters for different grid types
        plot_params = {
            "ax": ax,
            "add_colorbar": True,
            "vmax": vmax,
            "vmin": vmin,
            "cmap": cmap,
            "cbar_kwargs": merged_cbar_kwargs,
            **kwargs,
        }

        plot_params, use_pcolormesh = self._prepare_map_plot_params(ax, plot_params)

        # Find the shift with largest magnitude (max abs value), keeping the original sign
        abs_shifts = abs(shifts)
        # Fill NaN with -inf so argmax ignores them (won't affect valid data since we're finding max)
        # This prevents ValueError for all-NaN slices
        abs_argmax = abs_shifts.fillna(float("-inf")).argmax(dim=self.td.time_dim)
        # Select from original shifts (with sign) at max indices, masking all-NaN locations
        has_valid_data = ~abs_shifts.isnull().all(dim=self.td.time_dim)
        shifts_max = shifts.isel({self.td.time_dim: abs_argmax}).where(has_valid_data)

        # Apply cluster mask if cluster_ids specified
        if cluster_ids is not None:
            cluster_mask = self.td.get_cluster_mask_spatial(var, cluster_ids)
            shifts_max = shifts_max.where(cluster_mask)

        if use_pcolormesh:
            # Use pcolormesh explicitly for regular axes to ensure proper coordinate handling
            shifts_max.plot.pcolormesh(**plot_params)
        else:
            shifts_max.plot(**plot_params)

        ax.set_title(f"Maximum shift magnitude for {var}")

        return fig, ax

    def time_of_max_shift_map(
        self,
        var: str | None = None,
        *,
        cluster_ids: int | list[int] | range | None = None,
        ax: Optional[Axes] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        cmap: Optional[Union[str, Colormap]] = "turbo",
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
        shift_direction: Literal["both", "positive", "negative"] | str = "both",
        **kwargs: Any,
    ):
        """Plot a map showing the time at which the maximal shift occurs for a given variable.

        Args:
            var: Name of the variable for which to compute the time of maximum shift.
                If None, TOAD will attempt to infer which variable to use. A ValueError is raised
                if the variable cannot be uniquely determined.
            cluster_ids: Optional integer or list of integers specifying which cluster IDs to analyze.
                If None, analyzes all clusters. If specified, only analyzes grid cells belonging
                to the given cluster(s).
            ax: Matplotlib axes to plot on. Creates new figure if None.
            map_style: Configuration for the map style.
                Can be a MapStyle instance or a dictionary containing style settings. Defaults to None.
            cmap: Colormap to use for the plot. Can be a string name of a colormap
                recognized by matplotlib, or an actual Colormap object. Defaults to 'turbo'.
            shift_threshold: Threshold value for shift magnitude above which a transition
                is detected. This value is passed to `compute_transition_time`. Defaults to 0.5.
            shift_direction: Sign of shifts to map (``"both"``, ``"positive"``, or ``"negative"``).
                Passed to :meth:`TimeStats.compute_transition_time`. Defaults to ``"both"``.

        Returns:
            Tuple[FigureBase | None, matplotlib.axes.Axes]:
                The created matplotlib Figure (None if ax was provided) and Axes objects.
        """
        # Infer variable if not provided
        var = self.td._get_base_var_if_none(var)

        # Normalize map_style to MapStyle
        config = _normalize_map_style(map_style)

        # Create map if ax not provided
        if ax is None:
            fig, ax = self.map(map_style=config)
        else:
            fig = None

        transition_time = self.td.stats(var).time.compute_transition_time(
            cluster_ids=cluster_ids,
            shift_threshold=shift_threshold,
            shift_direction=shift_direction,
        )

        # Prepare plot parameters for different grid types
        plot_params = {
            "ax": ax,
            "add_colorbar": True,
            "cmap": cmap,
            **kwargs,
        }

        plot_params, use_pcolormesh = self._prepare_map_plot_params(ax, plot_params)

        # Use appropriate plotting method based on grid type
        if use_pcolormesh:
            # Use pcolormesh explicitly for regular axes to ensure proper coordinate handling
            transition_time.plot.pcolormesh(**plot_params)
        else:
            transition_time.plot(**plot_params)

        ax.set_title(f"Time of maximum shift for {var}")

        return fig, ax

    def cluster_occurrence_rate_map(
        self,
        *,
        cluster_vars: list[str] | None = None,
        ax: Optional[Axes] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        cmap: Optional[Union[str, Colormap]] = "cividis",
        **kwargs: Any,
    ):
        """Map the fraction of input clusterings that ever labelled each cell.

        Uses :meth:`toad.postprocessing.aggregation.Aggregation.cluster_occurrence_rate`.
        ``1`` means every included clustering assigned a non-noise label there at some
        time; ``0`` cells are masked out. Timing and cluster id are not compared across
        inputs (see that method for details).

        Args:
            cluster_vars: Label variables to aggregate; default all :attr:`TOAD.cluster_vars`.
            ax: Axes to draw on; if None, a new figure is created via :meth:`map`.
            map_style: Passed to :meth:`map` when *ax* is None.
            cmap: Colormap for the rate field.
            **kwargs: Extra arguments forwarded to :meth:`xarray.DataArray.plot` or
                ``plot.pcolormesh`` (e.g. ``vmin``, ``vmax``).

        Returns:
            Tuple[FigureBase | None, matplotlib.axes.Axes]:
            Figure and axes; figure is None if *ax* was provided.
        """
        config = _normalize_map_style(map_style)
        if ax is None:
            fig, ax = self.map(map_style=config)
        else:
            fig = None

        occ = self.td.aggregate.cluster_occurrence_rate(cluster_vars=cluster_vars)
        occ = occ.where(occ > 0)
        plot_params: dict[str, Any] = {
            "ax": ax,
            "add_colorbar": True,
            "cmap": cmap,
            "cbar_kwargs": {"label": "Cluster occurrence rate"},
            **kwargs,
        }
        plot_params, use_pcolormesh = self._prepare_map_plot_params(ax, plot_params)
        if use_pcolormesh:
            occ.plot.pcolormesh(**plot_params)
        else:
            occ.plot(**plot_params)
        ax.set_title("Cluster occurrence rate")
        return fig, ax

    def timeseries(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        *,
        timeseries_var: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        normalize: Optional[Literal["max", "max_each"]] | str = None,
        add_legend: bool = True,
        # Individual trajectories
        plot_trajectories: bool = True,
        max_trajectories: int = 1_000,
        trajectories_sample_seed: int = 0,
        trajectory_ids: Optional[List[int]] = None,
        trajectories_alpha: float = 0.5,
        trajectories_linewidth: float = 0.5,
        full_timeseries: bool = True,
        highlight_color: Optional[str] = None,
        highlight_alpha: float = 0.5,
        highlight_linewidth: float = 0.5,
        # Shift detection markers
        plot_shift_indicator: bool = False,
        shift_indicator_size: float = 5.0,
        plot_dts: bool = False,  # If True, plot shifts variable in timeseries
        # Aggregated statistics
        plot_median: bool = False,
        plot_mean: bool = False,
        median_linewidth: float = 3,
        mean_linewidth: float = 3,
        # Shaded regions
        plot_trajectory_range: bool = False,  # Full range (min to max)
        plot_trajectory_std: bool = False,  # 68% interquartile range (16th to 84th percentile)
        trajectory_range_alpha: float = 0.2,
        trajectory_std_alpha: float = 0.4,
        # Shift duration
        plot_cluster_duration: bool = True,
        cluster_duration_color: Optional[str] = None,  # Uses cluster color if None
        cluster_duration_alpha: float = 0.25,
        # Map options
        plot_map: bool = False,
        map_var: Optional[str] = None,
        map_cmap_other: Optional[Union[str, Colormap]] = default_cmap_other,
        map_include_all_clusters: bool = True,
        # Subplot layout
        subplots: bool = False,  # If True, create one subplot per cluster
        ncols: int = 1,  # Number of columns for subplot grid
        figsize: Optional[Tuple[float, float]] = None,
        wspace: float = 0.1,
        hspace: float = 0.1,
        show_ylabels: bool = False,  # Only relevant for subplots
        shared_ylabel: Optional[
            str
        ] = None,  # One label for subplot column; "auto" infers
        shared_ylabel_pad: float = 0.03,
        shared_ylabel_fontsize: Optional[float] = None,
        vertical: bool = False,  # Only relevant when plot_map=True
        width_ratios: Tuple[float, float] = (
            1.0,
            1.0,
        ),  # Only relevant when plot_map=True
        height_ratios: Optional[
            Tuple[float, float]
        ] = None,  # Only relevant when plot_map=True
        map_style: Optional[Union[MapStyle, dict]] = None,
        **plot_kwargs: Any,
    ) -> Tuple[FigureBase | None, Optional[Union[Axes, List[Axes], dict]]]:
        """Plot time series from clusters or all data.

        This function allows flexible plotting of individual trajectories, aggregated statistics
        (median/mean), shaded regions (full range and IQR), and shift duration indicators.
        If no clusters are provided, plots all timeseries from the dataset.

        Can optionally create separate subplots for each cluster, and optionally include a map
        showing cluster spatial locations alongside the timeseries.

        Args:
            var: Base variable or cluster variable. If None, TOAD will attempt
                to infer which variable to use. A ValueError is raised if the variable cannot be
                uniquely determined.
            cluster_ids: ID or list of IDs of clusters to plot. If None, plots all timeseries
                from the dataset (no clustering). Cannot be None if plot_map=True.
            timeseries_var: Variable name to plot (if different from var). Defaults to var.
            ax: Matplotlib axes to plot on. Creates new figure if None. Ignored if subplots=True
                or plot_map=True.
            color: Single color to use for all plotted clusters. Overrides cmap.
            cmap: Colormap to use if plotting multiple clusters and color is None.
            normalize: Method to normalize timeseries ('max', 'max_each'). Defaults to None.
            add_legend: If True, add a legend indicating cluster IDs.
            plot_trajectories: If True, plot individual cell trajectories.
            max_trajectories: Maximum number of individual trajectories to plot (per cluster if
                clusters provided, or total if plotting all data). Ignored when trajectory_ids is set.
            trajectories_sample_seed: Seed for the random number generator used to sample trajectories. Defaults to 0.
            trajectory_ids: Exact integer indices of cells to plot. Overrides max_trajectories and
                trajectories_sample_seed. Out-of-range indices are silently skipped.
            trajectories_alpha: Alpha transparency for individual time series lines. Defaults to 0.5.
            trajectories_linewidth: Linewidth for individual time series lines. Defaults to 0.5.
            full_timeseries: If True, plot the full timeseries for each cell. If False,
                only plot the segment belonging to the cluster.
            highlight_color: Color to highlight the actual cluster segment
                when full_timeseries is True.
            highlight_alpha: Alpha for the cluster highlight segment.
            highlight_linewidth: Line width for the cluster highlight segment.
            plot_shift_indicator: If True, overlay a dot on each trajectory at every timestep where
                that cell is assigned to the cluster (i.e. where the shift is detected).
                Only applies when clusters are provided.
            shift_indicator_size: Marker size (in points) for the shift detection dots. Defaults to 5.0.
            plot_dts: If True, plot shifts variable in timeseries instead of base variable.
            plot_median: If True, plot the median timeseries curve.
            plot_mean: If True, plot the mean timeseries curve.
            median_linewidth: Linewidth for the median curve.
            mean_linewidth: Linewidth for the mean curve.
            plot_trajectory_range: If True, plot the full range (min to max) as a shaded area.
            plot_trajectory_std: If True, plot the 68% interquartile range (16th to 84th percentile) as a shaded area.
            trajectory_range_alpha: Alpha transparency for the full range shaded area.
            trajectory_std_alpha: Alpha transparency for the IQR shaded area.
            plot_cluster_duration: If True, adds horizontal shading indicating the cluster's
                temporal extent (start to end). Only applies when clusters are provided.
            cluster_duration_color: Color for shift duration shading. Uses cluster color if None.
            cluster_duration_alpha: Alpha for the shift duration shading.
            plot_map: If True, include a map showing cluster spatial locations alongside timeseries.
                Defaults to False.
            map_var: Variable name whose data to plot in the map. Defaults to var if None. Only used if plot_map=True.
            map_cmap_other: Colormap for remaining clusters on map. Only used if plot_map=True.
            map_include_all_clusters: If True, plot all clusters on the map. If False, only plot selected clusters.
                Only used if plot_map=True.
            subplots: If True, create one subplot per cluster. Defaults to False. If plot_map=True and multiple
                clusters, subplots are automatically enabled.
            ncols: Number of columns for subplot grid when subplots=True. Defaults to 1.
            figsize: Figure size (width, height) in inches. Used when subplots=True or plot_map=True.
            wspace: Width space between timeseries subplots (if ncols > 1).
            hspace: Height space between timeseries rows.
            show_ylabels: If True, show y-axis label on the timeseries plots. Only relevant for subplots.
            shared_ylabel: If set, draw one rotated y-label beside the timeseries column instead of
                per-panel labels. Pass ``"auto"`` to use the first panel's inferred label or
                variable metadata. Ignored when ``subplots=False``.
            shared_ylabel_pad: Horizontal gap between the shared label and the leftmost subplot
                (figure fraction).
            shared_ylabel_fontsize: Font size for the shared label; defaults to ``axes.labelsize``.
            vertical: If True, arrange map above timeseries plots. Otherwise, map is placed to the left.
                Only used if plot_map=True.
            width_ratios: Tuple of relative widths for map vs. timeseries section (used in horizontal layout).
                Only used if plot_map=True.
            height_ratios: Optional tuple of relative heights for map vs. timeseries section (used in vertical layout).
                Only used if plot_map=True.
            **plot_kwargs: Additional arguments passed to xarray.plot for each trajectory.

        Returns:
            Tuple of (figure, axes).
            - If plot_map=False and single plot: (figure, Axes)
            - If plot_map=False and subplots=True: (figure, List[Axes])
            - If plot_map=True: (figure, dict) with keys 'map' and 'timeseries'
            Figure is None if ax was provided and subplots=False and plot_map=False.

        Raises:
            ValueError: If no timeseries found for a given cluster ID, if nothing is set to plot,
                if cluster_ids is None when plot_map=True, or if plotting all data when plot_map=True.
        """
        # Validate plot_map requirements
        if plot_map:
            if cluster_ids is None:
                raise ValueError(
                    "cluster_ids cannot be None when plot_map=True. Provide at least one cluster ID."
                )
            if ax is not None:
                raise ValueError(
                    "Cannot use ax parameter when plot_map=True. Set ax=None when using plot_map."
                )

        # Parse cluster IDs
        cluster_ids_list, single_plot, plot_all_data = self._parse_cluster_ids(
            cluster_ids, var
        )
        var = self.td._get_base_var_if_none(var)

        # Check if we have any valid clusters to plot
        if not plot_all_data and len(cluster_ids_list) == 0:
            logger.warning(f"No valid clusters found in clusters for variable {var}")
            if not plot_map:
                return None, None
            return self._empty_cluster_map_timeseries(
                figsize=figsize,
                vertical=vertical,
                width_ratios=width_ratios,
                height_ratios=height_ratios,
                hspace=hspace,
                wspace=wspace,
                ncols=ncols,
                map_style=map_style,
            )

        if plot_map and plot_all_data:
            raise ValueError(
                "Cannot plot map when cluster_ids is None (plotting all data). "
                "Set plot_map=False or provide cluster_ids."
            )

        if plot_map and map_var is None:
            map_var = var

        # Infer plot variable (pass map=plot_map)
        timeseries_var = self._infer_plot_var(
            var, timeseries_var, plot_dts, map=plot_map
        )

        # Validate that something will be plotted
        has_individual = plot_trajectories
        has_aggregate = (
            plot_median or plot_mean or plot_trajectory_range or plot_trajectory_std
        )
        if not has_individual and not has_aggregate:
            raise ValueError(
                "Nothing to plot: set at least one of plot_trajectories, plot_median, "
                "plot_mean, plot_trajectory_range, or plot_trajectory_std to True."
            )

        # Validate ncols
        if ncols <= 0:
            raise ValueError(f"ncols must be > 0, got {ncols}")

        # Determine if we need subplots
        # When plot_map=True and multiple clusters, automatically enable subplots
        if plot_map and len(cluster_ids_list) > 1:
            use_subplots = True
        else:
            use_subplots = subplots

        # Setup figure and axes layout
        fig, ts_axes_list, map_ax = self._setup_timeseries_axes(
            map=plot_map,
            use_subplots=use_subplots,
            cluster_ids_list=cluster_ids_list,
            n_subplots_col=ncols,
            figsize=figsize,
            vertical=vertical if plot_map else False,
            width_ratios=width_ratios if plot_map else (1.0, 1.0),
            height_ratios=height_ratios if plot_map else None,
            hspace=hspace,
            wspace=wspace,
            ax=ax if not plot_map else None,
            map_style=map_style,
        )

        # Get colors for clusters
        colors = self._assign_cluster_colors(
            cluster_ids_list, color, cmap, map=plot_map, use_subplots=use_subplots
        )

        # Plot map if requested
        if plot_map:
            if map_ax is None:
                raise ValueError("map_ax should be set when plot_map=True")
            self._plot_timeseries_map(
                map_var=map_var,
                cluster_ids_list=cluster_ids_list,
                map_ax=map_ax,
                colors=colors,
                color=color,
                map_include_all_clusters=map_include_all_clusters,
                map_cmap_other=map_cmap_other,
                map_style=map_style,
                **plot_kwargs,
            )

        # Single unified loop for both all-data and clustered plotting
        y_label = ""
        for i, id in enumerate(cluster_ids_list):
            # Get the axes for this cluster
            current_ax = ts_axes_list[i] if use_subplots else ts_axes_list[0]

            # Get color for this cluster
            id_color = self._get_cluster_color(i, cluster_ids_list, color, colors, cmap)

            # Use cluster color for shift duration if not specified
            shift_color = (
                cluster_duration_color
                if cluster_duration_color is not None
                else id_color
            )

            # Plot aggregated statistics first (so they appear behind individual trajectories)
            if plot_trajectory_range:
                self._plot_trajectory_range_band(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    range_alpha=trajectory_range_alpha,
                    normalize=normalize,
                    time_dim=self.td.time_dim,
                    full_timeseries=full_timeseries,
                )

            if plot_trajectory_std:
                self._plot_iqr_band(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    iqr_alpha=trajectory_std_alpha,
                    normalize=normalize,
                    time_dim=self.td.time_dim,
                    full_timeseries=full_timeseries,
                )

            if plot_mean:
                self._plot_mean_curve(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    mean_linewidth=mean_linewidth,
                    add_legend=add_legend,
                    normalize=normalize,
                    full_timeseries=full_timeseries,
                )

            if plot_median:
                self._plot_median_curve(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    median_linewidth=median_linewidth,
                    add_legend=add_legend,
                    normalize=normalize,
                    full_timeseries=full_timeseries,
                )

            # Plot shift duration (horizontal shading) - only for real clusters.
            # Skip when full_timeseries=False: the entire visible plot is already the
            # cluster segment, so the indicator would cover the whole background.
            if plot_cluster_duration and id is not None and full_timeseries:
                self._plot_cluster_duration(
                    current_ax=current_ax,
                    var=var,
                    cluster_id=id,
                    shift_color=shift_color,
                    cluster_duration_alpha=cluster_duration_alpha,
                )

            # Plot individual trajectories
            cells = None
            if plot_trajectories:
                cells = self._plot_individual_trajectories(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    trajectory_alpha=trajectories_alpha,
                    trajectory_linewidth=trajectories_linewidth,
                    max_trajectories=max_trajectories,
                    trajectories_sample_seed=trajectories_sample_seed,
                    trajectory_ids=trajectory_ids,
                    full_timeseries=full_timeseries,
                    normalize=normalize,
                    add_legend=add_legend,
                    use_subplots=use_subplots,
                    plot_shift_indicator=plot_shift_indicator,
                    shift_indicator_size=shift_indicator_size,
                    **plot_kwargs,
                )

                if highlight_color and id is not None:
                    self._highlight_cluster_segments(
                        current_ax=current_ax,
                        plot_var=timeseries_var,
                        var=var,
                        cluster_id=id,
                        highlight_color=highlight_color,
                        highlight_alpha=highlight_alpha,
                        highlight_linewidth=highlight_linewidth,
                        full_timeseries=full_timeseries,
                        normalize=normalize,
                        cells=cells,
                    )

            # Handle axis cleanup for subplots
            if use_subplots:
                y_label = self._cleanup_subplot_axes(
                    current_ax=current_ax,
                    i=i,
                    cluster_ids_list=cluster_ids_list,
                    n_subplots_col=ncols,
                    timeseries_ylabel=show_ylabels,
                )

            # Handle legend
            self._apply_legend(
                current_ax=current_ax,
                cluster_id=id,
                add_legend=add_legend,
                use_subplots=use_subplots,
                i=i,
                cluster_ids_list=cluster_ids_list,
            )

        # Set title
        self._set_timeseries_title(
            ts_axes_list=ts_axes_list,
            map=plot_map,
            use_subplots=use_subplots,
            plot_all_data=plot_all_data,
            cluster_ids_list=cluster_ids_list,
            plot_var=timeseries_var,
            var=var,
            plot_individual=plot_trajectories,
            has_aggregate=has_aggregate,
            single_plot=single_plot,
            max_trajectories=max_trajectories,
            full_timeseries=full_timeseries,
            normalize=normalize,
            y_label=y_label,
        )

        if shared_ylabel is not None and use_subplots and len(ts_axes_list) > 0:
            if shared_ylabel == "auto":
                shared_label = y_label or self._timeseries_var_ylabel(timeseries_var)
            else:
                shared_label = shared_ylabel
            n_ts_rows = int(np.ceil(len(ts_axes_list) / ncols))
            ylabel_axes = ts_axes_list[:n_ts_rows]
            plot_fig = fig if fig is not None else ts_axes_list[0].figure
            if plot_fig is not None:
                add_shared_ylabel(
                    plot_fig,
                    ylabel_axes,
                    shared_label,
                    pad=shared_ylabel_pad,
                    fontsize=shared_ylabel_fontsize,
                )

        # Return appropriate values using helper function
        return self._package_timeseries_result(
            fig=fig,
            map=plot_map,
            use_subplots=use_subplots,
            map_ax=map_ax,
            ts_axes_list=ts_axes_list,
        )

    def overview(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(6),
        map_style: Optional[Union[MapStyle, dict]] = None,
        mode: Literal["timeseries", "aggregated"] = "timeseries",
        **kwargs: Any,
    ) -> Tuple[FigureBase | None, dict]:
        """Create an overview plot with map and timeseries for clusters.

        This is a convenience method that creates a combined visualization showing
        both the spatial distribution of clusters on a map and their corresponding
        timeseries. It automatically enables subplots and map display.

        If no clusters exist (or none of ``cluster_ids`` are present), still returns a
        figure of the usual layout: the map basemap plus a timeseries panel reading
        ``"No clusters identified"``.

        Args:
            var: Base variable or cluster variable. If None, TOAD will attempt
                to infer which variable to use. A ValueError is raised if the variable cannot be
                uniquely determined.
            cluster_ids: ID or list of IDs of clusters to plot. Defaults to range(6) (clusters 0-5).
            map_style: Map style configuration. Can be a MapStyle object, a dict with
                MapStyle fields, or None (uses defaults). If dict, missing keys use defaults.
            mode: Visualization mode. "timeseries" shows individual trajectories,
                "aggregated" shows statistical summaries (median, range, IQR).
            **kwargs: Additional arguments passed to timeseries() method.

        Returns:
            Tuple of (figure, dict) with keys 'map' and 'timeseries'.
            - 'map': Axes for the map plot
            - 'timeseries': List of axes for timeseries subplots (one per cluster)
        """
        result = self.timeseries(
            var=var,
            cluster_ids=cluster_ids,
            plot_map=True,
            subplots=True,
            map_style=map_style,
            plot_trajectory_std=mode == "aggregated",
            plot_trajectories=mode == "timeseries",
            plot_trajectory_range=mode == "aggregated",
            plot_median=mode == "aggregated",
            **kwargs,
        )
        return cast(Tuple[FigureBase | None, dict], result)

    def _empty_cluster_map_timeseries(
        self,
        *,
        figsize: Optional[Tuple[float, float]],
        vertical: bool,
        width_ratios: Tuple[float, float],
        height_ratios: Optional[Tuple[float, float]],
        hspace: float,
        wspace: float,
        ncols: int,
        map_style: Optional[Union[MapStyle, dict]],
    ) -> Tuple[FigureBase | None, dict]:
        """Build map + placeholder panel when no clusters are available."""
        # One dummy entry so layout matches a single timeseries panel.
        fig, ts_axes_list, map_ax = self._setup_timeseries_axes(
            map=True,
            use_subplots=True,
            cluster_ids_list=[None],
            n_subplots_col=max(1, ncols),
            figsize=figsize,
            vertical=vertical,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            hspace=hspace,
            wspace=wspace,
            ax=None,
            map_style=map_style,
        )
        ts_ax = ts_axes_list[0]
        ts_ax.set_xticks([])
        ts_ax.set_yticks([])
        for spine in ts_ax.spines.values():
            spine.set_visible(False)
        ts_ax.text(
            0.5,
            0.5,
            "No clusters identified",
            ha="center",
            va="center",
            transform=ts_ax.transAxes,
            fontsize=12,
        )
        return cast(
            Tuple[FigureBase | None, dict],
            self._package_timeseries_result(
                fig=fig,
                map=True,
                use_subplots=True,
                map_ax=map_ax,
                ts_axes_list=ts_axes_list,
            ),
        )

    def shift_dist(self, figsize: Optional[tuple] = None, yscale: str = "log", bins=20):
        """Plot histograms showing the distribution of shifts for each shift variable.

        Args:
            figsize: Figure size (width, height) in inches. If None, defaults to
                (12, 2 * number of shift variables).
            yscale: Scale for the y-axis. Defaults to "log".
            bins: Number of bins for the histogram. Defaults to 20.

        Returns:
            Tuple of (figure, axes). Axes is a numpy array of axes (one per shift variable).
        """

        shift_vars = self.td.shift_vars
        all_data = [self.td.get_shifts(var).values.flatten() for var in shift_vars]

        if figsize is None:
            figsize = (12, 2 * len(shift_vars))

        fig, axs = plt.subplots(
            nrows=len(shift_vars),
            figsize=figsize,
        )
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        if len(axs) > 1:
            _remove_ticks(axs[:-1], keep_y=True)
            _remove_spines(axs[:-1], spines=["right", "top"])
        _remove_spines(axs[-1], spines=["right", "top"])

        for i in range(len(shift_vars)):
            data = all_data[i]
            # Do not force range, let matplotlib auto-determine
            axs[i].hist(
                data,
                bins=bins,
            )
            axs[i].set_ylabel(f"#{shift_vars[i]}", rotation=0, ha="right", va="center")
            axs[i].set_yscale(yscale)
            # Always draw xticks and label for every panel
            axs[i].tick_params(axis="x", which="both", labelbottom=True)
            axs[i].set_xlabel("Shift value")
        return fig, axs

    def _prepare_map_plot_params(
        self, ax: Axes, plot_params: dict[str, Any]
    ) -> Tuple[dict[str, Any], bool]:
        """Prepare plot parameters for different grid types and determine plotting method.

        Args:
            ax: Matplotlib axes to plot on.
            plot_params: Dictionary of plot parameters to update.

        Returns:
            Tuple of (updated_plot_params, use_pcolormesh) where use_pcolormesh indicates
            whether to use pcolormesh (True) or regular plot (False).
        """
        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        # Check if axes is a GeoAxes (has projection)
        projection_attr = getattr(ax, "projection", None)
        is_geoaxes = projection_attr is not None

        # plot on lat/lon coordinates if available
        if has_latlon:
            plot_params["x"] = lon_name
            plot_params["y"] = lat_name
            plot_params["transform"] = ccrs.PlateCarree()
        elif is_geoaxes:
            # GeoAxes but no lat/lon - use spatial dimensions explicitly
            # This ensures xarray.plot uses the correct dimensions
            space_dims = self.td.space_dims
            if len(space_dims) >= 2:
                plot_params["x"] = space_dims[1]  # x/lon dimension
                plot_params["y"] = space_dims[0]  # y/lat dimension
            # Don't set transform - let xarray handle it based on the GeoAxes projection
        else:
            # Regular axes, no lat/lon - use spatial dimensions explicitly
            # This ensures xarray.plot uses the correct dimensions
            space_dims = self.td.space_dims
            if len(space_dims) >= 2:
                plot_params["x"] = space_dims[1]  # x/lon dimension
                plot_params["y"] = space_dims[0]  # y/lat dimension

        space_dims = self.td.space_dims
        degenerate_spatial = len(space_dims) >= 2 and any(
            int(self.td.data.sizes.get(d, 0)) <= 1 for d in space_dims
        )

        # Regular axes without lat/lon use pcolormesh for explicit dim handling.
        # Lat/lon + a singleton spatial dimension must also use pcolormesh: xarray's
        # default plot path treats the field as 1D and rejects both x=lon and y=lat.
        use_pcolormesh = (not has_latlon and not is_geoaxes) or (
            has_latlon and degenerate_spatial
        )

        return plot_params, use_pcolormesh

    def _parse_cluster_ids(
        self,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]],
        var: Optional[str],
    ) -> Tuple[List[Optional[int]], bool, bool]:
        """Parse and validate cluster_ids input.

        Args:
            cluster_ids: Cluster IDs to parse
            var: Variable name for clusters

        Returns:
            Tuple of (cluster_ids_list, single_plot, plot_all_data)
        """
        plot_all_data = cluster_ids is None

        if plot_all_data:
            # Treat as single pseudo-cluster with id=None
            cluster_ids_list: List[Optional[int]] = [None]
            single_plot = True
            var = self.td._get_base_var_if_none(var)
        else:
            # Filter cluster_ids to only include existing clusters
            var = self.td._get_base_var_if_none(var)
            cluster_ids = _filter_by_existing_clusters(self.td, cluster_ids, var)

            # Check if we have any clusters to plot
            if len(cluster_ids) == 0:
                # Return empty list to signal no valid clusters (handled by caller)
                return [], False, plot_all_data

            # Convert single cluster_id to list for consistent handling
            if isinstance(cluster_ids, int):
                single_plot = True
                cluster_ids_list = [cluster_ids]
            else:
                single_plot = False
                cluster_ids_list = list(cluster_ids)

        return cluster_ids_list, single_plot, plot_all_data

    def _infer_plot_var(
        self,
        var: Optional[str],
        plot_var: Optional[str],
        plot_shifts: bool,
        map: bool,
    ) -> str:
        """Infer the plot variable for timeseries.

        Args:
            var: Base variable name
            plot_var: Explicitly provided plot variable
            plot_shifts: Whether to plot shifts variable
            map: Whether map is being plotted

        Returns:
            The inferred plot variable name
        """
        # Determine plot_var for timeseries
        if plot_var is None:
            plot_var = var
        plot_var = self.td._get_base_var_if_none(plot_var)

        # Handle map setup and determine plot_var for timeseries when map=True
        if var is not None:
            # Get base variable from clusters attrs for timeseries if plot_var wasn't explicitly set
            # (i.e., if it equals var, meaning user didn't specify a different variable)
            clusters_obj = self.td.get_clusters(var)
            if plot_var == var:
                plot_var = clusters_obj.attrs[_attrs.BASE_VARIABLE]
            if plot_shifts:
                plot_var = clusters_obj.attrs[_attrs.SHIFTS_VARIABLE]

        if plot_var is None:
            raise ValueError("Failed to infer plot_var")
        return plot_var

    def _setup_timeseries_axes(
        self,
        map: bool,
        use_subplots: bool,
        cluster_ids_list: List[Optional[int]],
        n_subplots_col: int,
        figsize: Optional[Tuple[float, float]],
        vertical: bool,
        width_ratios: Tuple[float, float],
        height_ratios: Optional[Tuple[float, float]],
        hspace: float,
        wspace: float,
        ax: Optional[Axes],
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[
        FigureBase | None,
        List[Axes],
        Optional[Axes],
    ]:
        """Setup figure and axes layout for timeseries plots.

        Args:
            map: Whether to include a map
            use_subplots: Whether to use subplots
            cluster_ids_list: List of cluster IDs to plot
            n_subplots_col: Number of columns for subplot grid
            figsize: Figure size
            vertical: Whether to arrange map vertically
            width_ratios: Width ratios for horizontal layout
            height_ratios: Height ratios for vertical layout
            hspace: Height space between subplots
            wspace: Width space between subplots
            ax: Optional existing axes
            map_style: Map style configuration

        Returns:
            Tuple of (figure, timeseries_axes_list, map_ax)
        """
        fig = None
        ts_axes_list: List[Axes] = []
        map_ax = None

        if map or use_subplots:
            # Create figure with constrained_layout
            fig = plt.figure(figsize=figsize, constrained_layout=True)

            if map:
                # Create map first, then timeseries subplots
                if vertical:
                    main_gs = fig.add_gridspec(
                        nrows=2,
                        ncols=1,
                        height_ratios=list(height_ratios) if height_ratios else [1, 1],
                        hspace=hspace,
                    )
                    _, map_ax = self.map(
                        nrows=1,
                        ncols=1,
                        subplot_spec=main_gs[0],
                        map_style=map_style,
                    )
                    ts_subplot_spec = main_gs[1]
                else:
                    main_gs = fig.add_gridspec(
                        nrows=1,
                        ncols=2,
                        width_ratios=list(width_ratios),
                    )
                    _, map_ax = self.map(
                        nrows=1,
                        ncols=1,
                        subplot_spec=main_gs[0, 0],
                        map_style=map_style,
                    )  # type: ignore
                    ts_subplot_spec = main_gs[0, 1]  # type: ignore

                # Create timeseries subplots in remaining space
                ts_axes_list = _create_timeseries_layout(
                    fig=fig,
                    n_clusters=len(cluster_ids_list),
                    n_subplots_col=n_subplots_col,
                    subplot_spec=ts_subplot_spec,
                    hspace=hspace,
                    wspace=wspace,
                )
            else:
                # Only subplots, no map
                ts_axes_list = _create_timeseries_layout(
                    fig=fig,
                    n_clusters=len(cluster_ids_list),
                    n_subplots_col=n_subplots_col,
                    hspace=hspace,
                    wspace=wspace,
                )
        else:
            # Single plot - use provided ax or create new one
            create_new_ax = ax is None
            if create_new_ax:
                fig, ax = plt.subplots()
            ts_axes_list = [ax]  # Use single ax for consistency

        return fig, ts_axes_list, map_ax

    def _assign_cluster_colors(
        self,
        cluster_ids_list: List[Optional[int]],
        color: Optional[str],
        cmap: Union[str, ListedColormap],
        map: bool,
        use_subplots: bool,
    ) -> Optional[List[str]]:
        """Assign colors to clusters.

        Args:
            cluster_ids_list: List of cluster IDs
            color: Single color override
            cmap: Colormap to use
            map: Whether map is being plotted
            use_subplots: Whether subplots are being used

        Returns:
            List of colors or None if single color should be used
        """
        colors = None
        if map or (use_subplots and len(cluster_ids_list) > 1):
            colors = _get_cmap_seq(stops=len(cluster_ids_list), cmap=cmap)
        return colors

    def _get_cluster_color(
        self,
        i: int,
        cluster_ids_list: List[Optional[int]],
        color: Optional[str],
        colors: Optional[List[str]],
        cmap: Union[str, ListedColormap],
    ) -> str:
        """Get color for a specific cluster.

        Args:
            i: Index of cluster in cluster_ids_list
            cluster_ids_list: List of cluster IDs
            color: Single color override
            colors: Pre-computed color list
            cmap: Colormap to use

        Returns:
            Color string for the cluster
        """
        if color:
            return color
        else:
            if len(cluster_ids_list) == 1:
                return "black"
            else:
                if colors:
                    return colors[i]
                else:
                    return _get_cmap_seq(stops=len(cluster_ids_list), cmap=cmap)[i]

    def _plot_timeseries_map(
        self,
        map_var: Optional[str],
        cluster_ids_list: List[Optional[int]],
        map_ax: Axes,
        colors: Optional[List[str]],
        color: Optional[str],
        map_include_all_clusters: bool,
        map_cmap_other: Optional[Union[str, Colormap]],
        map_style: Optional[Union[MapStyle, dict]],
        **plot_kwargs: Any,
    ) -> None:
        """Plot map alongside timeseries.

        Args:
            map_var: Variable for map
            cluster_ids_list: List of cluster IDs to plot on map
            map_ax: Axes for map
            colors: Pre-computed color list
            color: Single color override
            map_include_all_clusters: Whether to plot all clusters
            map_cmap_other: Colormap for remaining clusters
            map_style: Map style configuration (controls plot_contour, plot_fill, add_labels)
            **plot_kwargs: Additional plot arguments
        """
        # Don't plot remaining clusters on map if not requested
        map_cmap_other = None if not map_include_all_clusters else map_cmap_other
        # Filter out None values for cluster_ids
        cluster_ids_for_map = [id for id in cluster_ids_list if id is not None]
        # Determine color for map: single color if one cluster, list if multiple, or use provided color
        map_color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None
        if colors:
            if len(colors) == 1:
                map_color = colors[0]
            else:
                # Cast List[str] to List[Union[str, Tuple]] for type checker
                map_color = colors  # type: ignore[assignment]
        elif color:
            map_color = color
        self.cluster_map(
            map_var,
            cluster_ids=cluster_ids_for_map,
            color=map_color,
            ax=map_ax,
            map_cmap_other=map_cmap_other,
            map_style=map_style,
            **plot_kwargs,
        )

    def _plot_trajectory_range_band(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        range_alpha: float,
        normalize: Optional[Literal["max", "max_each"]] | str,
        time_dim: str,
        full_timeseries: bool = True,
    ) -> None:
        """Plot full range (min to max) as shaded area.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for the band
            range_alpha: Alpha transparency
            normalize: Normalization method
            time_dim: Time dimension name
            full_timeseries: If True, plot the full timeseries. If False, only plot the cluster segment.
        """
        ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
            "keep_full_timeseries": full_timeseries,
        }
        if cluster_id is not None:
            ts_kwargs["cluster_var"] = var

        min_ts = self.td.get_timeseries(
            aggregation="min",
            **ts_kwargs,
        )
        max_ts = self.td.get_timeseries(
            aggregation="max",
            **ts_kwargs,
        )
        current_ax.fill_between(
            self.td.data[time_dim].values,
            min_ts,
            max_ts,
            color=id_color,
            alpha=range_alpha,
            zorder=0,
        )

    def _plot_iqr_band(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        iqr_alpha: float,
        normalize: Optional[Literal["max", "max_each"]] | str,
        time_dim: str,
        percentile_lower: float = 0.16,
        percentile_upper: float = 0.84,
        fill_zorder: int = 1,
        full_timeseries: bool = True,
    ) -> None:
        """Plot interquartile range as shaded area between two percentiles.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for the band
            iqr_alpha: Alpha transparency
            normalize: Normalization method
            time_dim: Time dimension name
            percentile_lower: Lower percentile for band (default 0.16)
            percentile_upper: Upper percentile for band (default 0.84)
            fill_zorder: Z-order for fill_between (default 1)
            full_timeseries: If True, plot the full timeseries. If False, only plot the cluster segment.
        """
        ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
            "keep_full_timeseries": full_timeseries,
        }
        if cluster_id is not None:
            ts_kwargs["cluster_var"] = var

        p_low_ts = self.td.get_timeseries(
            aggregation="percentile",
            percentile=percentile_lower,
            **ts_kwargs,
        )
        p_up_ts = self.td.get_timeseries(
            aggregation="percentile",
            percentile=percentile_upper,
            **ts_kwargs,
        )
        current_ax.fill_between(
            self.td.data[time_dim].values,
            p_low_ts,
            p_up_ts,
            color=id_color,
            alpha=iqr_alpha,
            zorder=fill_zorder,
        )

    def _plot_mean_curve(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        mean_linewidth: float,
        add_legend: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
        full_timeseries: bool = True,
    ) -> None:
        """Plot mean timeseries curve.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for the curve
            mean_linewidth: Line width
            add_legend: Whether to add legend
            normalize: Normalization method
            full_timeseries: If True, plot the full timeseries. If False, only plot the cluster segment.
        """
        ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
            "keep_full_timeseries": full_timeseries,
        }
        if cluster_id is not None:
            ts_kwargs["cluster_var"] = var

        if cluster_id is None:
            label = "mean"
        else:
            label = f"#{cluster_id}"

        self.td.get_timeseries(
            aggregation="mean",
            **ts_kwargs,
        ).plot(
            ax=current_ax,
            color=id_color,
            lw=mean_linewidth,
            label=label if add_legend else "__nolegend__",
            zorder=3,
        )  # type: ignore

    def _plot_median_curve(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        median_linewidth: float,
        add_legend: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
        full_timeseries: bool = True,
    ) -> None:
        """Plot median timeseries curve.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for the curve
            median_linewidth: Line width
            add_legend: Whether to add legend
            normalize: Normalization method
            full_timeseries: If True, plot the full timeseries. If False, only plot the cluster segment.
        """
        ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
            "keep_full_timeseries": full_timeseries,
        }
        if cluster_id is not None:
            ts_kwargs["cluster_var"] = var

        if cluster_id is None:
            label = "median"
        else:
            label = f"#{cluster_id}"

        self.td.get_timeseries(
            aggregation="median",
            **ts_kwargs,
        ).plot(
            ax=current_ax,
            color=id_color,
            lw=median_linewidth,
            label=label if add_legend else "__nolegend__",
            zorder=3,
        )  # type: ignore

    def _plot_cluster_duration(
        self,
        current_ax: Axes,
        var: str,
        cluster_id: int,
        shift_color: str,
        cluster_duration_alpha: float,
    ) -> None:
        """Plot shift duration as horizontal shading.

        Args:
            current_ax: Axes to plot on
            var: Base variable name
            cluster_id: Cluster ID
            shift_color: Color for shading
            cluster_duration_alpha: Alpha transparency
        """

        start = self.td.stats(var).time.start(cluster_id)
        end = self.td.stats(var).time.end(cluster_id)

        if start == end:
            current_ax.axvline(
                start,  # type: ignore[arg-type]
                color=shift_color,
                alpha=cluster_duration_alpha,
                zorder=0,
            )
        else:
            # Matplotlib supports np.datetime64 and cftime.datetime directly at runtime
            current_ax.axvspan(
                self.td.stats(var).time.start(cluster_id),  # type: ignore[arg-type]
                self.td.stats(var).time.end(cluster_id),  # type: ignore[arg-type]
                facecolor=shift_color,
                edgecolor="none",
                alpha=cluster_duration_alpha,
                zorder=-100,
            )

    def _plot_individual_trajectories(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        trajectory_alpha: float,
        trajectory_linewidth: float,
        max_trajectories: int,
        trajectories_sample_seed: int,
        full_timeseries: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
        add_legend: bool,
        use_subplots: bool,
        trajectory_ids: Optional[List[int]] = None,
        plot_shift_indicator: bool = False,
        shift_indicator_size: float = 5.0,
        **plot_kwargs: Any,
    ) -> Optional[Any]:
        """Plot individual cell trajectories.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for trajectories
            trajectory_alpha: Alpha transparency
            trajectory_linewidth: Line width
            max_trajectories: Maximum number of trajectories
            trajectories_sample_seed: Seed for random sampling
            full_timeseries: Whether to plot full timeseries
            normalize: Normalization method
            add_legend: Whether to add legend
            use_subplots: Whether using subplots
            trajectory_ids: Exact cell indices to plot. Overrides max_trajectories/seed.
            plot_shift_indicator: If True, overlay dots at in-cluster timesteps on each trajectory.
            shift_indicator_size: Marker size (in points) for dots.
            **plot_kwargs: Additional plot arguments
        """
        is_real_cluster = cluster_id is not None

        individual_ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
            "aggregation": "raw",
        }
        if is_real_cluster:
            individual_ts_kwargs["cluster_var"] = var
            individual_ts_kwargs["keep_full_timeseries"] = full_timeseries

        cells = self.td.get_timeseries(**individual_ts_kwargs)

        if cells is None:
            if is_real_cluster:
                raise ValueError(f"No timeseries found for cluster {cluster_id}")
            else:
                raise ValueError(f"No timeseries found for {plot_var}")

        # Determine which cells to plot
        if trajectory_ids is not None:
            order = [i for i in trajectory_ids if i < len(cells)]
            if len(order) < len(trajectory_ids):
                skipped = [i for i in trajectory_ids if i >= len(cells)]
                logger.warning(
                    f"trajectory_ids {skipped} are out of range (cluster has {len(cells)} cells) and will be skipped."
                )
        else:
            max_trajectories_actual = int(np.min([max_trajectories, len(cells)]))
            order = np.arange(len(cells))
            np.random.seed(trajectories_sample_seed)
            np.random.shuffle(order)
            order = order[:max_trajectories_actual]

        # Pre-fetch per-cell detection timestep mask for shift dots.
        # Uses the spatio-temporal cluster mask (True only at the detection event per cell,
        # not at every timestep in the cluster window) — same logic as the cluster label
        # variable masked by get_cluster_mask().
        detection_mask_ts = None
        if plot_shift_indicator and is_real_cluster:
            cl = self.td.get_clusters(var).where(
                self.td.get_cluster_mask(var, cluster_id)
            )
            detection_mask_ts = cl.toad.to_timeseries(time_dim=self.td.time_dim)

        dot_color = id_color

        for plot_idx, cell_idx in enumerate(order):
            if is_real_cluster:
                # Add label on first trajectory if legend is enabled
                # For single plot: add label on first trajectory of each cluster
                # For subplots: don't add label to line (we'll use ax.text instead)
                if add_legend and plot_idx == 0 and not use_subplots:
                    # Label each cluster (only for single plot, not subplots)
                    add_label = f"#{cluster_id}"
                else:
                    add_label = "__nolegend__"
            else:
                add_label = "__nolegend__"
            cells[cell_idx].plot(
                ax=current_ax,
                color=id_color,
                alpha=trajectory_alpha,
                lw=trajectory_linewidth,
                label=add_label,
                **plot_kwargs,
            )

            if detection_mask_ts is not None and cell_idx < len(detection_mask_ts):
                ts = cells[cell_idx]
                det_ts = detection_mask_ts[cell_idx]
                valid = ~np.isnan(det_ts.values)
                if valid.any():
                    current_ax.plot(
                        ts[self.td.time_dim].values[valid],
                        ts.values[valid],
                        marker="o",
                        linestyle="none",
                        color=dot_color,
                        alpha=trajectory_alpha,
                        markersize=shift_indicator_size,
                        zorder=5,
                    )

        return cells

    def _highlight_cluster_segments(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: int,
        highlight_color: str,
        highlight_alpha: float,
        highlight_linewidth: float,
        full_timeseries: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
        cells: Optional[Any] = None,
    ) -> None:
        """Highlight cluster segments when full_timeseries is True.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID
            highlight_color: Color for highlight
            highlight_alpha: Alpha transparency
            highlight_linewidth: Line width
            full_timeseries: Whether full timeseries was plotted
            normalize: Normalization method
            cells: Optional pre-fetched cells (when full_timeseries=False)
        """
        if not full_timeseries:
            # Reuse cells if already fetched with keep_full_timeseries=False
            if cells is not None:
                cells_highlight = cells
            else:
                return
        else:
            highlight_ts_kwargs = {
                "var": plot_var,
                "cluster_id": cluster_id,
                "cluster_var": var,
                "normalize": normalize,
                "aggregation": "raw",
                "keep_full_timeseries": False,
            }
            cells_highlight = self.td.get_timeseries(**highlight_ts_kwargs)

        for ts in cells_highlight:
            ts.plot(
                ax=current_ax,
                color=highlight_color,
                alpha=highlight_alpha,
                lw=highlight_linewidth,
            )

    def _cleanup_subplot_axes(
        self,
        current_ax: Axes,
        i: int,
        cluster_ids_list: List[Optional[int]],
        n_subplots_col: int,
        timeseries_ylabel: bool,
    ) -> str:
        """Clean up axes for subplots.

        Args:
            current_ax: Current axes
            i: Index of current subplot
            cluster_ids_list: List of cluster IDs
            n_subplots_col: Number of columns
            timeseries_ylabel: Whether to show y-label

        Returns:
            y_label string (empty if not first subplot or timeseries_ylabel is True)
        """
        y_label = ""
        current_ax.set_title("")

        if not timeseries_ylabel:
            if i == 0:
                y_label = current_ax.get_ylabel()
            current_ax.set_ylabel("")

        # Determine if this subplot is in the bottom row
        # With column-major ordering: subplot i is at row (i % n_ts_rows) and column (i // n_ts_rows)
        n_ts = len(cluster_ids_list)
        n_ts_rows = int(np.ceil(n_ts / n_subplots_col))
        current_row = i % n_ts_rows
        current_col = i // n_ts_rows
        # A subplot is in the bottom row if it's in the last row position (n_ts_rows - 1)
        # This applies to all columns - all subplots in the bottom row should show xlabels
        # For incomplete columns, check if this is the last subplot in its column
        # The last subplot in column c would be at index: min((c+1)*n_ts_rows - 1, n_ts - 1)
        is_bottom_row = current_row == n_ts_rows - 1
        last_idx_in_column = min((current_col + 1) * n_ts_rows - 1, n_ts - 1)
        is_last_in_incomplete_column = (i == last_idx_in_column) and not is_bottom_row
        is_bottom_in_column = is_bottom_row or is_last_in_incomplete_column

        # Handle axis cleanup
        if not is_bottom_in_column:
            current_ax.set_xlabel("")
            _remove_spines(current_ax, ["right", "top", "bottom"])
        else:
            _remove_spines(current_ax, ["right", "top"])

        if not is_bottom_in_column:
            _remove_ticks(current_ax, keep_y=True)

        # capitalise labels, sometimes they are all lowercase in nc files
        # current_ax.set_xlabel(current_ax.get_xlabel().capitalize())
        # current_ax.set_ylabel(current_ax.get_ylabel().capitalize())

        return y_label

    def _apply_legend(
        self,
        current_ax: Axes,
        cluster_id: Optional[int],
        add_legend: bool,
        use_subplots: bool,
        i: int,
        cluster_ids_list: List[Optional[int]],
    ) -> None:
        """Apply legend or cluster ID label to axes.

        Args:
            current_ax: Axes to add legend to
            cluster_id: Cluster ID (None for all data)
            add_legend: Whether to add legend
            use_subplots: Whether using subplots
            i: Index of current cluster
            cluster_ids_list: List of all cluster IDs
        """
        if not add_legend:
            return

        is_real_cluster = cluster_id is not None

        if use_subplots:
            # For subplots: use ax.text to add cluster ID label (no color needed)
            if is_real_cluster:
                # Position text in upper right corner using axes coordinates
                current_ax.text(
                    1.02,
                    1.02,
                    f"#{cluster_id}",
                    ha="right",
                    va="top",
                    transform=current_ax.transAxes,
                )
        else:
            # For single plot: use legend
            # Check if there are any labeled artists before calling legend()
            handles, labels = current_ax.get_legend_handles_labels()
            has_labels = any(label and not label.startswith("_") for label in labels)

            if has_labels:
                # Single plot: only show legend on the last iteration
                # Position in upper right corner
                if i == len(cluster_ids_list) - 1:
                    legend = current_ax.legend(frameon=False, loc="upper right")
                    for handle in legend.get_lines():
                        handle.set_alpha(1.0)

    def _set_timeseries_title(
        self,
        ts_axes_list: List[Axes],
        map: bool,
        use_subplots: bool,
        plot_all_data: bool,
        cluster_ids_list: List[Optional[int]],
        plot_var: str,
        var: str,
        plot_individual: bool,
        has_aggregate: bool,
        single_plot: bool,
        max_trajectories: int,
        full_timeseries: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
        y_label: str,
    ) -> None:
        """Set title for timeseries plots.

        Args:
            ts_axes_list: List of timeseries axes
            map: Whether map is included
            use_subplots: Whether using subplots
            plot_all_data: Whether plotting all data
            cluster_ids_list: List of cluster IDs
            plot_var: Variable being plotted
            var: Base variable name
            plot_individual: Whether plotting individual trajectories
            has_aggregate: Whether plotting aggregated statistics
            single_plot: Whether single plot mode
            max_trajectories: Maximum trajectories
            full_timeseries: Whether full timeseries mode
            normalize: Normalization method
            y_label: Y-axis label text
        """
        # Set title for subplots (only for first subplot when map=True)
        if map and use_subplots and len(ts_axes_list) > 0:
            # Only label as "largest" if clusters are consecutive starting at 0
            cluster_ids_int = [id for id in cluster_ids_list if id is not None]
            if cluster_ids_int:
                is_zero_indexed = (
                    cluster_ids_int
                    == list(
                        range(
                            cluster_ids_int[0],
                            cluster_ids_int[0] + len(cluster_ids_int),
                        )
                    )
                    and cluster_ids_int[0] == 0
                )
                if (
                    len(cluster_ids_int) < len(self.td.get_cluster_ids(var))
                    and is_zero_indexed
                ):
                    title = f"{len(cluster_ids_int)} largest clusters"
                else:
                    title = "clusters"
                if y_label != "":
                    title += f" in {y_label}"
                ts_axes_list[0].set_title(title)

        # Set title for single plot (not subplots)
        if not use_subplots:
            is_all_data = plot_all_data or (
                len(cluster_ids_list) == 1 and cluster_ids_list[0] is None
            )
            current_ax = ts_axes_list[0]

            if is_all_data:
                # Title for all data case
                if plot_individual:
                    # Get cell count for title
                    if plot_var is None:
                        raise ValueError("Failed to infer plot_var")
                    cells = self.td.get_timeseries(
                        plot_var,
                        cluster_id=None,
                        aggregation="raw",
                        normalize=normalize,
                    )
                    if cells is not None:
                        max_trajectories_actual = np.min([max_trajectories, len(cells)])
                        if max_trajectories_actual < len(cells):
                            current_ax.set_title(
                                f"Random sample of {max_trajectories_actual} from total {len(cells)} timeseries for {plot_var}"
                            )
                        else:
                            current_ax.set_title(
                                f"{len(cells)} timeseries for {plot_var}"
                            )
                    else:
                        current_ax.set_title(f"{plot_var} timeseries")
                else:
                    current_ax.set_title(f"{plot_var} timeseries")
            else:
                # Title for clusters case
                if has_aggregate and not plot_individual:
                    # Aggregated statistics only
                    current_ax.set_title(
                        f"{plot_var} for clusters from {var} {cluster_ids_list}"
                    )
                elif plot_individual and single_plot:
                    # Individual trajectories for single cluster - get cell count for title
                    if plot_var is None:
                        raise ValueError("Failed to infer plot_var")
                    cells = self.td.get_timeseries(
                        plot_var,
                        cluster_ids_list[0],
                        cluster_var=var,
                        aggregation="raw",
                        keep_full_timeseries=full_timeseries,
                        normalize=normalize,
                    )
                    if cells is not None:
                        max_trajectories_actual = np.min([max_trajectories, len(cells)])
                        if max_trajectories_actual < len(cells):
                            current_ax.set_title(
                                f"Random sample of {max_trajectories_actual} from total {len(cells)} cell for {var} in cluster {cluster_ids_list[0]}"
                            )
                        else:
                            current_ax.set_title(
                                f"{len(cells)} timeseries for {var} in cluster {cluster_ids_list[0]}"
                            )
                elif plot_individual and not single_plot:
                    # Multiple clusters with individual trajectories
                    current_ax.set_title(
                        f"{plot_var} trajectories for clusters from {var} {cluster_ids_list}"
                    )
                elif plot_individual and has_aggregate:
                    # Both individual and aggregate
                    current_ax.set_title(
                        f"{plot_var} for clusters from {var} {cluster_ids_list}"
                    )

    def _package_timeseries_result(
        self,
        fig: FigureBase | None,
        map: bool,
        use_subplots: bool,
        map_ax: Optional[Axes],
        ts_axes_list: List[Axes],
    ) -> Tuple[FigureBase | None, Union[Axes, List[Axes], dict]]:
        """Package return values for timeseries method.

        Args:
            fig: Figure object
            map: Whether map is included
            use_subplots: Whether using subplots
            map_ax: Map axes
            ts_axes_list: List of timeseries axes

        Returns:
            Tuple of (figure, axes) with appropriate structure
        """
        if map:
            # Return dict with map and timeseries axes
            if use_subplots:
                return fig, {"map": map_ax, "timeseries": ts_axes_list}
            else:
                return fig, {"map": map_ax, "timeseries": ts_axes_list[0]}
        elif use_subplots:
            # Return list of axes
            return fig, ts_axes_list
        else:
            # Return single axes
            return fig, ts_axes_list[0]

    # ----- DEPRECATED FUNCTIONS -----

    def cluster_overview(
        self,
        **kwargs: Any,
    ):
        """Deprecated function: use td.plot.overview() instead."""
        raise DeprecationWarning("Use td.plot.overview() instead")


# end of Plotter


def _filter_by_existing_clusters(
    td, cluster_ids: Union[int, List[int], np.ndarray, range], var: str
) -> List[int]:
    """Filter cluster_ids to only include existing clusters.

    Args:
        td: TOAD object containing cluster data.
        cluster_ids: Single cluster ID or list/array/range of cluster IDs to filter.
        var: Variable name for clusters.

    Returns:
        List of cluster IDs that exist in the dataset (excluding noise cluster -1).
    """

    if isinstance(cluster_ids, int):
        cluster_ids = [cluster_ids]

    return [
        id for id in cluster_ids if id in td.get_cluster_ids(var, exclude_noise=False)
    ]


# Vertical mini-boxplot on consensus violins: thick IQR bar, median dot.
_VIOLIN_CONSENSUS_IQR_LW = 6.5
_VIOLIN_CONSENSUS_IQR_COLOR = "#2a2a2a"
_VIOLIN_CONSENSUS_MEDIAN_FACE = "#ffffff"
_VIOLIN_CONSENSUS_MEDIAN_EDGE = "#333333"


def _get_high_constrast_text_color(color: Union[tuple, str]) -> str:
    """Determines whether black or white text provides better contrast against a given background color.

    Args:
        color: The background color (matplotlib-compatible string or RGB tuple).

    Returns:
        '#ffffff' (white) or '#000000' (black) for the text color.
        Defaults to black if the color conversion fails.
    """
    try:
        brightness = (
            sum(
                to_rgb(color)[i] * factor
                for i, factor in enumerate([0.299, 0.587, 0.114])
            )
            * 255
        )
        return "#ffffff" if brightness < 128 else "#000000"
    except ValueError:
        print(f"Error converting {color} to RGB")
        return "#000000"


def _jitter_strip_x(
    n: int,
    x_center: float,
    half_span: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fixed-width horizontal jitter: ``x_center + U(-half_span, half_span)`` per point."""
    if n <= 0:
        return np.array([])
    if half_span <= 0:
        return np.full(n, x_center, dtype=np.float64)
    return x_center + rng.uniform(-half_span, half_span, size=n)


def _style_violin_bodies_iqr_median(
    ax: Axes,
    parts: Any,
    dataset: List[np.ndarray],
    positions: np.ndarray,
    colors: Sequence[str | tuple[float, ...]],
    *,
    clip_to_body: bool = True,
) -> None:
    """Style violin bodies and add a vertical mini-boxplot (IQR bar and median dot).

    By default the IQR segment and median are clipped to the violin patch so they sit inside
    the KDE. Set ``clip_to_body=False`` for half violins so the markers are drawn in full,
    centred on the category position (half over the KDE, half over the opposite strip).
    """
    bodies = parts.get("bodies", [])
    for i, pc in enumerate(bodies):
        color = colors[i % len(colors)]
        pc.set_facecolor(color)
        pc.set_alpha(0.85)
        pc.set_edgecolor("#333333")
        pc.set_linewidth(1.2)
        paths = pc.get_paths()
        if len(paths) == 0:
            continue
        clip_path = paths[0]
        arr = dataset[i]
        arr_valid = arr[~np.isnan(arr)]
        if len(arr_valid) < 1:
            continue
        q1, q2, q3 = np.percentile(arr_valid, [25, 50, 75])
        pos = float(positions[i])
        iqr_line = Line2D(
            [pos, pos],
            [q1, q3],
            color=_VIOLIN_CONSENSUS_IQR_COLOR,
            linestyle="-",
            linewidth=_VIOLIN_CONSENSUS_IQR_LW,
            solid_capstyle="round",
            zorder=10,
        )
        if clip_to_body:
            iqr_line.set_clip_path(clip_path, ax.transData)
        ax.add_line(iqr_line)
        sc = ax.scatter(
            [pos],
            [q2],
            s=28,
            c=_VIOLIN_CONSENSUS_MEDIAN_FACE,
            edgecolors=_VIOLIN_CONSENSUS_MEDIAN_EDGE,
            linewidths=0.6,
            zorder=11,
        )
        if clip_to_body:
            sc.set_clip_path(clip_path, ax.transData)


def _get_cmap_seq(
    cmap: Colormap | str,
    start: int = 0,
    end: int = -1,
    stops: int = 10,
    reverse: bool = False,
) -> List[str]:
    """Extracts a sequence of distinct colors from a matplotlib colormap.

    Args:
        cmap: Name of the matplotlib colormap.
        start: Starting index within the colormap.
        end: Ending index within the colormap. Defaults to the end of the cmap.
        stops: The number of distinct colors to extract.
        reverse: If True, reverse the order of the extracted colors.

    Returns:
        A list of color hex codes.
    """
    cmap = plt.get_cmap(cmap)
    end = (
        end if end != -1 else cmap.N
    )  # Use cmap.N to get the number of colors in the colormap
    cycle_index = np.linspace(start, end - 1, stops, dtype=int)
    colors = cmap(cycle_index)  # Generate colors using the indices
    if reverse:
        colors = colors[::-1]
    colors = [to_hex(color) for color in colors]
    return colors


def _remove_spines(
    axs: Union[Axes, List[Axes], np.ndarray],
    spines: Union[List[str], str, np.ndarray] = ["top", "right", "bottom", "left"],
):
    """Remove spines (borders) from matplotlib axes.

    Args:
        axs: Single axes, list of axes, or numpy array of axes.
        spines: Spine(s) to remove. Can be a single string, list of strings, or array.
            Valid values: "top", "right", "bottom", "left". Defaults to all four.
    """
    if isinstance(axs, Axes):
        axs = np.asarray([axs])

    if isinstance(spines, str):
        spines = np.asarray([spines])

    for ax in axs:
        for s in spines:
            ax.spines[s].set_visible(False)


# Not used...
def _replace_ax_projection(
    fig: FigureBase,
    axs: Union[np.ndarray, Axes],
    row: int,
    col: int,
    projection: str | ccrs.Projection,
) -> Union[np.ndarray, Axes]:
    """Replace the subplot at the given row and column of axs with a map projection.

    Args:
        fig: Matplotlib figure containing the subplots.
        axs: Array of axes or single Axes object.
        row: Row index of the subplot to replace.
        col: Column index of the subplot to replace.
        projection: Cartopy projection to use (string name or Projection object).

    Returns:
        Updated axes array or single Axes, matching the input type.
    """
    # Remember if input was a single Axes before conversion
    was_single_axes = isinstance(axs, Axes)

    if was_single_axes:
        # For single Axes, create 1x1 array
        axs = np.array([[axs]])
    else:
        axs = np.array(axs, ndmin=2)

    axs[row, col].remove()
    axs[row, col] = fig.add_subplot(
        axs.shape[0],
        axs.shape[1],
        row * axs.shape[1] + col + 1,
        projection=projection,
    )

    # Return single Axes if input was single Axes
    if was_single_axes:
        return axs[0, 0]
    return axs


def _remove_ticks(axs: Union[Axes, List[Axes], np.ndarray], keep_x=False, keep_y=False):
    """Remove tick marks and labels from matplotlib axes.

    Args:
        axs: Single axes, list of axes, or numpy array of axes.
        keep_x: If True, keep x-axis ticks. Defaults to False.
        keep_y: If True, keep y-axis ticks. Defaults to False.
    """
    if isinstance(axs, Axes):
        axs = np.asarray([axs])

    for ax in axs:
        if not keep_x:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            [label.set_visible(False) for label in ax.get_xticklabels()]
        if not keep_y:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            [label.set_visible(False) for label in ax.get_yticklabels()]


def _add_map_features(ax: GeoAxes, config: MapStyle) -> None:
    """Add standard map features to an axes.

    Args:
        ax: Matplotlib axes with cartopy projection
        config: Plot configuration
    """
    # Add continent shading
    if config.continent_shading:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "land",
                config.resolution,
                facecolor=config.continent_shading_color,
                edgecolor="none",
                alpha=1.0,
            ),
            zorder=0,
        )

    if config.ocean_shading:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "ocean",
                config.resolution,
                facecolor=config.ocean_shading_color,
                edgecolor="none",
                alpha=1.0,
            ),
            zorder=0,
        )

    ax.coastlines(resolution=config.resolution, linewidth=config.coastline_linewidth)

    if config.borders:
        ax.add_feature(
            cfeature.BORDERS, linestyle="-", linewidth=config.border_linewidth
        )

    if config.grid_lines:
        ax.gridlines(
            draw_labels=config.grid_labels,
            linewidth=config.grid_width,
            color=config.grid_color,
            alpha=config.grid_alpha,
            linestyle=config.grid_style,
        )


def _cluster_annotate(
    ax: Axes,
    x: float,
    y: float,
    cluster_id: int,
    acol: str,
    scale: float = 1,
    relative_coords: bool = False,
    transform: Optional[ccrs.Projection] = None,
):
    """Annotate a cluster on a map with its ID number.

    Args:
        ax: Matplotlib axes to annotate on.
        x: X coordinate for annotation (in data coordinates unless relative_coords=True).
        y: Y coordinate for annotation (in data coordinates unless relative_coords=True).
        cluster_id: Cluster ID number to display.
        acol: Background color for the annotation box.
        scale: Scale factor for font size. Defaults to 1.
        relative_coords: If True, x and y are in axes fraction coordinates (0-1).
            If False, x and y are in data coordinates. Defaults to False.
        transform: Optional cartopy projection transform for data coordinates.
            Only used when relative_coords=False.
    """
    black_or_white = _get_high_constrast_text_color(acol)
    ax.annotate(
        text=str(cluster_id),
        xy=(x, y),
        xycoords="axes fraction" if relative_coords else "data",
        annotation_clip=True,  # don't show if outside the extent of the axis
        color=acol,
        zorder=100,
        ha="center",
        va="center",
        fontsize=10 * scale,
        fontweight="bold",
        transform=transform,
        bbox=dict(
            facecolor=black_or_white,
            edgecolor="none",
            alpha=0.6 if black_or_white == "#000000" else 0.8,
            boxstyle="round4,pad=0.2",  # pad and rounding chosen to approximate square
        ),
    )


def _create_timeseries_layout(
    fig: FigureBase,
    n_clusters: int,
    n_subplots_col: int,
    subplot_spec: Any = None,
    hspace: float = 0.1,
    wspace: float = 0.1,
) -> List[Axes]:
    """Create subplot layout for timeseries plots.

    Args:
        fig: Figure to create subplots in
        n_clusters: Number of clusters (subplots to create)
        n_subplots_col: Number of columns in subplot grid
        subplot_spec: Optional gridspec subplot spec (if None, creates new gridspec)
        hspace: Height space between subplots
        wspace: Width space between subplots

    Returns:
        List of axes for timeseries plots.
    """
    n_ts_rows = int(np.ceil(n_clusters / n_subplots_col))
    ts_axes_list: List[Axes] = []

    if subplot_spec is not None:
        # Use provided subplot_spec (when map exists)
        gs = subplot_spec.subgridspec(
            nrows=n_ts_rows,
            ncols=n_subplots_col,
            hspace=hspace,
            wspace=wspace if n_subplots_col > 1 else 0,
        )
    else:
        # Create new gridspec (when no map)
        gs = fig.add_gridspec(
            nrows=n_ts_rows,
            ncols=n_subplots_col,
            hspace=hspace,
            wspace=wspace if n_subplots_col > 1 else 0,
        )

    # Create timeseries axes (column-major order: fill columns first)
    for i in range(n_clusters):
        row = i % n_ts_rows
        col = i // n_ts_rows
        ts_ax = fig.add_subplot(gs[row, col])
        ts_axes_list.append(ts_ax)

    # Hide any empty subplots (column-major order)
    for i in range(n_clusters, n_ts_rows * n_subplots_col):
        row = i % n_ts_rows
        col = i // n_ts_rows
        empty_ax = fig.add_subplot(gs[row, col])
        empty_ax.set_visible(False)

    return ts_axes_list


def _add_consensus_cluster_discrete_colorbar(
    fig: Any,
    ax: Axes,
    *,
    sorted_ids: np.ndarray,
    color_list: List[Any],
    label: str,
    shrink: float,
    pad: float,
    aspect: float,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    location: str | None = None,
) -> None:
    """Matplotlib colorbar: one segment per consensus cluster id."""
    n = len(sorted_ids)
    if n == 0:
        return
    boundaries = np.empty(n + 1, dtype=float)
    boundaries[0] = float(sorted_ids[0]) - 0.5
    for i in range(n - 1):
        boundaries[i + 1] = (float(sorted_ids[i]) + float(sorted_ids[i + 1])) / 2.0
    boundaries[-1] = float(sorted_ids[-1]) + 0.5
    cmap = ListedColormap(list(color_list))
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    horizontal_left = orientation == "horizontal" and location == "left"
    if horizontal_left:
        _add_horizontal_left_map_colorbar(
            fig,
            ax,
            sm,
            label,
            width_frac=shrink,
            pad=pad,
            aspect=aspect,
            ticks=sorted_ids.tolist(),
        )
        return
    if location is None:
        location = "bottom" if orientation == "horizontal" else "left"
    cb = fig.colorbar(
        sm,
        ax=ax,
        orientation=orientation,
        location=location,
        shrink=shrink,
        pad=pad,
        aspect=aspect,
        label=label,
    )
    cb.set_ticks(sorted_ids.tolist())


def _add_gradient_legend(
    ax: Axes,
    start: int,
    end: int,
    legend_pos: Optional[Tuple[float, float]] = None,
    legend_size: Tuple[float, float] = (0.05, 0.02),
    label_text: Optional[str] = None,
    fontsize: int = 10,
    alpha: float = 1.0,
    cmap: Optional[Union[str, Colormap]] = None,
    var: Optional[str] = None,
):
    """Add a custom gradient legend to a plot.

    This method adds a gradient legend to visualize cluster IDs from start to end.
    The legend can be automatically positioned based on the variable data or
    manually positioned using legend_pos.

    Args:
        ax: The matplotlib axes to add the legend to
        start: Starting cluster ID for the gradient
        end: Ending cluster ID for the gradient
        legend_pos: Optional tuple of (x, y) coordinates in axes fraction units
            for legend placement. If None, position is determined automatically.
        legend_size: Tuple of (width, height) for the legend size in axes fraction units.
            Defaults to (0.05, 0.02).
        label_text: Optional text label for the legend. If None, no label is added.
        fontsize: Font size for legend text. Defaults to 8.
        cmap: Optional colormap to use for the gradient. If None, uses the colormap
            from the last plotted image or defaults to cividis.
        var: Variable name used for optimal legend positioning when legend_pos is None.
            If None, uses projection-based default positions.

    Returns:
        None
    """

    # Handle automatic positioning
    if legend_pos is None:
        if var is not None:
            legend_pos = (
                0.01,
                -0.07,
            )
        else:
            # Fallback to projection-based positioning
            import cartopy.crs as ccrs

            projection = getattr(ax, "projection", None)
            if projection is not None and isinstance(projection, ccrs.Projection):
                if isinstance(projection, ccrs.PlateCarree):
                    legend_pos = (0.75, 0.95)  # top-right
                else:
                    legend_pos = (0.02, 0.95)  # top-left
            else:
                legend_pos = (0.02, 0.95)

    # Get colormap
    if cmap is None:
        # Try to get colormap from the last image in the axes
        images = [child for child in ax.get_children() if hasattr(child, "get_cmap")]
        if images:
            get_cmap_method = getattr(images[-1], "get_cmap", None)
            if get_cmap_method is not None:
                cmap = get_cmap_method()
            else:
                cmap = plt.get_cmap("cividis")  # fallback
        else:
            cmap = plt.get_cmap("cividis")  # fallback

    # Normalize cmap to Colormap (convert string to Colormap if needed)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Ensure cmap is a Colormap (fallback if somehow still None or invalid)
    if cmap is None or not isinstance(cmap, Colormap):
        cmap = plt.get_cmap("cividis")

    legend_x, legend_y = legend_pos
    legend_width, legend_height = legend_size

    # Check if we have a single cluster (start == end)
    is_single_cluster = start == end

    if is_single_cluster:
        # For single cluster, use a solid color square (middle of the colormap)
        single_color = cmap(0.5)  # Use middle color of the colormap

        rect = Rectangle(
            (legend_x, legend_y),
            legend_width,
            legend_height,
            facecolor=single_color,
            edgecolor="black",
            linewidth=0.5,
            clip_on=False,
            transform=ax.transAxes,
            zorder=1000,
        )
        ax.add_patch(rect)

        # Label for single cluster
        if label_text is None:
            label_text = f"{start}"
    else:
        # Create the gradient effect using multiple thin rectangles
        n_segments = 50
        segment_width = legend_width / n_segments

        for i in range(n_segments):
            color_val = i / (n_segments - 1)
            color = cmap(color_val)
            rect = Rectangle(
                (legend_x + i * segment_width, legend_y),
                segment_width,
                legend_height,
                facecolor=color,
                alpha=alpha,
                edgecolor="none",
                clip_on=False,
                transform=ax.transAxes,
                zorder=1000,
            )
            ax.add_patch(rect)

        # Add border around the gradient
        border_rect = Rectangle(
            (legend_x, legend_y),
            legend_width,
            legend_height,
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
            clip_on=False,
            transform=ax.transAxes,
            zorder=1000,
        )
        ax.add_patch(border_rect)

        # Label for multiple clusters
        if label_text is None:
            label_text = f"#{start}-{end}"

    # Add text label
    ax.text(
        legend_x + legend_width + 0.01,
        legend_y + legend_height / 2,
        label_text,
        transform=ax.transAxes,
        verticalalignment="center",
        fontsize=fontsize,
        clip_on=False,
    )
