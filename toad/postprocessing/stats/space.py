import inspect

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from toad.utils import _all_functions, detect_latlon_names


class SpaceStats:
    """Class containing functions for calculating space-related statistics for clusters, such as mean, median, std, etc."""

    def __init__(self, toad, var):
        """Initialize SpaceStats.

        Args:
            toad: TOAD object
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
        """
        self.td = toad
        self.var = var

    def _get_cluster_coordinate_values_spacetime(self, cluster_id):
        """Get coordinate arrays for space-time cluster analysis, preferring lat/lon."""
        from toad.utils import detect_latlon_names

        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        mask = self.td.get_cluster_mask(self.var, cluster_id)

        if has_latlon:
            y_coords = self.td.data[lat_name].where(mask)
            x_coords = self.td.data[lon_name].where(mask)
            return y_coords, x_coords
        else:
            y_coords = self.td.data[self.td.space_dims[0]].where(mask)
            x_coords = self.td.data[self.td.space_dims[1]].where(mask)
            return y_coords, x_coords

    def _get_cluster_coordinate_values(self, cluster_id):
        """Get coordinate arrays for spatial footprint analysis, preferring lat/lon."""
        from toad.utils import detect_latlon_names

        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        spatial_mask = self.td.get_cluster_mask_spatial(self.var, cluster_id)

        if has_latlon:
            # Use lat/lon coordinates (works for both 1D regular and 2D irregular grids)
            lat_values = self.td.data[lat_name].where(spatial_mask)
            lon_values = self.td.data[lon_name].where(spatial_mask)
            return lat_values, lon_values
        else:
            # Fallback to dimension coordinates when lat/lon not available
            y_coords = self.td.data[self.td.space_dims[0]].where(spatial_mask)
            x_coords = self.td.data[self.td.space_dims[1]].where(spatial_mask)
            return y_coords, x_coords

    def mean(self, cluster_id):
        """Returns the mean of the spatial coordinates across space and time."""
        y_coords, x_coords = self._get_cluster_coordinate_values_spacetime(cluster_id)
        return (float(y_coords.mean()), float(x_coords.mean()))

    def median(self, cluster_id):
        """Returns the median of the spatial coordinates across space and time."""
        y_coords, x_coords = self._get_cluster_coordinate_values_spacetime(cluster_id)
        return (float(y_coords.median()), float(x_coords.median()))

    def std(self, cluster_id):
        """Returns the standard deviation of the spatial coordinates across space and time."""
        y_coords, x_coords = self._get_cluster_coordinate_values_spacetime(cluster_id)
        return (float(y_coords.std()), float(x_coords.std()))

    def footprint_mean(self, cluster_id):
        """Returns the mean of the spatial coordinates of the cluster footprint."""
        y_coords, x_coords = self._get_cluster_coordinate_values(cluster_id)
        return (float(y_coords.mean()), float(x_coords.mean()))

    def footprint_median(self, cluster_id):
        """Returns the median of the spatial coordinates of the cluster footprint."""
        y_coords, x_coords = self._get_cluster_coordinate_values(cluster_id)
        return (float(y_coords.median()), float(x_coords.median()))

    def footprint_std(self, cluster_id):
        """Returns the standard deviation of the spatial coordinates of the cluster footprint."""
        y_coords, x_coords = self._get_cluster_coordinate_values(cluster_id)
        return (float(y_coords.std()), float(x_coords.std()))

    def central_point_for_labeling(self, cluster_id) -> tuple[float, float]:
        """Calculates a central point within the cluster's spatial footprint suitable for labeling.

        This method uses the Euclidean Distance Transform to find the point within
        the cluster footprint that is furthest from any edge (the "pole of
        inaccessibility"). This ensures the point is robustly inside the cluster shape,
        even for complex geometries like rings or C-shapes.

        Args:
            cluster_id: The ID of the cluster to analyze.

        Returns:
            A tuple containing the (y, x) coordinates of the calculated central point.
            Returns (np.nan, np.nan) if the footprint is empty.
        """
        # Get the 2D spatial footprint mask (ensure it's boolean or 0/1)
        spatial_mask = self.td.get_cluster_mask_spatial(self.var, cluster_id)

        # Ensure the mask is boolean or integer type for distance transform
        if spatial_mask.dtype != bool and not np.issubdtype(
            spatial_mask.dtype, np.integer
        ):
            # Attempt conversion, assuming non-zero means True
            spatial_mask = spatial_mask > 0
        elif spatial_mask.dtype == bool:
            # Keep as is
            pass
        else:  # Integer type
            spatial_mask = spatial_mask > 0  # Convert to boolean

        # Handle empty mask case
        if not spatial_mask.any():
            return (np.nan, np.nan)

        # Calculate the distance transform
        # distance_transform_edt returns distances for non-zero elements from the nearest zero element
        distance_map = np.asarray(
            distance_transform_edt(spatial_mask.values)
        )  # Pass numpy array

        # Find the index (row, col) of the maximum distance
        # np.argmax flattens the array first, so we need unravel_index
        max_dist_idx_flat = np.argmax(distance_map)
        max_dist_idx_unraveled = np.unravel_index(max_dist_idx_flat, distance_map.shape)  # type: ignore

        # Use consistent coordinate system
        from toad.utils import detect_latlon_names

        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        if has_latlon and self.td.data[lat_name].ndim == 2:
            # 2D lat/lon coordinates - get actual geographic values
            lat_2d = self.td.data[lat_name].values
            lon_2d = self.td.data[lon_name].values
            y_label_coord = lat_2d[max_dist_idx_unraveled[0], max_dist_idx_unraveled[1]]
            x_label_coord = lon_2d[max_dist_idx_unraveled[0], max_dist_idx_unraveled[1]]
        elif has_latlon:
            # 1D lat/lon coordinates
            lat_coords = spatial_mask[lat_name].values  # type: ignore
            lon_coords = spatial_mask[lon_name].values  # type: ignore
            y_label_coord = lat_coords[max_dist_idx_unraveled[0]]
            x_label_coord = lon_coords[max_dist_idx_unraveled[1]]
        else:
            # Fallback to dimension coordinates
            y_coords = spatial_mask[self.td.space_dims[0]].values  # type: ignore
            x_coords = spatial_mask[self.td.space_dims[1]].values  # type: ignore
            y_label_coord = y_coords[max_dist_idx_unraveled[0]]
            x_label_coord = x_coords[max_dist_idx_unraveled[1]]

        return (float(y_label_coord), float(x_label_coord))

    def footprint_cumulative_area(self, cluster_id) -> int:
        """Returns the total number of spatial cells that were ever touched by the cluster."""
        return int(self.td.get_cluster_mask_spatial(self.var, cluster_id).sum().sum())

    def footprint_table(
        self,
        cluster_ids: list[int] | range | None = None,
        *,
        events=None,
        exclude_noise: bool = True,
        center_y: str | None = None,
        center_x: str | None = None,
    ) -> pd.DataFrame:
        """Batch footprint area and footprint mean centre for many clusters.

        Same definitions as :meth:`footprint_cumulative_area` and
        :meth:`footprint_mean`, computed in one pass over labeled cells.
        """
        from toad.postprocessing.stats.cluster_events import (
            _group_slices,
            load_cluster_label_events,
        )

        if cluster_ids is None:
            cluster_ids = [
                int(c)
                for c in self.td.get_cluster_ids(self.var, exclude_noise=exclude_noise)
            ]
        else:
            cluster_ids = [int(c) for c in cluster_ids]

        if center_y is None or center_x is None:
            lat_name, lon_name = detect_latlon_names(self.td.data)
            if lat_name is not None and lon_name is not None:
                center_y, center_x = "center_lat", "center_lon"
            else:
                sd0, sd1 = self.td.space_dims
                center_y, center_x = f"center_{sd0}", f"center_{sd1}"

        if events is None:
            events = load_cluster_label_events(self.td, self.var, cluster_ids)

        by_id: dict[int, np.ndarray] = {}
        if events.labels.size:
            lp = np.unique(np.stack((events.labels, events.p_idx), axis=1), axis=0)
            order = np.argsort(lp[:, 0], kind="mergesort")
            lp = lp[order]
            for cid, s, e in _group_slices(lp[:, 0]):
                by_id[cid] = lp[s:e, 1]

        rows: list[dict] = []
        for cid in cluster_ids:
            ps = by_id.get(int(cid))
            if ps is None or ps.size == 0:
                rows.append(
                    {
                        "cluster_id": int(cid),
                        "footprint_area": 0,
                        center_y: np.nan,
                        center_x: np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "cluster_id": int(cid),
                    "footprint_area": int(ps.size),
                    center_y: float(np.mean(events.y_flat[ps])),
                    center_x: float(np.mean(events.x_flat[ps])),
                }
            )
        return pd.DataFrame(rows)

    def all_stats(self, cluster_id) -> dict:
        """Return all cluster stats"""
        dict = {}
        for method_name in _all_functions(self):
            if (
                not method_name.startswith("all_stats")
                and not method_name.startswith("_")
                and len(inspect.signature(getattr(self, method_name)).parameters) == 1
            ):
                dict[method_name] = getattr(self, method_name)(cluster_id)
        return dict
