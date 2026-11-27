from abc import ABC, abstractmethod

import numpy as np


class BaseRegridder(ABC):
    """
    Internal base class of TOAD's regridders.

    Not an extension point: ``compute_clusters`` accepts only None or a
    ``HealPixRegridder`` instance; custom regridders are no longer supported.

    A regridder provides:
      - regrid(): forward mapping from original grid to regridded coordinates
      - regrid_clusters_back(): inverse mapping from regridded labels to original grid
      - map_orig_to_regrid(): direct spatial index mapping without resampling

    The mapping method is lightweight and required for consensus-based workflows.
    """

    def __init__(self):
        # Full regridder state for inversion workflows
        self.df_healpix = None  # optional storage for regridded results
        self.original_coords = None  # original coordinate array cached from regrid()

    @abstractmethod
    def regrid(
        self,
        coords: np.ndarray,
        weights: np.ndarray,
        space_dims_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Regrid per-point values into a new coordinate system.

        Args:
            coords: Array (N, 3) containing (time, lat, lon) or similar spatial coords.
            weights: Array (N,) containing scalar values to aggregate/interpolate.
            space_dims_size: Original grid shape as (ny, nx).

        Returns:
            coords_regrid: Array (N', 3) of regridded (time, lat, lon) coordinates.
            weights_regrid: Array (N',) of aggregated/interpolated weights.
        """
        pass

    @abstractmethod
    def regrid_clusters_back(self, cluster_labels: np.ndarray) -> np.ndarray:
        """
        Project cluster labels from regridded space back to original grid.

        Args:
            cluster_labels: Array (N',) of labels corresponding to regridded coords.

        Returns:
            Array (N,) of labels aligned with original grid points.
        """
        pass

    @abstractmethod
    def map_orig_to_regrid(self, coords_2d: np.ndarray) -> np.ndarray:
        """
        Lightweight mapping from original spatial points → regridded index.

        This must *not* modify the data or allocate giant grids.
        Only spatial coordinates (e.g. lat/lon or x/y) are required.

        Args:
            coords_2d: Array (N, 2) containing spatial coord pairs.

        Returns:
            Array (N,) of integer indices into the regridded space.

        Example:
            hp_idx = map_orig_to_regrid(np.column_stack([lat, lon]))
        """
        pass
