import logging
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy_healpix import healpix_to_lonlat, lonlat_to_healpix

from toad.regridding.base import BaseRegridder

logger = logging.getLogger("TOAD")


def _pixels_to_order(n_pixels: float) -> float:
    """Compute HEALPix order from pixel count (npix = 12 * nside²)."""
    return 0.5 * np.log2(n_pixels / 12.0)


class HealPixRegridder(BaseRegridder):
    """Regrid data onto a equal-area HEALPix grid to avoid polar bias in clustering"""

    def __init__(self, nside: Optional[int] = None):
        """
        Args:
            nside: HEALPix parameter nside, which must be a power of 2. The total number of pixels in the regridded grid (npix) is calculated using the formula: npix = 12 * nside ** 2. If nside is not specified, it will be automatically determined based on the data's resolution.
        """

        self.df_healpix: pd.DataFrame = pd.DataFrame()

        # Make sure nside is a power of 2
        if nside is not None and not np.log2(nside).is_integer():
            raise ValueError(f"nside must be a power of 2, got {nside}")
        self.nside = nside

    def latlon_to_healpix(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Convert arrays of latitude and longitude to HEALPix pixel indices."""
        hp_pix = lonlat_to_healpix(
            np.asarray(lons, dtype=np.float64) * u.deg,
            np.asarray(lats, dtype=np.float64) * u.deg,
            self.nside,
            order="ring",
        )
        return np.asarray(hp_pix, dtype=np.int64)

    def healpix_to_latlon(self, pix: int) -> tuple:
        """Convert a HEALPix pixel index back to its center latitude and longitude."""
        lon, lat = healpix_to_lonlat(pix, self.nside, order="ring")
        lat_deg = float(lat.to_value(u.deg))
        lon_deg = float(np.mod(lon.to_value(u.deg), 360.0))
        return lat_deg, lon_deg

    def regrid(
        self,
        coords: np.ndarray,
        weights: np.ndarray,
        space_dims_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Regrid data to new coordinate system.

        Args:
            coords: 3dArray of coordinates (time, lat, lon) in that order
            weights: 1dArray of weights
            space_dims_size: Tuple of (nlat, nlon) sizes of the original grid dimensions
        Returns:
            3dArray of coordinates (time, lat, lon) in that order
            1dArray of weights
        """
        # Store original spatial coordinates
        self.original_coords = coords
        self.original_weights = weights

        # If nside is not provided, compute it automatically based on the resolution of the data
        if self.nside is None:
            n_pixels = space_dims_size[0] * space_dims_size[1]
            self.nside = 1 << int(np.ceil(_pixels_to_order(n_pixels)))
            logger.debug(
                f"HealPixRegridder: Automatically computed nside: {self.nside} based on grid resolution {space_dims_size[0]}x{space_dims_size[1]}"
            )

        # Get unique lat/lon pairs and compute healpix indices once
        unique_coords = np.unique(coords[:, 1:], axis=0)  # unique lat/lon pairs
        unique_hp_indices = self.map_orig_to_regrid(unique_coords)

        # Create mapping from lat/lon to healpix index
        coord_to_hp = {
            (lat, lon): hp_idx
            for (lat, lon), hp_idx in zip(map(tuple, unique_coords), unique_hp_indices)
        }

        # Create DataFrame with mapped healpix indices
        df = pd.DataFrame(
            {
                "time": coords[:, 0],
                "lat": coords[:, 1],
                "lon": coords[:, 2],
                "vals": weights,
                "hp_pix": [
                    coord_to_hp[(lat, lon)]
                    for lat, lon in zip(coords[:, 1], coords[:, 2])
                ],
            }
        )

        # Group and aggregate
        group_cols = [
            "time",
            "hp_pix",
        ]  # This means if multiple points fall in the same HEALPix pixel at the same time, they get averaged.
        df = df.groupby(group_cols)["vals"].mean().reset_index()

        # Add regridded coordinates
        df["lat"], df["lon"] = zip(
            *df["hp_pix"].apply(self.healpix_to_latlon)
        )  # Convert healpix index back to lat, lon

        if np.any(np.isnan(df["hp_pix"])):
            logger.warning(
                "Warning: Interpolation contains NaNs. Consider decreasing nside_p"
            )

        self.df_healpix = df
        return np.column_stack([df["time"], df["lat"], df["lon"]]), df[
            "vals"
        ].to_numpy()

    def regrid_clusters_back(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Map cluster labels back to original grid."""
        if self.original_coords is None or self.df_healpix is None:
            raise ValueError("Must call regrid() first")

        # Add cluster labels to healpix DataFrame
        self.df_healpix["cluster"] = cluster_labels

        # Create mapping dictionary from (time, hp_pix) to cluster label
        mapping = dict(
            zip(
                zip(
                    self.df_healpix["time"], self.df_healpix["hp_pix"]
                ),  # group by both time and hp_pix because hp_pix is constant with time
                cluster_labels,
            )
        )

        # Calculate healpix indices for original points
        hp_indices = self.map_orig_to_regrid(self.original_coords[:, 1:3])

        # Map back using time and healpix indices
        result = np.array(
            [
                mapping.get((time, hp), -1)  # if no mapping, return -1
                for time, hp in zip(self.original_coords[:, 0], hp_indices)
            ]
        )

        return result

    def map_orig_to_regrid(self, coords_2d: np.ndarray) -> np.ndarray:
        """
        Map (lat, lon) to a HEALPix pixel index.
        coords_2d must be array (N, 2): [(lat, lon), ...]
        """
        if coords_2d.shape[1] != 2:
            raise ValueError("coords_2d must be (N, 2) = (lat, lon)")

        lat = coords_2d[:, 0]
        lon = coords_2d[:, 1]

        # Ensure nside set — if None, infer from data resolution
        if self.nside is None:
            n_pixels = len(coords_2d)
            self.nside = 1 << int(np.ceil(_pixels_to_order(n_pixels)))

        return self.latlon_to_healpix(lat, lon)
