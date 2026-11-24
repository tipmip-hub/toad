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

    def __init__(self, nside: Optional[int] = None, fill_empty_pixels: bool = True):
        """
        Args:
            nside: HEALPix parameter nside, which must be a power of 2. The total number of pixels in the regridded grid (npix) is calculated using the formula: npix = 12 * nside ** 2. If nside is not specified, it will be automatically determined based on the data's resolution.
            fill_empty_pixels: Where source cells are wider than the pixels, some
                pixels contain no cell centre and would stay empty. If True and the
                source grid is known (see ``set_source_grid``), each such pixel takes
                the data of its nearest source cell. Set to False for the
                behaviour of versions up to 1.0.10.
        """

        self.df_healpix: pd.DataFrame = pd.DataFrame()
        self.fill_empty_pixels = bool(fill_empty_pixels)
        self._source_grid: tuple[np.ndarray, np.ndarray] | None = None
        self._fill_map: dict[tuple[float, float], np.ndarray] | None = None

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

    def set_source_grid(self, lats: np.ndarray, lons: np.ndarray) -> None:
        """Give the centres (degrees) of all source grid cells, events or not.

        ``lats`` and ``lons`` are 2D arrays of the same shape, one value per
        cell, in the grid's own row and column order. Needed to tell an empty
        pixel inside the grid (no cell centre falls in it) from a pixel
        without an event.
        """
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        if lats.ndim != 2 or lats.shape != lons.shape:
            raise ValueError("lats and lons must be 2D arrays of the same shape.")
        self._source_grid = (lats, lons)
        self._fill_map = None

    def copy_source_grid_from(self, other: "HealPixRegridder") -> None:
        """Share the source grid (and its fill map) of ``other``."""
        self.fill_empty_pixels = other.fill_empty_pixels
        self._source_grid = other._source_grid
        same_grid = other.nside is not None and other.nside == self.nside
        self._fill_map = other._fill_map if same_grid else None

    def empty_pixel_fill_map(self) -> dict[tuple[float, float], np.ndarray]:
        """Source cell ``(lat, lon)`` -> empty pixels that take its data.

        A pixel is empty when no source cell centre falls in it. It is given to
        its nearest source cell, if the pixel centre lies within that cell's
        reach: 1.25 times half the distance to the cell's nearest diagonal
        neighbour, i.e. a little more than its centre-to-corner distance.
        Pixels outside the grid, for example over land that an ocean grid
        leaves out, stay empty.
        """
        if self._fill_map is not None:
            return self._fill_map
        self._fill_map = {}
        if (
            not self.fill_empty_pixels
            or self._source_grid is None
            or self.nside is None
        ):
            return self._fill_map

        from scipy.spatial import cKDTree

        lat2d, lon2d = self._source_grid
        ok2d = np.isfinite(lat2d) & np.isfinite(lon2d) & (np.abs(lat2d) <= 90.0)
        if int(ok2d.sum()) < 4 or min(lat2d.shape) < 2:
            return self._fill_map

        def unit(lat_deg, lon_deg):
            la, lo = np.radians(lat_deg), np.radians(lon_deg)
            return np.stack(
                [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1
            )

        xyz = unit(lat2d, lon2d)
        xyz[~ok2d] = np.nan

        # Reach of each cell: chord to its diagonal neighbours (columns wrap,
        # rows do not), which is twice the centre-to-corner distance.
        # The nearest of the four is used, so that a neighbour with placeholder
        # coordinates (masked cells in some ocean grids) cannot widen the reach.
        nrow = lat2d.shape[0]
        diag = np.full(lat2d.shape, np.inf)
        for dj in (-1, 1):
            rows = np.arange(nrow) + dj
            inside = (rows >= 0) & (rows < nrow)
            for di in (-1, 1):
                other = np.roll(xyz[np.clip(rows, 0, nrow - 1)], di, axis=1)
                d = np.linalg.norm(xyz - other, axis=-1)
                d[~inside] = np.nan
                diag = np.fmin(diag, d)
        diag[~np.isfinite(diag)] = 0.0
        reach2d = 1.25 * 0.5 * diag
        # Cells far from all their neighbours are not part of a regular mesh
        # (placeholder coordinates again); they fill nothing.
        positive = ok2d & (reach2d > 0)
        typical = np.median(reach2d[positive]) if np.any(positive) else 0.0
        reach2d[reach2d > 3 * typical] = 0.0

        ok = ok2d.ravel()
        lats, lons = lat2d.ravel()[ok], lon2d.ravel()[ok]
        reach = reach2d.ravel()[ok]

        npix = 12 * int(self.nside) ** 2
        occupied = np.zeros(npix, dtype=bool)
        occupied[self.latlon_to_healpix(lats, lons)] = True
        empty = np.flatnonzero(~occupied)
        if empty.size == 0:
            return self._fill_map

        pix_lon, pix_lat = healpix_to_lonlat(empty, self.nside, order="ring")
        tree = cKDTree(xyz.reshape(-1, 3)[ok])
        dist, cell = tree.query(unit(pix_lat.to_value(u.deg), pix_lon.to_value(u.deg)))
        keep = dist <= reach[cell]
        if not keep.any():
            # Regional grid on pixels no finer than its cells: every empty
            # pixel lies outside the grid, so there is nothing to fill.
            return self._fill_map

        order = np.argsort(cell[keep], kind="stable")
        cell_k, pix_k = cell[keep][order], empty[keep][order]
        starts = np.flatnonzero(np.r_[True, np.diff(cell_k) != 0])
        for a, b in zip(starts, np.r_[starts[1:], cell_k.size]):
            c = cell_k[a]
            self._fill_map[(float(lats[c]), float(lons[c]))] = pix_k[a:b]
        logger.debug(
            f"HealPixRegridder: {int(keep.sum())} of {npix} pixels hold no source "
            "cell centre and take their nearest source cell"
        )
        return self._fill_map

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

        # Empty pixels inside the source grid take the data of their nearest cell
        fill_map = self.empty_pixel_fill_map()
        if fill_map:
            rows, pixels = [], []
            for i, key in enumerate(zip(coords[:, 1], coords[:, 2])):
                extra = fill_map.get(key)
                if extra is not None:
                    rows.append(np.full(extra.size, i))
                    pixels.append(extra)
            if rows:
                filled = df.iloc[np.concatenate(rows)].copy()
                filled["hp_pix"] = np.concatenate(pixels)
                df = pd.concat([df, filled], ignore_index=True)

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
