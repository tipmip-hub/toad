import numpy as np
import pandas as pd

from toad.postprocessing.stats.cluster_events import load_cluster_label_events
from toad.postprocessing.stats.general import GeneralStats
from toad.postprocessing.stats.space import SpaceStats
from toad.postprocessing.stats.time import TimeStats
from toad.utils import DEFAULT_SHIFT_THRESHOLD, _attrs, detect_latlon_names

__all__ = ["TimeStats", "SpaceStats", "GeneralStats", "Stats"]


class Stats:
    """Interface to access specialized statistics calculators for clusters: time, space, and general metrics.

    Used when calling td.stats(var) explicitly; _StatsAccessor in core.py delegates here for td.stats.time etc.
    """

    def __init__(self, toad, var):
        """Initialize the ClusterStats object.

        Args:
            toad (TOAD): TOAD object
            var (str): Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
        """
        self.td = toad
        self.var = var

    @property
    def time(self):
        """Access time-related statistics for clusters."""
        return TimeStats(self.td, self.var)

    @property
    def space(self):
        """Access space-related statistics for clusters."""
        return SpaceStats(self.td, self.var)

    @property
    def general(self):
        """Access general statistics for clusters."""
        return GeneralStats(self.td, self.var)

    def _center_column_names(self) -> tuple[str, str]:
        lat_name, lon_name = detect_latlon_names(self.td.data)
        if lat_name is not None and lon_name is not None:
            return "center_lat", "center_lon"
        sd0, sd1 = self.td.space_dims
        return f"center_{sd0}", f"center_{sd1}"

    def _empty_cluster_summary(
        self, extended: bool, center_y: str, center_x: str
    ) -> pd.DataFrame:
        cols = [
            "cluster_id",
            "start_time",
            "end_time",
            "duration_timesteps",
            "size",
            "footprint_area",
            "median_time",
            center_y,
            center_x,
        ]
        if extended:
            cols.extend(
                [
                    "iqr_68_start",
                    "iqr_68_end",
                    "avg_amplitude",
                    "max_amplitude",
                    "pooled_median_transition_time",
                    "pooled_std_transition_time",
                    "n_transition_cells",
                    "variable",
                    "method",
                    "shift_threshold",
                ]
            )
        return pd.DataFrame({c: [] for c in cols})

    def _shift_amplitudes(self, cluster_id: int) -> tuple[float, float]:
        """Mean and max |shift| over spacetime cluster members."""
        try:
            shifts_da = self.td.get_shifts(self.var)
            mask = self.td.get_cluster_mask(self.var, cluster_id)
            values = np.abs(shifts_da.where(mask).values)
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                return np.nan, np.nan
            return float(np.mean(valid)), float(np.max(valid))
        except Exception:
            return np.nan, np.nan

    def cluster_summary(
        self,
        cluster_ids: int | list[int] | range | None = None,
        *,
        extended: bool = False,
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
        exclude_noise: bool = True,
    ) -> pd.DataFrame:
        """Per-cluster overview table of time, space, and size metrics.

        Provides a compact dashboard of all detected clusters for quick inspection,
        CSV export, or downstream filtering — analogous to
        :meth:`toad.postprocessing.Aggregation.consensus_summary` for consensus labels.

        Minimal columns are assembled from the batch helpers
        :meth:`TimeStats.activity_table`, :meth:`SpaceStats.footprint_table`, and
        :meth:`GeneralStats.size_table` (same definitions as the single-cluster
        methods, one shared scan of the label field).

        Args:
            cluster_ids: Subset of clusters to include. Defaults to all non-noise clusters.
            extended: If True, add IQR timing bounds, shift amplitudes, pooled transition
                times, and clustering metadata from variable attributes.
            shift_threshold: Minimum shift magnitude for pooled transition-time columns
                (extended mode only).
            exclude_noise: Whether to exclude noise (cluster ID -1) when ``cluster_ids``
                is None.

        Returns:
            DataFrame with one row per cluster. Minimal columns always include
            ``cluster_id``, ``start_time``, ``end_time``, ``duration_timesteps``,
            ``size``, ``footprint_area``, ``median_time``, and spatial centre columns
            (``center_lat``/``center_lon`` when geographic coordinates exist).

        Example:
            >>> td.cluster_summary("temperature")
            >>> td.stats("temperature").cluster_summary(extended=True)
        """
        center_y, center_x = self._center_column_names()

        if cluster_ids is None:
            cluster_ids = [
                int(c)
                for c in self.td.get_cluster_ids(self.var, exclude_noise=exclude_noise)
            ]
        else:
            cluster_ids = [int(c) for c in cluster_ids]

        if len(cluster_ids) == 0:
            return self._empty_cluster_summary(extended, center_y, center_x)

        events = load_cluster_label_events(self.td, self.var, cluster_ids)
        df = (
            self.time.activity_table(cluster_ids, events=events)
            .merge(
                self.general.size_table(cluster_ids, events=events),
                on="cluster_id",
            )
            .merge(
                self.space.footprint_table(
                    cluster_ids,
                    events=events,
                    center_y=center_y,
                    center_x=center_x,
                ),
                on="cluster_id",
            )
        )
        df = df[
            [
                "cluster_id",
                "start_time",
                "end_time",
                "duration_timesteps",
                "size",
                "footprint_area",
                "median_time",
                center_y,
                center_x,
            ]
        ]

        if extended:
            cluster_var = str(self.td.get_clusters(self.var).name)
            attrs = self.td.data[cluster_var].attrs
            time_stats = self.time
            method = attrs.get(_attrs.METHOD_NAME)
            thr = attrs.get(_attrs.SHIFT_THRESHOLD, shift_threshold)
            base_var = self.td.get_base_var(self.var)
            ext_rows: list[dict] = []
            for cid in cluster_ids:
                iqr_start, iqr_end = time_stats.iqr_68(cid)
                avg_amp, max_amp = self._shift_amplitudes(cid)
                time_row = time_stats.summary(
                    [cid], shift_threshold=shift_threshold
                ).iloc[0]
                ext_rows.append(
                    {
                        "cluster_id": int(cid),
                        "iqr_68_start": iqr_start,
                        "iqr_68_end": iqr_end,
                        "avg_amplitude": avg_amp,
                        "max_amplitude": max_amp,
                        "pooled_median_transition_time": time_row[
                            "pooled_median_transition_time"
                        ],
                        "pooled_std_transition_time": time_row[
                            "pooled_std_transition_time"
                        ],
                        "n_transition_cells": int(time_row["n_transition_cells"]),
                        "variable": base_var,
                        "method": method,
                        "shift_threshold": thr,
                    }
                )
            df = df.merge(pd.DataFrame(ext_rows), on="cluster_id")

        return df.reset_index(drop=True)
