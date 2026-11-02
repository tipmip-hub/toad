"""Multi-model aggregation (MMA) for consensus clustering across saved cluster label files."""

import ast
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toad._version import __version__

if TYPE_CHECKING:
    from toad.plotting import MapStyle
    from toad.plotting.consensus_overview import MMAPlotView
from toad.plotting.consensus_maps import plot_collapsed_consensus_labels_map
from toad.regridding.healpix import HealPixRegridder
from toad.utils import _attrs, detect_latlon_names
from toad.utils.consensus_view import (
    build_simple_consensus_summary_df,
    collapse_consensus_for_map,
    infer_consensus_time_dim,
    nside_from_npix,
)

logger = logging.getLogger("TOAD")


class _ConsensusInputStore:
    """Minimal TOAD-like container for member-support consensus on MMA exports."""

    def __init__(self, data: xr.Dataset, time_dim: str):
        self.data = data
        self.time_dim = time_dim


TimeAlignment = Literal["union", "intersection", "strict"]


def _parse_source_paths(value: Any) -> list[str]:
    """Parse source export paths stored in netCDF attrs."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(p) for p in value]
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(p) for p in parsed]
        return [text]
    return [str(value)]


def _infer_time_dim(clusters: xr.DataArray) -> str:
    dim = infer_consensus_time_dim(clusters)
    if dim is None:
        return str(clusters.dims[0])
    return dim


def _merge_time_coords(
    time_coords: list[xr.DataArray],
    time_dim: str,
    alignment: TimeAlignment,
) -> xr.DataArray:
    """Build a shared time coordinate across MMA exports."""
    if not time_coords:
        raise ValueError("time_coords must not be empty.")
    if alignment == "strict":
        ref = time_coords[0]
        for i, tc in enumerate(time_coords[1:], start=1):
            if int(tc.sizes[time_dim]) != int(
                ref.sizes[time_dim]
            ) or not np.array_equal(np.asarray(tc.values), np.asarray(ref.values)):
                raise ValueError(
                    f"Export time axis {i} is incompatible with export 0 under "
                    f"time_alignment='strict'. Use time_alignment='union' or "
                    "'intersection' when models have different time ranges."
                )
        return ref

    if alignment == "union":
        common = time_coords[0]
        for tc in time_coords[1:]:
            common = (
                xr.concat([common, tc], dim=time_dim)
                .drop_duplicates(time_dim)
                .sortby(time_dim)
            )
        return common

    common = time_coords[0]
    for tc in time_coords[1:]:
        common, _ = xr.align(common, tc, join="inner")
    if int(common.sizes[time_dim]) == 0:
        raise ValueError(
            "Time intersection across MMA exports is empty. Check that exports "
            "use a shared calendar (e.g. calendar years, not per-model year 0)."
        )
    return common


def _align_labels_to_time(
    labels: np.ndarray,
    time_coord: xr.DataArray,
    time_dim: str,
    common_time: xr.DataArray,
    spatial_dims: tuple[str, ...],
    spatial_coords: dict[str, Any] | None = None,
) -> np.ndarray:
    """Reindex cluster labels onto a shared time axis (NaN where a model has no data)."""
    coords: dict[str, Any] = {time_dim: time_coord, **(spatial_coords or {})}
    for d in spatial_dims:
        if d not in coords:
            axis = spatial_dims.index(d)
            coords[d] = np.arange(labels.shape[axis + 1])
    da = xr.DataArray(
        labels,
        dims=(time_dim, *spatial_dims),
        coords=coords,
    )
    aligned = da.reindex({time_dim: common_time})
    return np.asarray(aligned.values, dtype=np.float64)


def _get_native_spatial_coords(
    ds: xr.Dataset, cluster: xr.DataArray
) -> tuple[np.ndarray, bool]:
    """Get (N, 2) spatial coords for KNN. Tries lat/lon first, then x/y.

    Returns:
        coords_2d: (n_spatial, 2) array for KNN.
        use_sphere: True if lat/lon (use spherical distance).
    """
    spatial_dims = list(cluster.dims[1:])  # skip time dim
    if len(spatial_dims) < 2:
        raise ValueError("Native cluster must have at least 2 spatial dimensions.")

    lat_name, lon_name = detect_latlon_names(ds)
    if (
        lat_name is not None
        and lon_name is not None
        and lat_name in ds
        and lon_name in ds
    ):
        c1 = np.asarray(ds[lat_name].values)
        c2 = np.asarray(ds[lon_name].values)
        use_sphere = True
    else:
        # Get coords for each spatial dim (x,y or dim index)
        coords_for_dims: list[np.ndarray] = []
        for d in spatial_dims:
            if d in ds.coords:
                coords_for_dims.append(np.asarray(ds[d].values))
            elif d in ds:
                coords_for_dims.append(np.asarray(ds[d].values))
            else:
                # Fallback: use arange for this dim
                coords_for_dims.append(np.arange(cluster.sizes[d], dtype=np.float64))
        c1, c2 = coords_for_dims[0], coords_for_dims[1]
        use_sphere = False

    if c1.ndim == 1 and c2.ndim == 1:
        c2g, c1g = np.meshgrid(c2, c1)
    else:
        c1g = c1
        c2g = c2
    coords_2d = np.column_stack([c1g.ravel(), c2g.ravel()])
    return coords_2d, use_sphere


def _load_cluster_sign_array(path: str) -> np.ndarray | None:
    """Load per-voxel ``cluster_sign`` grid from an MMA export."""
    ds = xr.open_dataset(path)
    if "cluster_sign" not in ds:
        ds.close()
        return None
    arr = np.asarray(ds["cluster_sign"].values, dtype=np.float32)
    ds.close()
    return arr


def _load_healpix_labels(path: str, nside: int) -> tuple[np.ndarray, str, xr.DataArray]:
    """Load full spacetime cluster labels from a HealPix MMA export."""
    ds = xr.open_dataset(path)
    if "cluster" not in ds:
        ds.close()
        raise ValueError(
            f"Export {path} has no 'cluster' variable. "
            "Re-export with compute_clusters(export_for_mma=...)."
        )
    cluster = ds["cluster"]
    npix = 12 * nside**2
    if "hp_pixel" not in cluster.dims:
        ds.close()
        raise ValueError(
            f"Export {path} has native-format cluster (no hp_pixel dim). "
            "Use nside=None for MMA with native exports."
        )
    arr = np.asarray(cluster.values, dtype=np.float64)
    time_dim = str(cluster.dims[0])
    time_coord = ds[time_dim]
    ds.close()
    if arr.shape[1] != npix:
        raise ValueError(
            f"cluster has {arr.shape[1]} pixels, expected {npix} for nside={nside}."
        )
    return arr, time_dim, time_coord


def _load_ever_in_healpix_masks(path: str, nside: int) -> list[tuple[int, np.ndarray]]:
    """Load cluster data from HealPix export and compute ever-in masks."""
    ds = xr.open_dataset(path)
    if "cluster" not in ds:
        ds.close()
        raise ValueError(
            f"Export {path} has no 'cluster' variable. "
            "Re-export with compute_clusters(export_for_mma=...)."
        )
    cluster = ds["cluster"]
    npix = 12 * nside**2
    if "hp_pixel" not in cluster.dims:
        ds.close()
        raise ValueError(
            f"Export {path} has native-format cluster (no hp_pixel dim). "
            "Use nside=None for MMA with native exports."
        )
    arr = np.asarray(cluster.values, dtype=np.float64)
    ds.close()
    if arr.shape[1] != npix:
        raise ValueError(
            f"cluster has {arr.shape[1]} pixels, expected {npix} for nside={nside}."
        )
    result: list[tuple[int, np.ndarray]] = []
    valid = (arr >= 0) & np.isfinite(arr)
    unique_cids = np.unique(arr[valid].astype(np.int64))
    for cid in unique_cids:
        mask = ((arr == cid) & valid).any(axis=0)
        if mask.any():
            result.append((int(cid), mask))
    return result


def _make_consensus_dataarrays(
    labels: np.ndarray,
    consistency: np.ndarray,
    dims: List[str],
    coords: dict,
    shared_attrs: dict,
    cluster_desc: str,
    consistency_desc: str,
    extra_attrs: Optional[dict] = None,
    sign_values: np.ndarray | None = None,
) -> (
    tuple[xr.DataArray, xr.DataArray] | tuple[xr.DataArray, xr.DataArray, xr.DataArray]
):
    """Build consensus_clusters and companion rate DataArrays with shared structure."""
    rate_name = "consensus_clusters_rate"
    base_attrs = {**shared_attrs, **(extra_attrs or {})}
    da_clusters = xr.DataArray(
        labels,
        dims=dims,
        coords=coords,
        name="consensus_clusters",
        attrs={
            **base_attrs,
            "description": cluster_desc,
            _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_CLUSTER,
        },
    )
    da_consistency = xr.DataArray(
        consistency,
        dims=dims,
        coords=coords,
        name=rate_name,
        attrs={
            **base_attrs,
            "description": consistency_desc,
            _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_RATE,
            _attrs.CONSENSUS_LABELS_VAR: "consensus_clusters",
        },
    )
    if sign_values is None:
        return da_clusters, da_consistency
    da_sign = xr.DataArray(
        sign_values,
        dims=dims,
        coords=coords,
        name="consensus_clusters_sign",
        attrs={
            **base_attrs,
            "description": "Per-voxel shift sign (+1/-1) for consensus cluster labels",
            _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_SIGN,
            _attrs.CONSENSUS_LABELS_VAR: "consensus_clusters",
        },
    )
    return da_clusters, da_consistency, da_sign


def _load_native_labels(
    path: str,
) -> tuple[np.ndarray, str, xr.DataArray, np.ndarray, bool, list[str], tuple[int, ...]]:
    """Load full spacetime cluster labels from a native MMA export."""
    ds = xr.open_dataset(path)
    if "cluster" not in ds:
        ds.close()
        raise ValueError(
            f"Export {path} has no 'cluster' variable. "
            "Re-export with compute_clusters(export_for_mma=...)."
        )
    cluster = ds["cluster"]
    if "hp_pixel" in cluster.dims:
        ds.close()
        raise ValueError(
            f"Export {path} has HealPix cluster. Use nside=... for MMA with HealPix exports."
        )
    arr = np.asarray(cluster.values, dtype=np.float64)
    time_dim = str(cluster.dims[0])
    time_coord = ds[time_dim]
    spatial_dims = [str(d) for d in cluster.dims[1:]]
    spatial_shape = tuple(cluster.sizes[d] for d in cluster.dims[1:])
    coords_2d, use_sphere = _get_native_spatial_coords(ds, cluster)
    ds.close()
    return arr, time_dim, time_coord, coords_2d, use_sphere, spatial_dims, spatial_shape


def _load_ever_in_native_masks(
    path: str,
) -> tuple[
    list[tuple[int, np.ndarray]],
    np.ndarray,
    bool,
    list[str],
    tuple[int, ...],
]:
    """Load cluster data from native export and compute ever-in masks.

    Returns:
        masks, coords_2d, use_sphere, spatial_dims, spatial_shape
    """
    ds = xr.open_dataset(path)
    if "cluster" not in ds:
        ds.close()
        raise ValueError(
            f"Export {path} has no 'cluster' variable. "
            "Re-export with compute_clusters(export_for_mma=...)."
        )
    cluster = ds["cluster"]
    if "hp_pixel" in cluster.dims:
        ds.close()
        raise ValueError(
            f"Export {path} has HealPix cluster. Use nside=... for MMA with HealPix exports."
        )
    arr = np.asarray(cluster.values, dtype=np.float64)
    # First dim is time, rest are spatial
    spatial_dims = [str(d) for d in cluster.dims[1:]]
    spatial_shape = tuple(cluster.sizes[d] for d in cluster.dims[1:])
    coords_2d, use_sphere = _get_native_spatial_coords(ds, cluster)
    ds.close()

    result: list[tuple[int, np.ndarray]] = []
    cluster_flat = arr.reshape(arr.shape[0], -1)
    valid = (cluster_flat >= 0) & np.isfinite(cluster_flat)
    unique_cids = np.unique(cluster_flat[valid].astype(np.int64))
    for cid in unique_cids:
        mask = ((cluster_flat == cid) & valid).any(axis=0)
        if mask.any():
            result.append((int(cid), mask))
    return result, coords_2d, use_sphere, spatial_dims, spatial_shape


class MMA:
    """Multi-model aggregation: consensus clustering across cluster label files.

    Loads cluster label files (HealPix or native format) exported via
    ``compute_clusters(export_for_mma=...)`` and runs consensus clustering.
    Supports both hp_pixel (HealPix) and native (x,y or lat/lon) grids.
    """

    def __init__(
        self,
        paths: List[str],
        nside: Optional[int] = 32,
        time_alignment: TimeAlignment = "union",
    ):
        """Load cluster label files for consensus.

        Args:
            paths: List of paths to netCDF files with cluster labels
                (from compute_clusters(export_for_mma=...)).
            nside: HealPix nside for HealPix exports; use None for native exports
                (x,y or lat/lon grids). Format is auto-detected from the first file.
            time_alignment: How to align differing time axes across exports.
                ``\"union\"`` (default): all timesteps from any model; missing steps
                are NaN (no shift). ``\"intersection\"``: only shared timesteps.
                ``\"strict\"``: require identical time coordinates.

        Raises:
            ValueError: If no paths, or if format/nside mismatch.

        See Also:
            :meth:`from_consensus` to reload a saved consensus result for plotting.
        """
        self._init_empty_state()
        if not paths:
            raise ValueError("paths must not be empty.")
        self.paths = [str(p) for p in paths]
        self.nside = nside
        self.time_alignment = time_alignment
        self._load_all()

    @classmethod
    def from_consensus(
        cls,
        path: str,
        *,
        source_paths: list[str] | None = None,
        nside: int | None = None,
        load_exports: bool = True,
    ) -> "MMA":
        """Reload a saved MMA consensus netCDF for plotting and inspection.

        Use after ``mma.data.to_netcdf(...)`` when you do not want to re-run
        :meth:`run_consensus`. Per-model export files are still required for
        :meth:`cluster_occurrence_rate` and shift-time helpers unless
        ``load_exports=False``.

        Args:
            path: Path to a netCDF file with ``consensus_clusters`` and
                ``consensus_clusters_rate`` (from :meth:`run_consensus`).
            source_paths: Per-model export paths. If omitted, read from file
                attrs (``source_paths`` on the dataset or consensus variables).
            nside: HEALPix nside for HealPix consensus. Inferred from attrs or
                ``hp_pixel`` size when omitted.
            load_exports: If True and export paths are known, load ever-in masks
                from exports for occurrence-rate and shift-time methods.

        Returns:
            An :class:`MMA` instance with :attr:`~MMA.data` populated; consensus
            does not need to be run again.
        """
        obj = cls.__new__(cls)
        obj._init_empty_state()
        obj._hydrate_from_consensus_file(
            path,
            source_paths=source_paths,
            nside=nside,
            load_exports=load_exports,
        )
        return obj

    def _init_empty_state(self) -> None:
        self.paths: list[str] = []
        self.nside: Optional[int] = 32
        self.time_alignment: TimeAlignment = "union"
        self._cluster_masks: list[list[tuple[int, np.ndarray]]] = []
        self._label_arrays: list[np.ndarray] = []
        self._sign_arrays: list[np.ndarray | None] = []
        self._cluster_var_names: list[str] = []
        self._time_dim: str = "time"
        self._time_coord: xr.DataArray | None = None
        self._format: str = "healpix"
        self._native_coords_2d: Optional[np.ndarray] = None
        self._native_use_sphere: bool = False
        self._native_spatial_dims: Optional[List[str]] = None
        self._native_spatial_shape: Optional[tuple[int, ...]] = None
        self._data: xr.Dataset | None = None
        self._plot_view: MMAPlotView | None = None

    @property
    def plot(self) -> "MMAPlotView":
        """Plotting helpers with the same API as :attr:`TOAD.plot`."""
        if self._data is None:
            raise AttributeError(
                "Run run_consensus() first, or load a consensus file via from_consensus()."
            )
        if self._plot_view is None:
            from toad.plotting.consensus_overview import MMAPlotView

            self._plot_view = MMAPlotView(self)
        return self._plot_view

    def _hydrate_from_consensus_file(
        self,
        path: str,
        *,
        source_paths: list[str] | None,
        nside: int | None,
        load_exports: bool,
    ) -> None:
        ds = xr.open_dataset(path)
        if "consensus_clusters" not in ds or "consensus_clusters_rate" not in ds:
            ds.close()
            raise ValueError(
                f"{path} is not an MMA consensus file. Expected variables "
                "'consensus_clusters' and 'consensus_clusters_rate'. "
                "Use MMA(paths, nside=...) with per-model export files instead."
            )

        clusters = ds["consensus_clusters"]
        self._data = ds
        self._time_dim = _infer_time_dim(clusters)
        self._time_coord = clusters.coords[self._time_dim]

        if "hp_pixel" in clusters.dims:
            self._format = "healpix"
            npix = int(clusters.sizes["hp_pixel"])
            resolved_nside = nside
            if resolved_nside is None:
                for candidate in (
                    ds.attrs.get("nside"),
                    clusters.attrs.get("nside"),
                    ds.attrs.get("NSIDE"),
                ):
                    if candidate is not None:
                        resolved_nside = int(candidate)
                        break
            if resolved_nside is None:
                resolved_nside = nside_from_npix(npix)
            if 12 * int(resolved_nside) ** 2 != npix:
                raise ValueError(
                    f"HEALPix nside={resolved_nside} implies npix="
                    f"{12 * int(resolved_nside) ** 2}, but file has npix={npix}."
                )
            self.nside = int(resolved_nside)
        else:
            self._format = str(ds.attrs.get("format", "native"))
            self.nside = None
            self._native_spatial_dims = [
                str(d) for d in clusters.dims if d != self._time_dim
            ]
            self._native_spatial_shape = tuple(
                int(clusters.sizes[d]) for d in self._native_spatial_dims
            )
            lat_name, lon_name = detect_latlon_names(ds)
            if (
                lat_name is not None
                and lon_name is not None
                and lat_name in ds
                and lon_name in ds
            ):
                c1 = np.asarray(ds[lat_name].values)
                c2 = np.asarray(ds[lon_name].values)
                self._native_use_sphere = True
                if c1.ndim == 1 and c2.ndim == 1:
                    c2g, c1g = np.meshgrid(c2, c1)
                else:
                    c1g, c2g = c1, c2
                self._native_coords_2d = np.column_stack([c1g.ravel(), c2g.ravel()])

        paths_attr = source_paths
        if paths_attr is None:
            for candidate in (
                ds.attrs.get("source_paths"),
                clusters.attrs.get("source_paths"),
                ds.attrs.get("source_paths_json"),
            ):
                parsed = _parse_source_paths(candidate)
                if parsed:
                    paths_attr = parsed
                    break
        self.paths = [str(p) for p in paths_attr or []]

        if load_exports and self.paths:
            self._load_masks_from_exports()

        logger.info(
            f"MMA: loaded consensus from {path} "
            f"({self._format}, n_models={len(self.paths) or 'unknown'})"
        )

    def _load_masks_from_exports(self) -> None:
        """Load ever-in cluster masks from per-model exports (no label arrays)."""
        self._cluster_masks = []
        if self._format == "healpix":
            if self.nside is None:
                raise ValueError("HealPix MMA requires nside to load export masks.")
            for path in self.paths:
                self._cluster_masks.append(
                    _load_ever_in_healpix_masks(path, cast(int, self.nside))
                )
            return

        for path in self.paths:
            masks, coords_2d, use_sphere, spatial_dims, spatial_shape = (
                _load_ever_in_native_masks(path)
            )
            if self._native_coords_2d is None:
                self._native_coords_2d = coords_2d
                self._native_use_sphere = use_sphere
                self._native_spatial_dims = spatial_dims
                self._native_spatial_shape = spatial_shape
            elif spatial_shape != self._native_spatial_shape:
                raise ValueError(
                    f"Export {path} has spatial shape {spatial_shape}, expected "
                    f"{self._native_spatial_shape}."
                )
            self._cluster_masks.append(masks)

    def _load_all(self) -> None:
        first_path = self.paths[0]
        ds = xr.open_dataset(first_path)
        has_hp = "cluster" in ds and "hp_pixel" in ds["cluster"].dims
        ds.close()

        if has_hp:
            if self.nside is None:
                raise ValueError(
                    "First file has HealPix format. Pass nside=... for MMA."
                )
            pending_hp: list[
                tuple[str, np.ndarray, str, xr.DataArray, np.ndarray | None]
            ] = []
            for path in self.paths:
                ds = xr.open_dataset(path)
                file_nside = ds.attrs.get("nside")
                ds.close()
                if file_nside is not None and int(file_nside) != self.nside:
                    raise ValueError(
                        f"File {path} has nside={file_nside}, but MMA uses nside={self.nside}. "
                        "All files must use the same nside."
                    )
                labels, time_dim, time_coord = _load_healpix_labels(path, self.nside)
                sign_arr = _load_cluster_sign_array(path)
                pending_hp.append((path, labels, time_dim, time_coord, sign_arr))

            self._time_dim = pending_hp[0][2]
            if any(td != self._time_dim for _, _, td, _, _ in pending_hp):
                raise ValueError(
                    "All MMA exports must use the same time dimension name."
                )
            common_time = _merge_time_coords(
                [tc for _, _, _, tc, _ in pending_hp],
                self._time_dim,
                self.time_alignment,
            )
            self._time_coord = common_time
            npix = 12 * cast(int, self.nside) ** 2

            for path, labels, time_dim, time_coord, sign_arr in pending_hp:
                if int(time_coord.sizes[time_dim]) != int(common_time.sizes[time_dim]):
                    logger.info(
                        f"MMA: aligned {path} from {int(time_coord.sizes[time_dim])} to "
                        f"{int(common_time.sizes[time_dim])} time steps "
                        f"(time_alignment={self.time_alignment!r})"
                    )
                aligned = _align_labels_to_time(
                    labels,
                    time_coord,
                    time_dim,
                    common_time,
                    spatial_dims=("hp_pixel",),
                    spatial_coords={"hp_pixel": np.arange(npix)},
                )
                aligned_sign = None
                if sign_arr is not None:
                    aligned_sign = _align_labels_to_time(
                        sign_arr,
                        time_coord,
                        time_dim,
                        common_time,
                        spatial_dims=("hp_pixel",),
                        spatial_coords={"hp_pixel": np.arange(npix)},
                    )
                var_name = f"mma_model_{len(self._label_arrays)}_cluster"
                self._label_arrays.append(aligned)
                self._sign_arrays.append(aligned_sign)
                self._cluster_var_names.append(var_name)
                self._cluster_masks.append(
                    _load_ever_in_healpix_masks(path, self.nside)
                )
            self._format = "healpix"
            logger.info(
                f"MMA: loaded {len(self._cluster_masks)} HealPix cluster file(s), "
                f"nside={self.nside}, T={int(common_time.sizes[self._time_dim])}"
            )
        else:
            if self.nside is not None:
                raise ValueError(
                    "First file has native format. Use nside=None for MMA with native exports."
                )
            ref_shape: Optional[tuple[int, ...]] = None
            pending_native: list[
                tuple[
                    str,
                    np.ndarray,
                    str,
                    xr.DataArray,
                    np.ndarray,
                    bool,
                    list[str],
                    tuple[int, ...],
                    np.ndarray | None,
                ]
            ] = []
            for path in self.paths:
                (
                    labels,
                    time_dim,
                    time_coord,
                    coords_2d,
                    use_sphere,
                    spatial_dims,
                    spatial_shape,
                ) = _load_native_labels(path)
                sign_arr = _load_cluster_sign_array(path)
                if ref_shape is None:
                    ref_shape = spatial_shape
                    self._native_coords_2d = coords_2d
                    self._native_use_sphere = use_sphere
                    self._native_spatial_dims = spatial_dims
                    self._native_spatial_shape = spatial_shape
                elif spatial_shape != ref_shape:
                    raise ValueError(
                        f"File {path} has spatial shape {spatial_shape}, "
                        f"expected {ref_shape}. All native files must have the same grid."
                    )
                pending_native.append(
                    (
                        path,
                        labels,
                        time_dim,
                        time_coord,
                        coords_2d,
                        use_sphere,
                        spatial_dims,
                        spatial_shape,
                        sign_arr,
                    )
                )

            self._time_dim = pending_native[0][2]
            if any(
                td != self._time_dim for _, _, td, _, _, _, _, _, _ in pending_native
            ):
                raise ValueError(
                    "All MMA exports must use the same time dimension name."
                )
            common_time = _merge_time_coords(
                [tc for _, _, _, tc, _, _, _, _, _ in pending_native],
                self._time_dim,
                self.time_alignment,
            )
            self._time_coord = common_time
            spatial_dims_tuple = tuple(cast(List[str], self._native_spatial_dims))
            ds0 = xr.open_dataset(self.paths[0])
            spatial_coords = {d: ds0[d] for d in spatial_dims_tuple if d in ds0}
            ds0.close()

            for (
                path,
                labels,
                time_dim,
                time_coord,
                _coords_2d,
                _use_sphere,
                spatial_dims,
                _spatial_shape,
                sign_arr,
            ) in pending_native:
                if int(time_coord.sizes[time_dim]) != int(common_time.sizes[time_dim]):
                    logger.info(
                        f"MMA: aligned {path} from {int(time_coord.sizes[time_dim])} to "
                        f"{int(common_time.sizes[time_dim])} time steps "
                        f"(time_alignment={self.time_alignment!r})"
                    )
                aligned = _align_labels_to_time(
                    labels,
                    time_coord,
                    time_dim,
                    common_time,
                    spatial_dims=tuple(spatial_dims),
                    spatial_coords=spatial_coords,
                )
                aligned_sign = None
                if sign_arr is not None:
                    aligned_sign = _align_labels_to_time(
                        sign_arr,
                        time_coord,
                        time_dim,
                        common_time,
                        spatial_dims=tuple(spatial_dims),
                        spatial_coords=spatial_coords,
                    )
                var_name = f"mma_model_{len(self._label_arrays)}_cluster"
                self._label_arrays.append(aligned)
                self._sign_arrays.append(aligned_sign)
                self._cluster_var_names.append(var_name)
                masks, _, _, _, _ = _load_ever_in_native_masks(path)
                self._cluster_masks.append(masks)
            self._format = "native"
            logger.info(
                f"MMA: loaded {len(self._cluster_masks)} native cluster file(s), "
                f"T={int(common_time.sizes[self._time_dim])}"
            )

    def cluster_occurrence_rate(self) -> xr.DataArray:
        """Calculate the normalized occurrence rate of points being part of any cluster.

        For each spatial point, computes the fraction of models (export files) where that
        point was ever part of a cluster (not noise), i.e. how often it was assigned to
        a cluster label >= 0 across time. Values range from 0 (never in a cluster) to 1
        (always in a cluster in every model).

        Returns:
            DataArray with the same spatial structure as the MMA grid (HealPix or native),
            with values in [0, 1].

        Example:
            >>> mma = MMA(paths, nside=None)
            >>> rate = mma.cluster_occurrence_rate()
            >>> rate.plot()
        """
        n_models = len(self._cluster_masks)
        if n_models == 0:
            raise ValueError("No cluster data loaded.")

        # Per-model: ever_in_cluster[p] = True if point p was in any cluster
        if self._format == "healpix":
            npix = 12 * cast(int, self.nside) ** 2
            n_spatial = npix
            ever_in = np.zeros(n_spatial, dtype=np.float64)
            for masks in self._cluster_masks:
                combined = np.zeros(n_spatial, dtype=bool)
                for _cid, mask in masks:
                    combined |= mask
                ever_in += combined.astype(np.float64)
            rate = ever_in / n_models
            dims = ["hp_pixel"]
            coords: dict = {"hp_pixel": np.arange(npix)}
        else:
            spatial_shape = cast(tuple[int, ...], self._native_spatial_shape)
            n_spatial = int(np.prod(spatial_shape))
            ever_in = np.zeros(n_spatial, dtype=np.float64)
            for masks in self._cluster_masks:
                combined = np.zeros(n_spatial, dtype=bool)
                for _cid, mask in masks:
                    combined |= mask
                ever_in += combined.astype(np.float64)
            rate = (ever_in / n_models).reshape(spatial_shape)
            spatial_dims = cast(List[str], self._native_spatial_dims)
            dims = spatial_dims
            ds0 = xr.open_dataset(self.paths[0])
            coords = {d: ds0[d] for d in spatial_dims if d in ds0}
            ds0.close()

        da = xr.DataArray(
            rate,
            dims=dims,
            coords=coords,
            name="cluster_occurrence_rate",
            attrs={
                "description": "Normalized occurrence rate of points being part of any cluster across models",
                "n_models": n_models,
                _attrs.METHOD_NAME: "mma_cluster_occurrence_rate",
                _attrs.TOAD_VERSION: __version__,
            },
        )
        return da

    def _build_consensus_inputs(self) -> _ConsensusInputStore:
        from toad.postprocessing.member_support_consensus import (
            sign_var_for_cluster_var,
        )

        coords = {self._time_dim: cast(xr.DataArray, self._time_coord)}
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        if self._format == "healpix":
            npix = 12 * cast(int, self.nside) ** 2
            coords["hp_pixel"] = np.arange(npix)
            for name, labels, sign_arr in zip(
                self._cluster_var_names,
                self._label_arrays,
                self._sign_arrays,
            ):
                data_vars[name] = ((self._time_dim, "hp_pixel"), labels)
                if sign_arr is not None:
                    data_vars[sign_var_for_cluster_var(name)] = (
                        (self._time_dim, "hp_pixel"),
                        sign_arr,
                    )
        else:
            spatial_dims = cast(List[str], self._native_spatial_dims)
            ds0 = xr.open_dataset(self.paths[0])
            for d in spatial_dims:
                if d in ds0:
                    coords[d] = ds0[d]
            ds0.close()
            for name, labels, sign_arr in zip(
                self._cluster_var_names,
                self._label_arrays,
                self._sign_arrays,
            ):
                data_vars[name] = ((self._time_dim, *spatial_dims), labels)
                if sign_arr is not None:
                    data_vars[sign_var_for_cluster_var(name)] = (
                        (self._time_dim, *spatial_dims),
                        sign_arr,
                    )

        ds = xr.Dataset(
            {
                name: (dims, arr.astype(np.float32))
                for name, (dims, arr) in data_vars.items()
            },
            coords=coords,
        )
        return _ConsensusInputStore(ds, self._time_dim)

    def run_consensus(
        self,
        min_consensus: float = 0.5,
        temporal_tolerance: int = 0,
        spatial_tolerance: int | float = 1,
        min_cluster_area: int | None = 2,
        k_neighbors: int = 8,
        show_progress: bool = True,
    ) -> xr.Dataset:
        """Run member-support consensus and store results in self.data.

        Args:
            min_consensus: Minimum fraction of models required per retained voxel.
            temporal_tolerance: Time-step radius for support dilation and connectivity.
            spatial_tolerance: HEALPix-hop or native-grid-cell radius for dilation
                and connectivity. Use ``np.inf`` for unlimited spatial reach
                (temporal agreement only).
            min_cluster_area: Minimum distinct spatial footprint for a consensus cluster.
                Use ``None`` to disable.
            k_neighbors: Deprecated; ignored. HEALPix consensus uses ring-1 neighbours.
            show_progress: Whether to show a progress bar.

        Returns:
            The internal dataset with consensus_clusters, consensus_clusters_rate,
            and consensus_clusters_sign when sign-aware inputs are present.
        """
        from toad import TOAD
        from toad.postprocessing.healpix_member_support_consensus import (
            run_healpix_member_support_consensus,
        )

        from toad.postprocessing.member_support_consensus import (
            spatial_tolerance_attr_value,
            validate_spatial_tolerance,
        )

        spatial_tolerance = validate_spatial_tolerance(spatial_tolerance)
        inputs = self._build_consensus_inputs()
        shared_attrs = {
            "consensus_method": "member_support",
            "min_consensus": min_consensus,
            "temporal_tolerance": temporal_tolerance,
            "spatial_tolerance": spatial_tolerance_attr_value(spatial_tolerance),
            "min_cluster_area": min_cluster_area,
            "k_neighbors": k_neighbors,
            "n_models": len(self._label_arrays),
            "source_paths": self.paths,
            "cluster_vars": self._cluster_var_names,
            _attrs.METHOD_NAME: "mma_member_support_consensus",
            _attrs.TOAD_VERSION: __version__,
        }

        if self._format == "healpix":
            ds_out = run_healpix_member_support_consensus(
                inputs,
                cluster_vars=self._cluster_var_names,
                min_consensus=min_consensus,
                temporal_tolerance=temporal_tolerance,
                spatial_tolerance=spatial_tolerance,
                nside=cast(int, self.nside),
                k_neighbors=k_neighbors,
                min_cluster_area=min_cluster_area,
                show_progress=show_progress,
            )
            labels = ds_out["clusters"].values
            rate = ds_out["rate"].values
            signs = ds_out["signs"].values if "signs" in ds_out else None
            npix = 12 * cast(int, self.nside) ** 2
            cluster_extra: dict[str, Any] = {}
            u = np.unique(labels[np.isfinite(labels) & (labels >= 0)]).astype(int)
            if u.size:
                cluster_extra[_attrs.CLUSTER_IDS] = u
            consensus_arrays = _make_consensus_dataarrays(
                labels,
                rate,
                dims=[self._time_dim, "hp_pixel"],
                coords={
                    self._time_dim: cast(xr.DataArray, self._time_coord),
                    "hp_pixel": np.arange(npix),
                },
                shared_attrs=shared_attrs,
                cluster_desc="Spacetime member-support consensus on HealPix.",
                consistency_desc="Member-support rate per HealPix pixel and time.",
                extra_attrs={"nside": self.nside, **cluster_extra},
                sign_values=signs,
            )
            self._data = xr.merge(
                list(consensus_arrays),
                combine_attrs="override",
                compat="override",
            )
            self._data.attrs["nside"] = self.nside
        else:
            td = TOAD(inputs.data, time_dim=self._time_dim, log_level="CRITICAL")
            td.compute_consensus(
                cluster_vars=self._cluster_var_names,
                min_consensus=min_consensus,
                temporal_tolerance=temporal_tolerance,
                spatial_tolerance=spatial_tolerance,
                min_cluster_area=min_cluster_area,
                show_progress=show_progress,
            )
            consensus_var = td.consensus_cluster_vars[-1]
            rate_var = td.consensus_rate_var_name(consensus_var)
            sign_var = f"{consensus_var}{_attrs.CONSENSUS_SIGN_SUFFIX}"
            consensus_da = td.data[consensus_var]
            labels = consensus_da.values
            rate = td.data[rate_var].values
            signs = td.data[sign_var].values if sign_var in td.data else None
            spatial_dims = cast(List[str], self._native_spatial_dims)
            ds0 = xr.open_dataset(self.paths[0])
            coords = {
                self._time_dim: cast(xr.DataArray, self._time_coord),
                **{d: ds0[d] for d in spatial_dims if d in ds0},
            }
            ds0.close()
            cluster_extra = {}
            if _attrs.CLUSTER_IDS in consensus_da.attrs:
                cluster_extra[_attrs.CLUSTER_IDS] = consensus_da.attrs[
                    _attrs.CLUSTER_IDS
                ]
            consensus_arrays = _make_consensus_dataarrays(
                labels,
                rate,
                dims=[self._time_dim, *spatial_dims],
                coords=coords,
                shared_attrs=shared_attrs,
                cluster_desc="Spacetime member-support consensus on native grid.",
                consistency_desc="Member-support rate per grid cell and time.",
                extra_attrs=cluster_extra or None,
                sign_values=signs,
            )
            self._data = xr.merge(
                list(consensus_arrays),
                combine_attrs="override",
                compat="override",
            )

        self._data.attrs["format"] = self._format
        self._data.attrs["source_paths"] = json.dumps(self.paths)

        labels_arr = self._data["consensus_clusters"].values
        n_clusters = len(self.consensus_cluster_ids())
        logger.info(
            f"MMA member-support consensus: {n_clusters} clusters, "
            f"noise={int(np.sum(labels_arr == -1))}, "
            f"nan={int(np.sum(np.isnan(labels_arr)))}"
        )
        return self._data

    @property
    def data(self) -> xr.Dataset:
        """Internal dataset with consensus_clusters, rate, and optional sign grid."""
        if self._data is None:
            raise AttributeError(
                "Run run_consensus() first to compute and store results."
            )
        return self._data

    @data.setter
    def data(self, value: xr.Dataset) -> None:
        self._data = value

    def consensus_cluster_ids(self) -> list[int]:
        """Return sorted consensus cluster IDs in ``consensus_clusters``.

        Prefer ``cluster_ids`` stored on the data array (O(1)); otherwise scan
        distinct non-negative, finite labels (excludes noise ``-1`` and NaN).
        """
        da = self.data["consensus_clusters"]
        stored = da.attrs.get(_attrs.CLUSTER_IDS)
        if stored is not None:
            ids = np.asarray(stored, dtype=int)
            ids = ids[ids >= 0]
            return sorted(int(x) for x in ids)
        clusters = da.values
        ids = np.unique(clusters[(clusters >= 0) & np.isfinite(clusters)])
        return [int(x) for x in ids]

    def get_healpix_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """Get (lat, lon) in degrees for each HealPix pixel, for plotting with cartopy etc.

        Returns:
            lats, lons: 1D arrays of length npix (12 * nside**2).

        Raises:
            AttributeError: If run_consensus() has not been called yet.
            ValueError: If MMA uses native format (use data coords instead).
        """
        if self._format != "healpix":
            raise ValueError(
                "get_healpix_latlon is for HealPix format. "
                "For native format, use data['consensus_clusters'] coords."
            )
        regridder = HealPixRegridder(nside=cast(int, self.nside))
        npix = 12 * cast(int, self.nside) ** 2
        return regridder.pixels_to_latlon(np.arange(npix))

    def map_consensus_to_coords(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Assign consensus cluster ID to each (lat, lon) by HealPix index lookup.

        HealPix format only. For native format, use map_consensus_to_dataset.

        Args:
            lat: Latitude(s) in degrees. Any shape.
            lon: Longitude(s) in degrees. Same shape as lat.

        Returns:
            Array of same shape as lat/lon with consensus cluster IDs.
        """
        if self._format != "healpix":
            raise ValueError(
                "map_consensus_to_coords is for HealPix format. "
                "For native format, use map_consensus_to_dataset."
            )
        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have the same shape")
        regridder = HealPixRegridder(nside=cast(int, self.nside))
        coords_2d = np.column_stack([lat.ravel(), lon.ravel()])
        hp_indices = regridder.map_orig_to_regrid(coords_2d)
        clusters = collapse_consensus_for_map(self.data["consensus_clusters"])
        return clusters[hp_indices].reshape(lat.shape)

    def map_consensus_to_dataset(self, ds: xr.Dataset) -> xr.DataArray:
        """Assign consensus cluster IDs to an xarray Dataset's spatial grid.

        Uses lat/lon (HealPix) or same grid (native) from the dataset.

        Args:
            ds: Dataset with lat/lon (HealPix) or matching spatial dims (native).

        Returns:
            DataArray of consensus cluster IDs with same spatial dims as ds.
        """
        if self._format == "healpix":
            lat_name, lon_name = detect_latlon_names(ds)
            if lat_name is None or lon_name is None:
                raise ValueError(
                    "Dataset must have lat/lon coordinates for HealPix MMA lookup."
                )
            lat = ds[lat_name].values
            lon = ds[lon_name].values
            if lat.ndim == 1 and lon.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
                dims = list(ds[lat_name].dims) + list(ds[lon_name].dims)
            else:
                dims = list(ds[lat_name].dims)
            cluster_ids = self.map_consensus_to_coords(lat, lon)
        else:
            cc = self.data["consensus_clusters"]
            for d in cc.dims:
                if d not in ds:
                    raise ValueError(
                        f"Dataset must have dimension '{d}' for native MMA (same grid)."
                    )
            cluster_ids = np.asarray(cc.values)
            dims = list(cc.dims)
        coords = {d: ds[d] for d in dims if d in ds}
        return xr.DataArray(
            cluster_ids,
            dims=dims,
            coords=coords,
            name="consensus_cluster_id",
            attrs={"description": "Consensus cluster ID per grid cell"},
        )

    def _consensus_tolerances(self) -> tuple[int, int | float]:
        """Return ``(temporal_tolerance, spatial_tolerance)`` stored with consensus.

        ``spatial_tolerance`` may be ``np.inf`` when consensus used unlimited
        spatial reach.
        """
        from toad.postprocessing.member_support_consensus import (
            validate_spatial_tolerance,
        )

        for source in (
            self.data["consensus_clusters"].attrs,
            self.data["consensus_clusters_rate"].attrs,
            self.data.attrs,
        ):
            temporal = source.get("temporal_tolerance")
            spatial = source.get("spatial_tolerance")
            if temporal is not None and spatial is not None:
                return int(temporal), validate_spatial_tolerance(spatial)
        return 0, 1

    def _dilate_healpix_model_support(self, event_mask: np.ndarray) -> np.ndarray:
        """Dilate a model event mask with the same spacetime tolerances as consensus."""
        from toad.postprocessing.healpix_member_support_consensus import (
            _dilate_healpix_support_mask,
            build_healpix_spatial_edges,
        )

        temporal_tolerance, spatial_tolerance = self._consensus_tolerances()
        mask = np.asarray(event_mask, dtype=bool)
        nside = cast(int, self.nside)
        npix = mask.shape[1]
        if 12 * nside**2 != npix:
            raise ValueError(
                f"HEALPix event mask has npix={npix}, expected {12 * nside**2} for nside={nside}."
            )
        spatial_rows, spatial_cols = build_healpix_spatial_edges(nside)
        return _dilate_healpix_support_mask(
            mask,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            spatial_rows=spatial_rows,
            spatial_cols=spatial_cols,
        )

    def _dilate_native_model_support(self, event_mask: np.ndarray) -> np.ndarray:
        """Dilate a native-grid model event mask with consensus tolerances."""
        from toad.postprocessing.member_support_consensus import (
            _dilate_boolean_support_mask,
        )

        temporal_tolerance, spatial_tolerance = self._consensus_tolerances()
        stitch = bool(self.data.attrs.get("stitch_meridian", True))
        return _dilate_boolean_support_mask(
            np.asarray(event_mask, dtype=bool),
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            stitch_meridian=stitch,
        )

    def get_member_support_agreement_from_export(
        self,
        export_path: str,
        consensus_cluster_id: int,
    ) -> np.ndarray | None:
        """Return a bool mask where consensus ``cid`` meets dilated model support.

        HEALPix exports return ``(time, hp_pixel)``; native exports return the
        cluster array shape ``(time, …spatial…)``. Agreement uses the same
        spacetime dilation as member-support consensus voting, not exact voxel
        co-location of cluster labels.
        """
        ds = xr.open_dataset(export_path)
        try:
            if "cluster" not in ds:
                return None
            clusters = ds["cluster"]
            time_dims = [d for d in clusters.dims if d in ds.coords]
            if not time_dims:
                return None
            time_dim = time_dims[0]
            consensus = self.data["consensus_clusters"]

            if "hp_pixel" in clusters.dims:
                cluster_da = clusters.transpose(time_dim, "hp_pixel")
                if self._time_coord is not None and self._time_dim in consensus.dims:
                    cluster_da = cluster_da.reindex({time_dim: self._time_coord})
                    consensus_vals = consensus.transpose(
                        self._time_dim, "hp_pixel"
                    ).values
                elif self._time_dim in consensus.dims:
                    consensus_vals = consensus.transpose(
                        self._time_dim, "hp_pixel"
                    ).values
                else:
                    consensus_vals = consensus.values
                    consensus_vals = consensus_vals[np.newaxis, :]
                arr = cluster_da.values
                region_mask = (consensus_vals == consensus_cluster_id) & np.isfinite(
                    consensus_vals
                )
                in_cluster = (arr != -1) & np.isfinite(arr)
                model_support = self._dilate_healpix_model_support(in_cluster)
                return region_mask & model_support

            cluster_da = clusters
            if self._time_coord is not None and self._time_dim in consensus.dims:
                cluster_da = cluster_da.reindex({time_dim: self._time_coord})
                region_mask = (
                    (consensus == consensus_cluster_id) & consensus.notnull()
                ).values
            else:
                consensus_ids = self.map_consensus_to_dataset(ds)
                region_mask = (
                    (consensus_ids == consensus_cluster_id) & consensus_ids.notnull()
                ).values
            in_cluster = ((cluster_da != -1) & cluster_da.notnull()).values
            if region_mask.shape != in_cluster.shape:
                region_mask = np.broadcast_to(region_mask, in_cluster.shape)
            model_support = self._dilate_native_model_support(in_cluster)
            return region_mask & model_support
        finally:
            ds.close()

    def _shift_times_from_agreement_mask(
        self,
        combined: np.ndarray,
        time_coord: xr.DataArray,
        time_dim: str,
        cluster_dims: tuple[str, ...],
    ) -> np.ndarray:
        time_axis = cluster_dims.index(time_dim)
        broadcast_shape = [1] * combined.ndim
        broadcast_shape[time_axis] = int(time_coord.sizes[time_dim])
        event_times = np.broadcast_to(
            np.asarray(time_coord.values).reshape(broadcast_shape),
            combined.shape,
        )[combined]
        return event_times

    def get_shift_times_from_export(
        self,
        export_path: str,
        consensus_cluster_id: int,
        numeric: bool = True,
    ) -> np.ndarray:
        """Extract shift times for a consensus cluster from an MMA export file.

        The export must contain the ``cluster`` variable (time × space), written
        when using ``export_for_mma``. Space is either hp_pixel (HealPix) or native dims.

        Shift times are recorded at each consensus spacetime cell where the
        model's **dilated** cluster support (same ``temporal_tolerance`` /
        ``spatial_tolerance`` as consensus) covers that cell — matching the
        member-support voting rule, not strict label co-location.

        Args:
            export_path: Path to the per-model export file (from compute_clusters).
            consensus_cluster_id: Consensus cluster ID (e.g. 0, 1, 2).
            numeric: If True, return numeric time values. If False, return native times.

        Returns:
            Flattened array of time values for events in the consensus region.
        """
        combined = self.get_member_support_agreement_from_export(
            export_path, consensus_cluster_id
        )
        if combined is None or not np.any(combined):
            return np.array([])

        ds = xr.open_dataset(export_path)
        try:
            clusters = ds["cluster"]
            time_dims = [d for d in clusters.dims if d in ds.coords]
            time_dim = time_dims[0]
            time_coord = ds[time_dim]
            if self._time_coord is not None:
                time_coord = self._time_coord
            event_times = self._shift_times_from_agreement_mask(
                combined,
                time_coord,
                time_dim,
                tuple(clusters.dims),
            )
        finally:
            ds.close()

        if numeric:
            out = event_times[np.isfinite(event_times)]
        else:
            out = event_times[event_times == event_times]
        return np.asarray(out)

    def get_shift_times_per_consensus_cluster(
        self,
        numeric: bool = True,
    ) -> Dict[int, np.ndarray]:
        """Aggregate shift times for each consensus cluster across all input exports.

        For each consensus cluster ID, collects shift times from every per-model
        export file and returns them as a single array per cluster.

        Returns:
            Dict mapping consensus cluster ID to 1D array of shift times
            (aggregated across all exports).

        Example:
            >>> times_by_cluster = mma.get_shift_times_per_consensus_cluster()
            >>> for cid, times in times_by_cluster.items():
            ...     plt.hist(times, bins=30, alpha=0.5, label=f"Cluster {cid}")
        """
        out: Dict[int, np.ndarray] = {}
        for cid in self.consensus_cluster_ids():
            times_list: List[np.ndarray] = []
            for path in self.paths:
                times_list.append(
                    self.get_shift_times_from_export(
                        path,
                        consensus_cluster_id=cid,
                        numeric=numeric,
                    )
                )
            out[cid] = np.concatenate(times_list) if times_list else np.array([])
        return out

    def get_consensus_summary(self, numeric: bool = True) -> pd.DataFrame:
        """Build a summary DataFrame for each consensus cluster.

        Uses `get_shift_times_per_consensus_cluster` and pixel-wise consensus
        rate to compute per-cluster statistics.

        Args:
            numeric: If True, shift times are numeric (for mean/std). If False, times
                are kept as native (e.g. cftime); mean_mean_shift_time and
                std_mean_shift_time will be NaN.

        Returns:
            DataFrame with columns: cluster_id, size, mean_consensus_rate,
            mean_mean_shift_time, std_mean_shift_time.
        """
        clusters = self.data["consensus_clusters"]
        rate = self.data["consensus_clusters_rate"]
        times_by_cluster = self.get_shift_times_per_consensus_cluster(numeric=numeric)
        return build_simple_consensus_summary_df(
            clusters,
            rate,
            times_by_cluster,
            time_dim=self._time_dim,
            numeric=numeric,
        )

    def plot_consensus_clusters(
        self,
        ax: Optional[Axes] = None,
        map_style: Optional[Union[Dict[str, Any], "MapStyle"]] = None,
        cmap: str = "tab10",
        s: float = 10,
        vmin: float | None = None,
        vmax: float | None = None,
        show_noise: bool = False,
        add_colorbar: bool = True,
        cluster_ids: Optional[Sequence[int]] = None,
        edge_step: int = 1,
        add_labels: bool | None = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """Plot consensus clusters on a map.

        Args:
            ax: Axes to plot on. If None, creates a new figure with map features.
            map_style: Map style (projection, coastlines, etc.). Pass a dict, e.g.
                ``{"projection": ccrs.Orthographic(-40, 15), "continent_shading": True}``
                or ``{"projection": "mollweide"}``. Used only when ax is None.
            cmap: Colormap for clusters. Defaults to "tab10".
            s: Deprecated; kept for API compatibility (HealPix maps use polygons).
            vmin: Lower bound for colour scale. Defaults to -0.5.
            vmax: Upper bound for colour scale. Defaults to 9.5.
            show_noise: If True, plot noise pixels (-1) as grey. Defaults to False.
            add_colorbar: Whether to add a colourbar. Defaults to True.
            cluster_ids: Optional sequence of cluster IDs to plot. If specified, only these clusters will be shown.
            edge_step: HEALPix polygon edge resolution (1 = cell corners).
            add_labels: If True, annotate cluster ids at median pixel centres. Defaults
                to ``map_style.add_labels`` (True).
            **kwargs: Passed through to the HEALPix polygon collection.

        Returns:
            Tuple of (figure, axes).

        Raises:
            AttributeError: If run_consensus() has not been called yet.
        """
        return plot_collapsed_consensus_labels_map(
            self.data["consensus_clusters"],
            self.data,
            time_dim=self._time_dim,
            nside=self.nside,
            ax=ax,
            map_style=map_style,
            cmap=cmap,
            s=s,
            vmin=vmin,
            vmax=vmax,
            show_noise=show_noise,
            add_colorbar=add_colorbar,
            cluster_ids=cluster_ids,
            edge_step=edge_step,
            add_labels=add_labels,
            **kwargs,
        )

    def consensus_overview(self, **kwargs: Any) -> Tuple[Figure, Axes, Axes]:
        """Two-panel consensus overview (map + shift-time panel).

        Delegates to :meth:`plot.consensus_overview` (same API as
        :meth:`Plotter.consensus_overview`).
        """
        return self.plot.consensus_overview(**kwargs)
