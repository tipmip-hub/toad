# Multi-Model Aggregation (MMA) Workflow

This document describes the full workflow for running consensus clustering across multiple models or runs using TOAD's MMA pipeline.

## Overview

1. **Per model**: Compute shifts → cluster → export spacetime cluster labels (HealPix or native) via `export_for_mma`
2. **MMA**: Load exported files → run **member-support** consensus → inspect results
3. **Shift time stats**: Extract shift time distributions per consensus cluster from exports (no original dataset needed)

MMA uses the same **member-support** algorithm as `td.compute_consensus()`:

- Native event voxels (non-noise cluster labels) are dilated in time and space for **support counting**
- A voxel is retained when enough models agree after dilation (`min_consensus`)
- Retained voxels are grouped into consensus clusters using the same tolerances for **connectivity**

Consensus output is **spacetime** (`time × space`), not time-collapsed. Map plots time-collapse for display.

---

## Step 1: Per-Model Pipeline (Export for MMA)

For each model or run, compute shifts and clusters, then export cluster labels for MMA. Use a **fixed `nside`** when exporting HealPix so all files share the same grid.

### Option A: HealPix export (recommended for mixed grids)

```python
from toad import TOAD
from toad.regridding import HealPixRegridder
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

NSIDE = 32  # Use same nside for all models

model_paths = ["model_a.nc", "model_b.nc", "model_c.nc"]
export_paths = []

for i, path in enumerate(model_paths):
    td = TOAD(path, time_dim="time")
    td.compute_shifts(td.base_vars[0], method=ASDETECT())
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=25),
        time_weight=1.0,
        shift_threshold=0.8,
        regridder=HealPixRegridder(nside=NSIDE),
        export_for_mma=f"clusters_model_{i}.nc",
        mma_grid="healpix",
    )
    export_paths.append(f"clusters_model_{i}.nc")
```

**Note:** `mma_grid='healpix'` requires `regridder=HealPixRegridder(nside=...)` and lat/lon on the source grid (including curvilinear `j/i` with 2D lat/lon). For projected grids (e.g. x,y in km), use Option B instead.

### Option B: Native export (same grid across models)

```python
from toad import TOAD
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

model_paths = ["model_a.nc", "model_b.nc", "model_c.nc"]
export_paths = []

for i, path in enumerate(model_paths):
    td = TOAD(path, time_dim="time")
    td.compute_shifts(td.base_vars[0], method=ASDETECT())
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=25),
        time_weight=1.0,
        shift_threshold=0.8,
        export_for_mma=f"clusters_model_{i}.nc",
        mma_grid="native",
    )
    export_paths.append(f"clusters_model_{i}.nc")
```

**Note:** All native exports must share the same spatial grid and time axis.

---

## Step 2: Run MMA Consensus

```python
from toad import MMA

paths = ["clusters_model_0.nc", "clusters_model_1.nc", "clusters_model_2.nc"]

mma = MMA(paths, nside=NSIDE)  # HealPix: same NSIDE as export. Native: nside=None

# Models with different time ranges: default time_alignment="union" pads missing
# timesteps with NaN (no shift). Use a shared calendar in exports (e.g. calendar years).
# time_alignment="intersection" keeps only timesteps present in every model.

ds = mma.run_consensus(
    min_consensus=0.5,
    temporal_tolerance=2,   # time-step radius for dilation and connectivity
    spatial_tolerance=1,    # HEALPix hops or native grid cells
    min_cluster_area=2,     # minimum distinct spatial footprint; None to disable
    k_neighbors=8,            # HEALPix spatial graph only
    show_progress=True,
)

# Results in mma.data
print(mma.data)
# consensus_clusters, consensus_clusters_rate
```

| Parameter | Meaning |
|-----------|---------|
| `min_consensus` | Fraction of models required per retained voxel |
| `temporal_tolerance` | Time-step radius for support dilation and cluster connectivity |
| `spatial_tolerance` | HEALPix-hop or native-grid-cell radius; ``np.inf`` = unlimited spatial reach |
| `min_cluster_area` | Drop consensus clusters with smaller spatial footprint |
| `k_neighbors` | K for the HEALPix KNN graph (ignored for native format) |

### Time alignment across models

CMIP models often have different simulation lengths. MMA aligns exports on load:

| `time_alignment` | Behaviour |
|------------------|-----------|
| `"union"` (default) | All timesteps from any model; missing steps are NaN (no shift) |
| `"intersection"` | Only timesteps present in every export |
| `"strict"` | Require identical time coordinates (old behaviour) |

```python
mma = MMA(paths, nside=32, time_alignment="union")
```

**Important:** use a **shared calendar** in exports (e.g. actual calendar years), not per-model year indices starting at 0. Otherwise timesteps align by value but not by real time.

---

## Step 3: Inspect and Plot Results

```python
import numpy as np

clusters = mma.data["consensus_clusters"]
rate = mma.data["consensus_clusters_rate"]

# Number of consensus clusters (spacetime field)
labels = clusters.values
n_clusters = len(np.unique(labels[(labels >= 0) & np.isfinite(labels)]))
print(f"Consensus: {n_clusters} clusters")

# Per-point model agreement (ever-in, time-collapsed; no run_consensus needed)
occurrence = mma.cluster_occurrence_rate()

# Summary table per consensus cluster
summary = mma.get_consensus_summary()

mma.data.to_netcdf("consensus_result.nc")
```

### Reloading a saved consensus

After saving with ``mma.data.to_netcdf(...)``, reload for plotting without re-running consensus:

```python
from toad import MMA

mma = MMA.from_consensus("consensus_result.nc")
mma.plot_consensus_clusters()

# Shift times / occurrence rate need the original per-model exports
# (paths are read from file attrs if saved after run_consensus)
times_by_cluster = mma.get_shift_times_per_consensus_cluster()
```

Pass ``source_paths=[...]`` explicitly if the netCDF attrs do not contain them.
Use ``load_exports=False`` for plotting-only reloads.

### Plotting consensus clusters

`plot_consensus_clusters` time-collapses the spacetime consensus field for map display.

```python
import cartopy.crs as ccrs

fig, ax = mma.plot_consensus_clusters(
    map_style={"projection": "mollweide", "continent_shading": True},
    s=10,
    show_noise=True,
)
```

### Plotting cluster occurrence rate

```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

rate = mma.cluster_occurrence_rate()

if mma._format == "healpix":
    lats, lons = mma.get_healpix_latlon()
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Mollweide()))
    ax.scatter(
        lons, lats, c=rate.values, cmap="viridis", s=1,
        transform=ccrs.PlateCarree(), vmin=0, vmax=1,
    )
    ax.coastlines()
else:
    rate.plot(cmap="viridis", vmin=0, vmax=1)
```

---

## Complete Minimal Example

```python
from toad import MMA, TOAD
from toad.regridding import HealPixRegridder
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

NSIDE = 32

for i, name in enumerate(["model_a", "model_b", "model_c"]):
    td = TOAD(f"{name}.nc", time_dim="time")
    td.compute_shifts(td.base_vars[0], method=ASDETECT())
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=20),
        regridder=HealPixRegridder(nside=NSIDE),
        export_for_mma=f"{name}_clusters.nc",
        mma_grid="healpix",
    )

mma = MMA(
    ["model_a_clusters.nc", "model_b_clusters.nc", "model_c_clusters.nc"],
    nside=NSIDE,
)
mma.run_consensus(min_consensus=0.5, temporal_tolerance=1, spatial_tolerance=1)
mma.data.to_netcdf("consensus.nc")
```

---

## File Formats

| Format   | When to use            | MMA init              | Consensus output dims |
|----------|------------------------|-----------------------|------------------------|
| HealPix  | Mixed grids, global    | `MMA(paths, nside=32)` | `(time, hp_pixel)` |
| Native   | Same grid across models | `MMA(paths, nside=None)` | `(time, …spatial…)` |

Exports store full spacetime `cluster` labels (`TOAD_cluster_labels_v1` convention).

---

## Shift time distributions per consensus cluster

```python
times_by_cluster = mma.get_shift_times_per_consensus_cluster()
```

For a single cluster from one export: `mma.get_shift_times_from_export(path, consensus_cluster_id=0)`.

To map consensus onto an original model grid:

```python
td = TOAD("model_a.nc", time_dim="time")
consensus_ids = mma.map_consensus_to_dataset(td.data)  # time-collapsed on HealPix
```

---

## MMA method reference

| Method | Requires `run_consensus` | Description |
|--------|--------------------------|-------------|
| `MMA.from_consensus(path)` | No (consensus pre-loaded) | Reload saved consensus netCDF |
| `cluster_occurrence_rate()` | No | Ever-in fraction of models that clustered each point [0, 1] |
| `run_consensus(...)` | — | Member-support consensus; populates `mma.data` |
| `get_consensus_summary()` | Yes | Per-cluster size, rate, shift-time stats |
| `plot_consensus_clusters(...)` | Yes | Map plot (time-collapsed consensus) |
| `get_healpix_latlon()` | No (HealPix only) | (lat, lon) for each HEALPix pixel |
| `get_shift_times_from_export(...)` | Yes | Shift times from one export for one cluster |
| `get_shift_times_per_consensus_cluster()` | Yes | Dict of shift times per cluster (all exports) |
| `map_consensus_to_coords(lat, lon)` | Yes (HealPix) | Time-collapsed cluster ID lookup |
| `map_consensus_to_dataset(ds)` | Yes | Map time-collapsed consensus onto a dataset grid |
