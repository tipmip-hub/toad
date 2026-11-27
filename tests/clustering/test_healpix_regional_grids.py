"""Default HEALPix regridding on regional lat/lon grids."""

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import HDBSCAN

from toad import TOAD


@pytest.mark.parametrize("step,n", [(0.25, 40), (1.0, 20), (5.0, 12)])
def test_compute_clusters_runs_on_a_regional_grid(step, n):
    """Default regridding on a regional lat/lon grid must not crash."""
    lat = 40.0 + step * np.arange(n)
    lon = 10.0 + step * np.arange(n)
    time = np.arange(60.0)
    rng = np.random.default_rng(0)
    data = 0.05 * rng.standard_normal((time.size, n, n))
    half = n // 2
    data[30:, :half, :half] += 3.0
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    td = TOAD(ds)
    td.compute_shifts("v")
    td.compute_clusters(
        "v",
        method=HDBSCAN(min_cluster_size=5),
        shift_threshold=0.5,
        shift_direction="positive",
    )
    assert "v_dts_cluster" in td.data
