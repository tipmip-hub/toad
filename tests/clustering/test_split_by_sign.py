import gc

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import DBSCAN, HDBSCAN  # type: ignore

from toad import TOAD
from toad.clustering import compute_clusters
from toad.utils import _attrs


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test."""
    yield
    gc.collect()


def _assert_clusters_sign_homogeneous(clusters_da, shifts_da):
    """Every non-noise cluster label must have uniform shift sign."""
    c = clusters_da.values
    s = shifts_da.values
    for cid in np.unique(c[np.isfinite(c) & (c >= 0)]):
        mask = c == cid
        signs = np.sign(s[mask])
        assert np.all(signs > 0) or np.all(signs < 0), (
            f"Cluster {int(cid)} has mixed signs"
        )


def test_both_directions_split_by_sign_synthetic():
    """shift_direction='both' must not produce mixed-sign clusters."""
    time = np.arange(10, dtype=float)
    lat = np.array([0.0, 10.0])
    lon = np.array([0.0, 10.0])
    sh = np.zeros((10, 2, 2), dtype=float)
    # Pos/neg pairs at the same time and nearby in space would merge without sign split.
    sh[5, 0, 0] = 1.0
    sh[5, 0, 1] = 1.0
    sh[5, 1, 0] = -1.0
    sh[5, 1, 1] = -1.0

    ds = xr.Dataset(
        {"test_dts": (("time", "lat", "lon"), sh)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["test_dts"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT

    td = TOAD(ds)
    td.data = compute_clusters(
        td,
        var="test_dts",
        method=DBSCAN(eps=500.0, min_samples=2),
        shift_threshold=0.5,
        shift_direction="both",
        shift_selection="all",
        disable_regridder=True,
        overwrite=True,
    )
    _assert_clusters_sign_homogeneous(td.data["test_dts_cluster"], td.data["test_dts"])


def test_both_directions_sign_homogeneous_on_synth_data():
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=4, lon=4, boundary="trim").reduce(np.mean)

    td.compute_clusters(
        shift_threshold=0.5,
        method=HDBSCAN(min_cluster_size=10),
        shift_direction="both",
        shift_selection="all",
        disable_regridder=True,
        overwrite=True,
    )

    cluster_var = td.cluster_vars[0]
    shifts_var = td.data[cluster_var].attrs[_attrs.SHIFTS_VARIABLE]
    _assert_clusters_sign_homogeneous(td.data[cluster_var], td.data[shifts_var])
