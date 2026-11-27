"""Guard: a plain compute_clusters() call with the default regridder gives the
same labels as toad 1.0.10 on a small global grid. Update the expected numbers only for a deliberate change."""

import numpy as np
import xarray as xr
from sklearn.cluster import HDBSCAN

from toad import TOAD
from toad.utils import _attrs


def _ds():
    rng = np.random.default_rng(0)
    time = np.arange(30, dtype=float)
    lat = np.arange(-87.5, 90.0, 5.0)
    lon = np.arange(2.5, 360.0, 5.0)
    lon2d, lat2d = np.meshgrid(lon, lat)
    sh = np.zeros((time.size, lat.size, lon.size))
    blobs = [(40, 20, 8, 1, 15), (-30, 200, 18, -1, 15), (70, 300, 24, 1, 12)]
    blobs.append((0, 120, 12, -1, 10))
    for la0, lo0, t0, sgn, r in blobs:
        dlon = (lon2d - lo0 + 180) % 360 - 180
        m = np.hypot(lat2d - la0, dlon * np.cos(np.deg2rad(lat2d))) < r
        tt = np.clip(t0 + rng.integers(-2, 3, size=m.sum()), 0, time.size - 1)
        sh[tt, np.nonzero(m)[0], np.nonzero(m)[1]] = sgn * rng.uniform(
            0.6, 1.0, size=m.sum()
        )
    ds = xr.Dataset(
        {"x_dts": (("time", "lat", "lon"), sh)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["x_dts"].attrs[_attrs.BASE_VARIABLE] = "x"
    ds["x_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT
    return ds


def test_plain_compute_clusters_labels_unchanged():
    td = TOAD(_ds(), log_level="ERROR")
    td.compute_clusters("x_dts", method=HDBSCAN(min_cluster_size=5))
    c = td.data["x_dts_cluster"]
    v = c.values.ravel()
    fin = np.isfinite(v)
    labels, counts = np.unique(v[fin], return_counts=True)
    assert dict(zip(labels.astype(int).tolist(), counts.tolist())) == {
        0: 54,
        1: 40,
        2: 34,
        3: 12,
    }
    assert int(np.sum(np.flatnonzero(fin) * (v[fin].astype(int) + 2))) == 18151436
    assert c.attrs["regridder_nside"] == "16"
