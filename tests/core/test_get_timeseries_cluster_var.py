"""get_timeseries(keep_full_timeseries=False) must use the given cluster variable."""

import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN  # type: ignore

from toad import TOAD


def test_crop_uses_given_cluster_var_when_two_clusterings_exist():
    rng = np.random.default_rng(0)
    nt, ny, nx = 60, 6, 6
    t = np.arange(nt, dtype=float)
    data = (t >= 30).astype(float)[:, None, None] * np.ones((nt, ny, nx))
    data = data + 0.01 * rng.standard_normal(data.shape)
    ds = xr.Dataset(
        {"v": (("time", "y", "xx"), data)},
        coords={"time": t, "y": np.arange(ny), "xx": np.arange(nx)},
    )
    td = TOAD(ds, log_level="WARNING")
    td.compute_shifts(show_progress=False)
    td.compute_clusters(method=DBSCAN(eps=100, min_samples=1))
    td.compute_clusters(
        method=DBSCAN(eps=100, min_samples=1), output_label_suffix="second"
    )
    assert len(td.cluster_vars) == 2

    for cv in td.cluster_vars:
        ts = td.get_timeseries(
            var=cv, cluster_id=0, aggregation="mean", keep_full_timeseries=False
        )
        start = td.stats(cv).time.start_timestep(0)
        end = td.stats(cv).time.end_timestep(0)
        finite = np.flatnonzero(np.isfinite(ts.values))
        assert finite.min() == start
        assert finite.max() == end
