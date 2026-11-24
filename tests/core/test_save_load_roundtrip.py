"""Cluster id attributes must survive a netCDF save and load round trip."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from sklearn.cluster import DBSCAN  # type: ignore  # noqa: E402

from toad import TOAD  # noqa: E402


def _make_td(n_patches: int) -> TOAD:
    """Grid with ``n_patches`` separate square patches that shift at t=30."""
    rng = np.random.default_rng(0)
    nt, ny, nx = 60, 8, 8
    t = np.arange(nt, dtype=float)
    data = 0.01 * rng.standard_normal((nt, ny, nx))
    step = (t >= 30).astype(float)[:, None, None]
    if n_patches == 1:
        data = data + step
    elif n_patches == 2:
        data[:, :3, :3] += step
        data[:, 5:, 5:] += step
    ds = xr.Dataset(
        {"v": (("time", "y", "xx"), data)},
        coords={"time": t, "y": np.arange(ny), "xx": np.arange(nx)},
    )
    td = TOAD(ds, log_level="WARNING")
    td.compute_shifts(show_progress=False)
    return td


@pytest.mark.parametrize(
    "case, expected_ids",
    [
        ("one_cluster_no_noise", [0]),
        ("noise_only", [-1]),
        ("several_clusters", [-1, 0, 1]),
    ],
)
def test_cluster_ids_survive_save_and_load(tmp_path, case, expected_ids):
    if case == "one_cluster_no_noise":
        td = _make_td(1)
        td.compute_clusters(method=DBSCAN(eps=100, min_samples=1))
    elif case == "noise_only":
        td = _make_td(1)
        td.compute_clusters(method=DBSCAN(eps=1e-9, min_samples=10_000))
    else:
        td = _make_td(2)
        td.compute_clusters(method=DBSCAN(eps=2, min_samples=5))

    cv = td.cluster_vars[0]
    assert sorted(td.get_cluster_ids(exclude_noise=False).tolist()) == expected_ids

    path = str(tmp_path / f"{case}.nc")
    td.save(path=path)
    td2 = TOAD(path, log_level="WARNING")

    ids = td2.get_cluster_ids(cv, exclude_noise=False)
    assert ids.ndim == 1
    assert sorted(ids.tolist()) == expected_ids
    non_noise = [i for i in expected_ids if i != -1]
    assert sorted(td2.get_cluster_ids(cv).tolist()) == non_noise

    for cid in expected_ids:
        mask = td2.get_cluster_mask(cv, cluster_id=cid)
        assert int(mask.sum()) == int(td.get_cluster_mask(cv, cluster_id=cid).sum())

    if non_noise:
        summary = td2.cluster_summary(cv)
        assert len(summary) == len(non_noise)
        td2.plot.overview(cv)
        plt.close("all")
