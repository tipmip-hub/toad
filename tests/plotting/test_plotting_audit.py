"""Regression tests for cluster-id types, consensus colour bar and highlight."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import HDBSCAN

import toad.plotting as _tp
from toad import TOAD, shifts

_attrs = _tp._attrs


def _build(nt=60, n=8):
    a = np.zeros((nt, n, n))
    a[30:, 2:6, 2:6] += 5.0
    a[:, 2:6, 2:6] += np.linspace(0, 1, nt)[:, None, None] * 0.3
    da = xr.DataArray(
        a,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(nt),
            "lat": np.arange(n) * 1.0,
            "lon": np.arange(n) * 1.0,
        },
    )
    return xr.Dataset({"v": da})


@pytest.fixture(scope="module")
def td_clusters():
    td = TOAD(_build())
    td.compute_shifts("v", method=shifts.ASDETECT(), show_progress=False)
    td.compute_clusters(
        "v",
        method=HDBSCAN(min_cluster_size=5, allow_single_cluster=True),
        disable_regridder=True,
    )
    return td


def test_numpy_integer_cluster_ids_loop(td_clusters):
    td = td_clusters
    ids = td.get_cluster_ids("v")
    assert len(ids) > 0
    for i in ids:
        for fn in (td.plot.timeseries, td.plot.overview, td.plot.cluster_map):
            out = fn(cluster_ids=i)
            plt.close(out[0])
    plt.close("all")


def _consensus_td():
    n = 20
    ds = _build(nt=40, n=n)
    td = TOAD(ds)
    lab = np.full((40, n, n), np.nan)
    lab[:, 2:6, 2:6] = 2
    lab[:, 12:17, 12:17] = 5
    td.data["cons"] = xr.DataArray(
        lab,
        dims=("time", "lat", "lon"),
        coords=ds.v.coords,
        attrs={_attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_CLUSTER},
    )
    return td


@pytest.mark.parametrize("ids", [[2, 5], [5, 2]])
def test_consensus_colorbar_matches_fill(ids):
    td = _consensus_td()
    fig, ax = td.plot.consensus_map("cons", cluster_ids=ids)
    fills = {}
    for c in ax.collections:
        if type(c).__name__.endswith("QuadMesh"):
            arr = np.ma.getdata(c.get_array()).reshape(20, 20)
            msk = np.ma.getmaskarray(c.get_array()).reshape(20, 20)
            for cid, (r, cc) in {2: (3, 3), 5: (14, 14)}.items():
                if np.isfinite(arr[r, cc]) and not msk[r, cc]:
                    fills[cid] = tuple(np.round(c.get_cmap()(0.5), 3))
    assert set(fills) == {2, 5}
    cbax = [a for a in fig.axes if a is not ax][-1]
    mesh = [c for c in cbax.get_children() if type(c).__name__ == "QuadMesh"][0]
    cm, nm = mesh.get_cmap(), mesh.norm
    ticks = list(cbax.get_xticks()) if len(cbax.get_xticks()) else []
    if not ticks or set(np.round(ticks).astype(int)) != {2, 5}:
        ticks = list(cbax.get_yticks())
    assert sorted(int(round(t)) for t in ticks) == [2, 5]
    for t in ticks:
        cid = int(round(t))
        assert tuple(np.round(cm(nm(t)), 3)) == fills[cid]
    plt.close(fig)


def test_highlight_lies_on_trajectory_with_normalize(td_clusters):
    td = td_clusters
    cid = int(td.get_cluster_ids("v")[0])
    for norm in ["max", "max_each"]:
        fig, ax = td.plot.timeseries(
            "v",
            cluster_ids=cid,
            normalize=norm,
            highlight_color="red",
            highlight_alpha=1.0,
            trajectories_alpha=1.0,
            plot_cluster_duration=False,
        )
        ax0 = np.ravel(ax)[0] if hasattr(ax, "__len__") else ax
        lines = ax0.get_lines()
        red = [ln for ln in lines if ln.get_color() == "red"]
        base = [
            ln for ln in lines if ln.get_color() != "red" and len(ln.get_xdata()) == 60
        ]
        assert red and base
        for r in red:
            yr = np.asarray(r.get_ydata(), float)
            m = np.isfinite(yr)
            assert m.any()
            assert any(
                np.allclose(np.asarray(b.get_ydata(), float)[m], yr[m]) for b in base
            )
        plt.close(fig)
