"""Tests for per-event and cluster-level shift magnitude (ordered extremes)."""

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import DBSCAN  # type: ignore

from toad import TOAD
from toad.clustering import compute_clusters
from toad.postprocessing.stats.shift_magnitude import (
    _aggregate_cluster_shift_magnitudes,
    aggregate_shift_magnitudes_by_cluster,
    compute_shift_magnitudes,
    event_shift_magnitude,
    magnitude_var_names,
    resolve_lmin_lmax,
)
from toad.utils import _attrs


def test_resolve_lmin_lmax_reads_shifts_prefixed_attrs():
    """compute_shifts writes shifts_lmin/shifts_lmax; magnitude code must read them."""
    da = xr.DataArray(np.zeros(10), attrs={"shifts_lmin": "5", "shifts_lmax": "12"})
    assert resolve_lmin_lmax(da, n_time=40) == (5, 12)

    da_bare = xr.DataArray(np.zeros(10), attrs={"lmin": 4, "lmax": 9})
    assert resolve_lmin_lmax(da_bare, n_time=40) == (4, 9)

    da_default = xr.DataArray(np.zeros(10), attrs={})
    assert resolve_lmin_lmax(da_default, n_time=40) == (5, 13)


def test_event_shift_magnitude_positive_step():
    y = np.zeros(30, dtype=float)
    y[11:] = 5.0
    out = event_shift_magnitude(y, t_star=10, lmin=3, lmax=8, sign=1.0)
    assert out["flag"] == "ok"
    assert out["magnitude"] == pytest.approx(5.0, abs=1e-6)
    assert out["duration"] > 0
    assert out["t_pre"] is not None and out["t_post"] is not None
    assert out["t_post"] > out["t_pre"]
    assert out["n_pre"] >= 3
    assert out["n_post"] >= 3


def test_event_shift_magnitude_negative_step():
    y = np.full(30, 5.0)
    y[11:] = 0.0
    out = event_shift_magnitude(y, t_star=10, lmin=3, lmax=8, sign=-1.0)
    assert out["flag"] == "ok"
    assert out["magnitude"] == pytest.approx(-5.0, abs=1e-6)


def test_event_shift_magnitude_duration_uses_time_coords():
    y = np.zeros(30, dtype=float)
    y[11:] = 5.0
    time = np.arange(30, dtype=float) * 0.1  # e.g. GWL steps of 0.1
    out = event_shift_magnitude(y, t_star=10, lmin=3, lmax=8, sign=1.0, time=time)
    assert out["flag"] == "ok"
    assert out["duration"] == pytest.approx(
        time[out["t_post"]] - time[out["t_pre"]], abs=1e-9
    )


def test_event_shift_magnitude_ignores_post_spike_for_negative():
    # Collapse 100 → 0, with a tall spike in the post window
    y = np.full(40, 100.0)
    y[16:] = 0.0
    y[22] = 80.0
    out = event_shift_magnitude(y, t_star=15, lmin=3, lmax=10, sign=-1.0)
    assert out["flag"] == "ok"
    # Ordered extremes use min(post), so the spike must not inflate |magnitude|
    assert out["magnitude"] == pytest.approx(-100.0, abs=5.0)


def test_event_shift_magnitude_edge_near_boundary():
    y = np.zeros(20, dtype=float)
    y[3:] = 1.0
    out = event_shift_magnitude(y, t_star=2, lmin=5, lmax=10, sign=1.0)
    assert out["flag"] == "edge"
    assert np.isnan(out["magnitude"])
    assert np.isnan(out["duration"])


def test_neighbour_truncation_cuts_post_window():
    # Two negative drops: first to 50, second to 0. Without truncation the
    # first event's post window would reach the second drop and inflate |D|.
    y = np.full(60, 100.0)
    y[20:] = 50.0
    y[40:] = 0.0
    # Large lmax would swallow the second event
    open_ = event_shift_magnitude(
        y, t_star=20, lmin=3, lmax=30, sign=-1.0, neighbours=(None, None)
    )
    trunc = event_shift_magnitude(
        y, t_star=20, lmin=3, lmax=30, sign=-1.0, neighbours=(None, 40)
    )
    assert open_["flag"] == "ok" and trunc["flag"] == "ok"
    assert trunc["post_hi"] == (40 + 20) // 2  # midpoint
    assert trunc["post_hi"] < open_["post_hi"]
    # Truncated D should stay near the first step (~-50), not the full ~-100
    assert trunc["magnitude"] == pytest.approx(-50.0, abs=5.0)
    assert abs(open_["magnitude"]) > abs(trunc["magnitude"]) + 20.0


def test_compute_shift_magnitudes_dense_fields():
    time = np.arange(40, dtype=float)
    lat = np.array([0.0])
    lon = np.array([0.0, 1.0])

    base = np.zeros((40, 1, 2))
    base[20:, 0, 0] = 4.0
    base[20:, 0, 1] = 1.0

    dts = np.zeros((40, 1, 2))
    dts[19:21, 0, 0] = 1.0
    dts[20, 0, 0] = 2.0
    dts[19:21, 0, 1] = 1.0
    dts[20, 0, 1] = 2.0

    ds = xr.Dataset(
        {
            "tas": (("time", "lat", "lon"), base),
            "tas_dts": (("time", "lat", "lon"), dts),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["tas_dts"].attrs[_attrs.BASE_VARIABLE] = "tas"
    ds["tas_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT
    ds["tas_dts"].attrs["lmin"] = 5
    ds["tas_dts"].attrs["lmax"] = 12

    mag = compute_shift_magnitudes(
        ds["tas"], ds["tas_dts"], "time", shift_threshold=0.5
    )
    mag_name, duration_name = magnitude_var_names("tas_dts")
    assert set(mag.data_vars) == {mag_name, duration_name}

    d0 = float(mag[mag_name].isel(lat=0, lon=0).sel(time=20))
    assert d0 == pytest.approx(4.0, abs=0.1)
    dur0 = float(mag[duration_name].isel(lat=0, lon=0).sel(time=20))
    assert np.isfinite(dur0) and dur0 > 0
    assert mag[mag_name].attrs.get("magnitude_method") == "ordered_extremes"


def test_toad_compute_shift_magnitudes_and_cluster_aggregate():
    time = np.arange(40, dtype=float)
    lat = np.array([0.0, 10.0])
    lon = np.array([0.0, 10.0])

    base = np.zeros((40, 2, 2))
    base[20:, 0, 0] = 5.0
    base[20:, 1, 1] = 3.0

    dts = np.zeros((40, 2, 2))
    dts[20, 0, 0] = 2.0
    dts[19, 0, 0] = 1.0
    dts[21, 0, 0] = 1.0
    dts[20, 1, 1] = 2.0
    dts[19, 1, 1] = 1.0
    dts[21, 1, 1] = 1.0

    ds = xr.Dataset(
        {
            "tas": (("time", "lat", "lon"), base),
            "tas_dts": (("time", "lat", "lon"), dts),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["tas_dts"].attrs[_attrs.BASE_VARIABLE] = "tas"
    ds["tas_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT
    ds["tas_dts"].attrs["lmin"] = 5
    ds["tas_dts"].attrs["lmax"] = 12

    td = TOAD(ds)
    td.compute_shift_magnitudes(var="tas", overwrite=True)

    mag_name, duration_name = magnitude_var_names("tas_dts")
    assert mag_name in td.data
    assert duration_name in td.data

    td.data = compute_clusters(
        td,
        var="tas_dts",
        method=DBSCAN(eps=500.0, min_samples=1),
        shift_threshold=0.5,
        shift_selection="local",
        disable_regridder=True,
        overwrite=True,
    )

    summary = td.cluster_summary("tas")
    assert "median_event_magnitude" in summary.columns
    assert "median_event_duration" in summary.columns
    assert "n_events" in summary.columns
    assert len(summary) >= 1
    assert summary["n_events"].sum() >= 1

    # Vectorised path matches per-cluster mask helper
    clusters = td.get_clusters("tas")
    for cid in summary["cluster_id"].tolist():
        masked = _aggregate_cluster_shift_magnitudes(
            td.data[mag_name],
            td.get_cluster_mask("tas", int(cid)),
            duration=td.data[duration_name],
        )
        row = summary.loc[summary["cluster_id"] == cid].iloc[0]
        assert row["n_events"] == masked["n_events"]
        if masked["n_events"] > 0:
            assert row["median_event_magnitude"] == pytest.approx(
                masked["median_event_magnitude"]
            )
            assert row["median_event_duration"] == pytest.approx(
                masked["median_event_duration"]
            )

    empty = _aggregate_cluster_shift_magnitudes(
        td.data[mag_name],
        np.zeros(td.data[mag_name].shape, dtype=bool),
        duration=td.data[duration_name],
    )
    assert empty["n_events"] == 0
    assert np.isnan(empty["median_event_magnitude"])
    assert np.isnan(empty["median_event_duration"])

    by_id = aggregate_shift_magnitudes_by_cluster(
        td.data[mag_name],
        clusters,
        [int(c) for c in summary["cluster_id"].tolist()],
        duration=td.data[duration_name],
    )
    assert len(by_id) == len(summary)


def test_overwrite_guard():
    time = np.arange(30, dtype=float)
    lat = np.array([0.0])
    lon = np.array([0.0])
    base = np.zeros((30, 1, 1))
    base[15:] = 2.0
    dts = np.zeros((30, 1, 1))
    dts[15] = 2.0

    ds = xr.Dataset(
        {
            "tas": (("time", "lat", "lon"), base),
            "tas_dts": (("time", "lat", "lon"), dts),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["tas_dts"].attrs[_attrs.BASE_VARIABLE] = "tas"
    ds["tas_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT

    td = TOAD(ds)
    td.compute_shift_magnitudes(overwrite=True)
    with pytest.raises(ValueError, match="already exist"):
        td.compute_shift_magnitudes(overwrite=False)
    td.compute_shift_magnitudes(overwrite=True)

    td.drop_shifts()
    assert "tas_dts" not in td.data
    assert "tas_dts_magnitude" not in td.data
    assert "tas_dts_duration" not in td.data
