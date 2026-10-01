import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import DBSCAN  # type: ignore

from toad import TOAD
from toad.clustering import compute_clusters
from toad.utils import _attrs
from toad.utils.shift_selection_utils import (
    _compute_dts_peak_sign_mask,
    _plateau_magnitude,
)


def test_plateau_magnitude_window_means():
    # peak at 3: pre [0:2)=100, post [5:7)=150  with L=3, G=1
    base = np.array([100.0, 100.0, 100.0, 125.0, 150.0, 150.0, 150.0])
    mag = _plateau_magnitude(base, peak=3, L=3, G=1)
    assert mag == pytest.approx(50.0)


def test_local_peak_mask_filters_small_episodes():
    time = np.arange(8, dtype=float)
    lat = np.array([0.0])
    lon = np.array([0.0])

    var = np.full((8, 1, 1), 100.0)
    var[4:, 0, 0] = 110.0  # 10-unit change

    dts = np.zeros((8, 1, 1), dtype=float)
    dts[3:5, 0, 0] = 1.0

    var2 = np.full((8, 1, 1), 100.0)
    var2[4:, 0, 0] = 140.0  # 40-unit change
    dts2 = np.zeros((8, 1, 1), dtype=float)
    dts2[3:5, 0, 0] = 1.0

    var_all = np.concatenate([var, var2], axis=2)
    dts_all = np.concatenate([dts, dts2], axis=2)
    lon = np.array([0.0, 1.0])

    ds = xr.Dataset(
        {
            "test": (("time", "lat", "lon"), var_all),
            "test_dts": (("time", "lat", "lon"), dts_all),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["test_dts"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT

    mask = _compute_dts_peak_sign_mask(
        ds["test_dts"],
        "time",
        shift_threshold=0.5,
        shift_selection="local",
        base=ds["test"],
        min_event_magnitude=25.0,
        min_event_magnitude_L=3,
        min_event_magnitude_G=1,
    )
    assert not (mask.sel(lon=0.0) != 0).any()
    assert (mask.sel(lon=1.0) != 0).any()


def test_compute_clusters_respects_min_event_magnitude():
    time = np.arange(8, dtype=float)
    lat = np.array([0.0, 10.0])
    lon = np.array([0.0, 10.0])

    var = np.full((8, 2, 2), 100.0)
    var[4:, 0, 0] = 110.0
    var[4:, 1, 1] = 140.0

    dts = np.zeros((8, 2, 2), dtype=float)
    dts[3:5, 0, 0] = 1.0
    dts[3:5, 1, 1] = 1.0

    ds = xr.Dataset(
        {
            "test": (("time", "lat", "lon"), var),
            "test_dts": (("time", "lat", "lon"), dts),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["test_dts"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT

    td = TOAD(ds)
    td.data = compute_clusters(
        td,
        var="test_dts",
        method=DBSCAN(eps=500.0, min_samples=1),
        shift_threshold=0.5,
        shift_selection="local",
        min_event_magnitude=25.0,
        min_event_magnitude_L=3,
        min_event_magnitude_G=1,
        disable_regridder=True,
        overwrite=True,
    )
    clusters = td.data["test_dts_cluster"].values
    assert not np.isfinite(clusters[:, 0, 0]).any()
    assert np.isfinite(clusters[:, 1, 1]).any()
    assert td.data["test_dts_cluster"].attrs[_attrs.MIN_EVENT_MAGNITUDE] == 25.0
    assert td.data["test_dts_cluster"].attrs[_attrs.MIN_EVENT_MAGNITUDE_L] == 3
    assert td.data["test_dts_cluster"].attrs[_attrs.MIN_EVENT_MAGNITUDE_G] == 1
    assert td.data["test_dts_cluster"].attrs[_attrs.N_DATA_POINTS] == 1


def test_compute_clusters_save_without_min_event_magnitude(tmp_path):
    time = np.arange(8, dtype=float)
    lat = np.array([0.0, 10.0])
    lon = np.array([0.0, 10.0])

    var = np.full((8, 2, 2), 100.0)
    var[4:, 0, 0] = 110.0

    dts = np.zeros((8, 2, 2), dtype=float)
    dts[3:5, 0, 0] = 1.0

    ds = xr.Dataset(
        {
            "test": (("time", "lat", "lon"), var),
            "test_dts": (("time", "lat", "lon"), dts),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["test_dts"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT

    td = TOAD(ds)
    td.data = compute_clusters(
        td,
        var="test_dts",
        method=DBSCAN(eps=500.0, min_samples=1),
        shift_threshold=0.5,
        shift_selection="local",
        disable_regridder=True,
        overwrite=True,
    )
    assert _attrs.MIN_EVENT_MAGNITUDE not in td.data["test_dts_cluster"].attrs
    td.save(path=str(tmp_path / "clusters.nc"), overwrite=True)


def test_collect_episode_magnitudes_local_episodes():
    time = np.arange(10, dtype=float)
    var = np.array(
        [100.0, 100.0, 100.0, 100.0, 100.0, 130.0, 130.0, 130.0, 130.0, 130.0]
    )
    dts = np.array([0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0])

    ds = xr.Dataset(
        {
            "test": (("time",), var),
            "test_dts": (("time",), dts),
        },
        coords={"time": time},
    )
    from toad.utils.shift_selection_utils import collect_episode_magnitudes

    mags = collect_episode_magnitudes(
        ds["test_dts"],
        ds["test"],
        "time",
        shift_threshold=0.5,
        L=3,
        G=1,
    )
    assert mags.size == 1
    # peak at index 4: pre mean[1:3]=100, post mean[6:8]=130
    assert mags[0] == pytest.approx(30.0)
