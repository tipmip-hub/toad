"""Empty HEALPix pixels inside a coarse source grid take their nearest cell."""

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy_healpix import healpix_to_lonlat
from sklearn.cluster import HDBSCAN

from toad import TOAD
from toad.regridding import HealPixRegridder


def _pixel_centres(nside):
    pix = np.arange(12 * nside**2)
    lon, lat = healpix_to_lonlat(pix, nside, order="ring")
    return pix, lat.to_value(u.deg), np.mod(lon.to_value(u.deg), 360.0)


def _coarse_grid():
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.25, 360.0, 2.5)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d


def _block_events(lat2d, lon2d):
    """One time step with an event in every cell of a 10 x 10 block."""
    block = (np.abs(lat2d - 40.0) < 10) & (np.abs(lon2d - 220.0) < 12.5)
    coords = np.column_stack([np.zeros(block.sum()), lat2d[block], lon2d[block]])
    return coords, np.ones(len(coords))


def test_without_source_grid_pixels_stay_empty():
    lat2d, lon2d = _coarse_grid()
    coords, weights = _block_events(lat2d, lon2d)
    regridder = HealPixRegridder(nside=32)
    regridder.regrid(coords, weights, lat2d.shape)
    assert len(regridder.df_healpix) <= len(coords)


def test_fill_gives_every_pixel_in_the_block_a_value():
    lat2d, lon2d = _coarse_grid()
    coords, weights = _block_events(lat2d, lon2d)

    plain = HealPixRegridder(nside=32)
    plain.regrid(coords, weights, lat2d.shape)
    filled = HealPixRegridder(nside=32)
    filled.set_source_grid(lat2d, lon2d)
    filled.regrid(coords, weights, lat2d.shape)

    pix_plain = set(plain.df_healpix["hp_pix"])
    pix_filled = set(filled.df_healpix["hp_pix"])
    assert pix_plain < pix_filled

    # Every pixel whose centre lies well inside the block now holds an event.
    all_pix, plat, plon = _pixel_centres(32)
    inside = (np.abs(plat - 40.0) < 8) & (np.abs(plon - 220.0) < 10)
    assert inside.sum() > 0
    assert set(all_pix[inside]) <= pix_filled
    # ... and nothing is spread far outside it.
    outside = (np.abs(plat - 40.0) > 13) | (np.abs(plon - 220.0) > 16)
    assert not (set(all_pix[outside]) & pix_filled)


def test_labels_map_back_to_every_source_cell():
    lat2d, lon2d = _coarse_grid()
    coords, weights = _block_events(lat2d, lon2d)
    regridder = HealPixRegridder(nside=32)
    regridder.set_source_grid(lat2d, lon2d)
    regridded, _ = regridder.regrid(coords, weights, lat2d.shape)
    back = regridder.regrid_clusters_back(np.zeros(len(regridded), dtype=int))
    assert back.shape == (len(coords),)
    assert (back == 0).all()


def test_fine_grid_has_nothing_to_fill():
    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(0.25, 360.0, 0.5)
    lon2d, lat2d = np.meshgrid(lon, lat)
    regridder = HealPixRegridder(nside=32)
    regridder.set_source_grid(lat2d, lon2d)
    assert regridder.empty_pixel_fill_map() == {}


def test_fill_can_be_switched_off():
    lat2d, lon2d = _coarse_grid()
    regridder = HealPixRegridder(nside=32, fill_empty_pixels=False)
    regridder.set_source_grid(lat2d, lon2d)
    assert regridder.empty_pixel_fill_map() == {}


def _regional_grid(step=1.0, n=20, lat0=40.0, lon0=10.0):
    lat = lat0 + step * np.arange(n)
    lon = lon0 + step * np.arange(n)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d


@pytest.mark.parametrize("nside", [4, 16])
def test_regional_grid_with_empty_pixels_all_out_of_reach(nside):
    """Pixels as large as the cells or larger: all empty pixels lie outside."""
    lat2d, lon2d = _regional_grid()
    regridder = HealPixRegridder(nside=nside)
    regridder.set_source_grid(lat2d, lon2d)
    assert regridder.empty_pixel_fill_map() == {}


def test_regional_grid_finer_pixels_fill_only_inside():
    lat2d, lon2d = _regional_grid(step=2.0, n=10)
    regridder = HealPixRegridder(nside=64)
    regridder.set_source_grid(lat2d, lon2d)
    fill = regridder.empty_pixel_fill_map()
    assert fill
    _, plat, plon = _pixel_centres(64)
    filled = np.concatenate(list(fill.values()))
    # Within about one cell (2 degrees) of the grid's outer cell centres.
    assert plat[filled].min() > 37.0 and plat[filled].max() < 61.0
    assert plon[filled].min() > 6.0 and plon[filled].max() < 32.0


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
