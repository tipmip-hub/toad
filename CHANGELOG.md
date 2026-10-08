# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.9] - 2026-10-08

### Added
- **`EDGE`** shift detection method (`toad.shifts.EDGE`) — alternative abrupt-shift detector based on Bathiany et al. (2020) and Terpstra et al. (2025); TOAD implementation by Sjoerd Terpstra. Scores are signed abruptness (not normalised to [-1, 1]); apply the usual 4-sigma cut via `shift_threshold=4` in `compute_clusters`. See `tutorials/edge_detection.ipynb`.
- E3SM NH March sea-ice test data for the EDGE tutorial, with attribution in `DATA_ATTRIBUTION.md`.
- `cbar_kwargs` on `max_shift_map` for colourbar customisation.
- `overwrite=True` on `TOAD.save` to replace the source path in place.

### Changed
- Plots and time stats support **non-normalised shift scores** (e.g. EDGE abruptness): shift maps/histograms auto-scale when scores fall outside [-1, 1]; transition-time stats use shift variables before clustering.
- Documented that built-in clusterers (HDBSCAN, DBSCAN, SpaceTimeDBSCAN) ignore shift magnitudes passed as `y` — magnitudes only affect point selection via `shift_threshold` / `shift_selection`.
- Opening netCDF paths defaults to **cftime** decoding (`CFDatetimeCoder(use_cftime=True)`); override via `decode_times`.
- ASDETECT trims leading/trailing NaN padding, maps scores back with zeros outside the finite core, and only skips cells with internal NaN gaps.
- `clean_for_toad` drops static (time-less) grid geometry variables before spatial-dim inference (helps irregular grids such as CESM POP).
- Overview / map+timeseries plots return a placeholder panel (`"No clusters identified"`) when no clusters are present, instead of `(None, None)`.

### Fixed
- Optuna clustering optimisation: fixed parameters (e.g. `min_cluster_size`) were dropped from the final fit because `best_params` only contains suggested ranges.
- Cartopy cluster contour crash on PlateCarree maps with 2D gridded coordinates.
- `TOAD.save` no longer crashes when the destination netCDF is open for read (always writes via temp file + atomic replace).
- `repr_html` cluster counts handle scalar `cluster_ids` attrs from netCDF round-trips (e.g. noise-only `-1`).
- README links; fallback font when Verdana is unavailable.

## [1.0.8] - 2026-06-22

### Changed
- **`compute_consensus` redesigned** — spacetime member-support consensus on the native grid; requires `min_consensus`, `temporal_tolerance`, and `spatial_tolerance`. Replaces the old edge-vote method; results will differ. See `tutorials/consensus.ipynb`.
- Consensus companion field and plot API renamed: `*_consistency` → `*_rate`, `consensus_consistency_map()` → `consensus_rate_map()`, summary column `mean_consistency` → `mean_consensus_rate`.
- HealPix regridding now uses **`astropy-healpix`** instead of the `healpix` package.
- Clustering and transition times work with **time-last** dimension order (e.g. `y`, `x`, `time`).

### Added
- **`SpaceTimeDBSCAN`** — built-in clusterer with separate spatial and temporal neighbourhood sizes (`spatial_eps`, `temporal_eps`).
- Consensus **rate** field (`{consensus_var}_rate`), **`stitch_meridian`**, **`cluster_occurrence_rate()`**, expanded summary/plots, and consensus tutorial + docs.
- **`TOAD(..., auto_clean=True)`** — optional CMIP-style cleanup on load (off by default).
- **`cluster_summary()`** on `TOAD` and `Stats` — per-cluster overview table (time span, size, footprint, centre; optional IQR, shift amplitudes, pooled transition times, and clustering metadata). Analogous to `consensus_summary` for standard cluster variables.
- **`TimeStats` timing summaries** — `median_activity_time`, `pooled_median_transition_time`, `pooled_std_transition_time`, and `summary()` DataFrame contrasting cluster temporal footprint vs per-cell peak-shift times (same field as `compute_transition_time` / `time_of_max_shift_map`).

### Removed
- Old edge-vote consensus and **`regridder=`** on `compute_consensus` (HealPix remains for `compute_clusters`).

### Fixed
- Dimension-order bugs in clustering and transition times; HealPix regridder out-of-bounds error.
- Explicit **`astropy` dependency** for HealPix lon/lat unit conversion (CI deptry).
- HDBSCAN **`allow_single_cluster=True` by default** when sklearn supports it.

## [1.0.7] - 2026-03-26

### Fixed
- Timeseries subplots no longer force y-axis lower limit to zero (removed spurious `axhline(0)` call in subplot cleanup)

## [1.0.6] - 2026-03-11

### Added
- Stats can now be accessed as a property (e.g. `td.stats.time.start(cluster_id=0)`) when using a single cluster variable
- Function for removing specific clusters by id (useful for filtering clusters of no interest)
- Function for sorting clusters (e.g. by magnitude, median shift time, or to reset indexing after removal)
- New time stats: `value_at_start`, `value_at_end`, `value_change`, `value_at_iqr_90_start`/`value_at_iqr_90_end`, `value_change_iqr_90`
- `get_cluster_times()` — returns flattened array of time values for every cell in a cluster
- Variable inference for `get_base_var()`, `get_shifts()`, and `get_timeseries()`. 
- `trajectory_ids` parameter in td.plot.timeseries() shows exact cell indices (instead of random sample)
- `plot_shift_indicator` in timeseries plots to mark detection timestep per cell on each trajectory
- `DEFAULT_SHIFT_THRESHOLD` constant in `toad.utils` for the default clustering threshold

### Changed
- `get_cluster_timeseries()` deprecated in favour of `get_timeseries()`; deprecated alias logs a warning
- Timeseries plot: `plot_shift_indicator` now marks detection timesteps; duration shading renamed to `plot_cluster_duration`
- Replaced internal `assert` checks with proper error messages (validation now works when Python is run with optimisations)

### Deprecated
- `get_cluster_timeseries()` — use `get_timeseries()` instead

### Removed
- Synthetic shifts generator (development tool)
- Redundant core methods: `get_active_clusters_count_per_timestep`, `apply_cluster_mask`, `apply_cluster_mask_spatial`, `apply_cluster_mask_temporal`, `get_cluster_density_temporal`, `get_cluster_density_spatial`, `get_cluster_mask_temporal`, `get_cluster_data` — use stats API instead

### Fixed
- More robust variable selection in consensus clustering (cluster variable is now specified directly instead of inferred from shifts)
- `compute_transition_time` now restricted to cluster period (previously used largest shift in each grid cell regardless of period)
- Z-order of contours in cluster maps (clusters now correctly layered)
- Enhanced error handling in asdetect for short time series (validates `lmin` < `lmax` with guiding error message for common defaults)
- Steepest gradient: fix handling of NaN values and time indexing
- Timeseries plot: `keep_full_timeseries` now correctly applied throughout; cluster duration shading hidden when showing only the cluster period

## [1.0.5] - 2026-02-12

### Added
- Allow custom cluster methods to turn off temporal scaling by adding `skip_time_scaling = True` as a class variable to the method.
- Allow disabling the gradient legend in cluster maps (set `other_legend=False` in `MapStyle`)
- Added progress bar for cluster consensus computation (`show_progress` parameter)
- Added comprehensive tests for plotting and postprocessing functions

### Changed
- Cluster output now distinguishes between grid cells classified as noise by the clustering algorithm (-1) and grid cells with no detected abrupt shifts (NaN). Previously both were labeled -1.
- Enhanced variable inference in TOAD methods
- Relaxed dependency version pins for numpy, scipy, and scikit-learn (removed upper bounds)

### Fixed
- Fixed base variable inference in `get_cluster_timeseries` for scoring functions
- Fixed dependency configuration: added `tqdm` as direct dependency, corrected deptry module name mappings and ignore lists

## [1.0.4] - 2026-01-27

### Changed
- Updated README.md installation instructions to include pip installation command. 

## [1.0.3] - 2026-01-27

### Fixed
- Fixed image paths in README.md to use absolute URLs for proper display on PyPI

## [1.0.2] - 2026-01-27

### Changed
- Updated PyPI publishing workflow to use Trusted Publishers (OIDC) instead of API tokens for improved security

## [1.0.1] - 2026-01-27

### Changed
- Updated README.md with improved installation and usage instructions
- Enhanced documentation structure and content
- Reorganized and updated existing tutorials

### Added
- Added CITATION.cff for citation support
- Added CONTRIBUTING.md with contribution guidelines
- Added GitHub Actions workflow for publishing to PyPI — first PyPI release!
- Added Github Actions workflow for automatically creating draft release when tag with v* is pushed to main. 

## [1.0.0] - 2026-01-22

### Added
- First public release of the TOAD package

[Unreleased]: https://github.com/tipmip-methods/toad/compare/v1.0.9...HEAD
[1.0.9]: https://github.com/tipmip-methods/toad/compare/v1.0.8...v1.0.9
[1.0.8]: https://github.com/tipmip-methods/toad/compare/v1.0.7...v1.0.8
[1.0.7]: https://github.com/tipmip-methods/toad/compare/v1.0.6...v1.0.7
[1.0.6]: https://github.com/tipmip-methods/toad/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/tipmip-methods/toad/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/tipmip-methods/toad/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/tipmip-methods/toad/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/tipmip-methods/toad/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/tipmip-methods/toad/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tipmip-methods/toad/releases/tag/v1.0.0
