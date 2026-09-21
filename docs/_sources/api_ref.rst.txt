#############
API Reference
#############

:Date: |today|

This page provides an auto-generated summary of TOAD's API. **All interaction with TOAD happens through the :class:`toad.TOAD` class**, which serves as the main entry point for the entire framework.

Main Entry Point
================

The :class:`toad.TOAD` class is the primary interface for all TOAD functionality. You initialize it with your data (xarray Dataset or path to netCDF file), then use its methods and properties to perform shift detection, clustering, visualization, and analysis.

.. autosummary::
	:toctree: generated/
	:recursive:

	toad.TOAD

The TOAD class provides:

- **Computation methods**: :meth:`toad.TOAD.compute_shifts` and :meth:`toad.TOAD.compute_clusters` for the main analysis pipeline; :meth:`toad.TOAD.compute_consensus` to merge several cluster maps (see :doc:`consensus_clustering`)
- **Plotting interface**: Access via :attr:`toad.TOAD.plot` property (returns a :class:`toad.plotting.Plotter` instance)
- **Statistics interface**: Access via :meth:`toad.TOAD.stats` method (returns a :class:`toad.postprocessing.Stats` instance)
- **Aggregation interface**: Access via :attr:`toad.TOAD.aggregate` property (returns a :class:`toad.postprocessing.Aggregation` instance)
- **Preprocessing interface**: Access via :attr:`toad.TOAD.preprocess` property
- **Data access**: The underlying xarray Dataset is available via :attr:`toad.TOAD.data`

Shift Detection Methods
=======================

Shift detection algorithms that can be passed to :meth:`toad.TOAD.compute_shifts`:

.. autosummary::
	:toctree: generated/
	:recursive:

	toad.shifts.ASDETECT
	toad.shifts.methods.ShiftsMethod

Clustering Methods
==================

Clustering algorithms from scikit-learn (or custom implementations) can be passed to :meth:`toad.TOAD.compute_clusters`. TOAD is compatible with any clustering method that follows the scikit-learn `ClusterMixin` interface.

Common examples:
- :class:`sklearn.cluster.HDBSCAN`
- :class:`sklearn.cluster.DBSCAN`

Consensus clustering
====================

When you have two or more cluster label fields on the same grid (different models, parameters, or
variables), :meth:`toad.TOAD.compute_consensus` merges them into one spacetime consensus field.

- **Concepts and parameters**: :doc:`consensus_clustering`
- **Worked example**: :doc:`Consensus tutorial <tutorials/consensus>`

Plotting and Visualization
===========================

The plotting functionality is accessed through the :attr:`toad.TOAD.plot` property, which returns a :class:`toad.plotting.Plotter` instance. You can customize map styling using :class:`toad.plotting.MapStyle`.

.. autosummary::
	:toctree: generated/
	:recursive:

	toad.plotting.Plotter
	toad.plotting.MapStyle

Regridding
==========

Regridding utilities for ensuring equal spacing in global datasets. These are typically used automatically when needed, but can be configured via the `regridder` parameter in :meth:`toad.TOAD.compute_clusters`.

.. autosummary::
	:toctree: generated/
	:recursive:

	toad.regridding.HealPixRegridder
	toad.regridding.base.BaseRegridder

Supporting Modules
==================

These modules provide the underlying functionality used by the TOAD class:

.. autosummary::
	:toctree: generated/
	:recursive:

	toad.shifts
	toad.clustering
	toad.postprocessing
	toad.utils