Quick Start
===========

This guide will get you up and running with TOAD in just a few minutes. For more detailed examples and explanations, see the :doc:`tutorials` section.

Minimal Example
---------------

Here's a simple example that demonstrates the core TOAD workflow:

.. code-block:: python

    from toad import TOAD
    from toad.shifts import ASDETECT
    from sklearn.cluster import HDBSCAN

    # Initialize TOAD object with your data file
    td = TOAD("data.nc")

    # Detect abrupt shifts using ASDETECT method
    td.compute_shifts("tas", method=ASDETECT())

    # Cluster detected shifts using HDBSCAN
    td.compute_clusters(
        var="tas",
        method=HDBSCAN(min_cluster_size=10),
    )

    # Visualize the results
    td.plot.overview("tas")

Next Steps
----------

- **Learn the basics**: Check out the :doc:`tutorials/basics` tutorial for a comprehensive introduction
- **Customize methods**: Learn how to :doc:`tutorials/custom_clustering` and :doc:`tutorials/custom_shifts_detection`
- **Combine clusterings**: When you have several label fields on the same grid, see :doc:`consensus_clustering` and the :doc:`Consensus tutorial <tutorials/consensus>`

Data Format
-----------

TOAD expects input data as:

- **NetCDF files** (``.nc``) readable by xarray
- **xarray Dataset or DataArray** objects
- Data structured as 3D arrays: ``space × space × time``

The ``time`` dimension can represent actual time or any other forcing variable or bifurcation parameter.

For more information about data requirements and formats, see the :doc:`tutorials/basics` tutorial.
