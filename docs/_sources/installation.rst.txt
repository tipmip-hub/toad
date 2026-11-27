Installation
============

TOAD can be installed via PyPI (recommended) or directly from the GitHub repository.

PyPI Installation (Recommended)
-------------------------------

The easiest way to install TOAD is using pip:

.. code-block:: bash

   pip install tipmip-toad

This will install TOAD along with all required dependencies.

Installation from Source
-------------------------

To install the latest development version directly from GitHub:

.. code-block:: bash

   git clone https://github.com/tipmip-methods/toad.git
   cd toad
   pip install .

Developer Installation
----------------------

If you plan to modify TOAD or contribute to the project, install in editable mode with development dependencies:

.. code-block:: bash

   git clone https://github.com/tipmip-methods/toad.git
   cd toad
   pip install -e .[dev]

The ``-e`` flag installs the package in "editable" mode, meaning changes to the source code are immediately reflected without needing to reinstall.

System Requirements
------------------

TOAD requires:

* **Python**: 3.10 or higher
* **Dependencies**: NumPy, SciPy, scikit-learn, xarray, matplotlib, cartopy, pandas, and others (see `pyproject.toml <https://github.com/tipmip-methods/toad/blob/main/pyproject.toml>`_ for complete list)

All dependencies are automatically installed when installing via pip.

Verifying Installation
----------------------

To verify that TOAD is installed correctly, try importing it:

.. code-block:: python

   >>> import toad
   >>> print(toad.__version__)

Next Steps
----------

Once installed, proceed to the :doc:`quick_start` guide to run your first TOAD analysis.
