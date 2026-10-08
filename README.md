<div align="center">

![TOAD](https://raw.githubusercontent.com/tipmip-methods/toad/main/docs/source/resources/toad.png)

# TOAD: **T**ipping and **O**ther **A**brupt events **D**etector

[![Documentation](https://img.shields.io/badge/docs-toad--docs-blue)](https://toad-docs.pages.dev)
[![PyPI version](https://img.shields.io/pypi/v/tipmip-toad?)](https://pypi.org/project/tipmip-toad/)
[![CI](https://github.com/tipmip-methods/toad/actions/workflows/build_and_tests.yml/badge.svg)](https://github.com/tipmip-methods/toad/actions/workflows/build_and_tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-BSD--2--Clause-blue)](LICENSE.txt) <br>
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18316437.svg)](https://doi.org/10.5281/zenodo.18316437) 
[![Code style](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

</div>

TOAD is a **Python** package designed for **detecting and clustering spatio-temporal patterns in gridded Earth-system  datasets**, such as model outputs. The input data should be structured as a 3D array in the format `space` × `space` × `time`, where `time` can represent actual time or some other forcing variable or bifurcation parameter. TOAD provides a streamlined pipeline for identifying and analyzing spatio-temporal regions or clusters that exhibit similar dynamical responses to external forcing.

Currently, TOAD focuses on identifying regions that experience similar abrupt transitions, such as the sudden loss of ice volume in a specific area. The goal, however, is to expand the package's functionality to support broader use cases, enabling the detection and clustering of diverse types of dynamical shifts across a wide range of systems.

The TOAD pipeline consists of three main components:

1. **Shift Detection:** Performs time series analysis at each individual grid cell to identify abrupt transitions or dynamical shifts using configurable detection methods (e.g., ASDETECT) with adjustable sensitivity parameters.
2. **Clustering:** Groups the detected shifts spatially and temporally (or along a bifurcation parameter) to reveal cohesive patterns using clustering methods (e.g., HDBSCAN) with configurable space/time scaling.
3. **Aggregation & Synthesis:** Aggregates results across multiple cluster maps from different datasets, models, variables, realisations, or methods to produce consensus clusters and statistics, and generates plots and summaries of the identified clusters for insights and interpretation.


![TOAD Pipeline](https://raw.githubusercontent.com/tipmip-methods/toad/main/docs/source/resources/TOAD_pipeline.png)

TOAD's core functionality is exposed through the `TOAD` class, which analyzes netCDF files or xarray datasets. The primary methods - `compute_shifts`, `compute_clusters`, and `compute_consensus()` - handle the three main pipeline steps. The pipeline supports aggregation of results across multiple runs, enabling consensus-based analysis. Additional helper functions and visualization tools make it easy to explore and understand the results.


## Installation

[![PyPI version](https://img.shields.io/pypi/v/tipmip-toad)](https://pypi.org/project/tipmip-toad/)

```bash
$ pip install tipmip-toad
```

or

```bash
$ git clone https://github.com/tipmip-methods/toad.git
$ cd toad
$ pip install .
```

### Simple use case

```python
from toad import TOAD
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN


# init TOAD object with ice-thickness field and a custom time dimension.
td = TOAD("ice_thickness.nc", time_dim="GMST")

# Compute shifts for variable 'thk' using the method ASDETECT (Boulton & Lenton, 2019)
td.compute_shifts("thk", method=ASDETECT(timescale=(0.5, 3.5)))

# Compute clusters using HDBSCAN from scikit-learn (McInnes, 2017)
td.compute_clusters(
    var="thk",
    method=HDBSCAN(min_cluster_size=25),
    time_weight=2.0,
)

# Plot largest clusters in ccrs.SouthPolarStereo() projection
td.plot.overview("thk", map_style={"projection": "south_pole"}, mode="aggregated");
```

<div align="center">
    <img src="https://raw.githubusercontent.com/tipmip-methods/toad/main/docs/source/resources/cluster_overview_example.png" width="1000px" alt="Cluster Overview Example">
</div>

For more details, check out the tutorials.

## Tutorials

- [Basics](https://github.com/tipmip-methods/toad/blob/main/tutorials/basics.ipynb): Learn the core concepts and workflow
- [Aggregation](https://github.com/tipmip-methods/toad/blob/main/tutorials/consensus.ipynb): Explore how to identify regions of abrupt shifts that are consistent across models/members/variables/methods. 
- [Custom clustering](https://github.com/tipmip-methods/toad/blob/main/tutorials/custom_clustering.ipynb): Implement new clustering methods
- [Custom event detection](https://github.com/tipmip-methods/toad/blob/main/tutorials/custom_shifts_detection.ipynb): Implement new event detection algorithms

### Community additions

Alternative detection methods contributed to TOAD. These are not part of the peer-reviewed TOAD methodology, but may be useful for comparison or follow-up work.

- [EDGE shift detection](https://github.com/tipmip-methods/toad/blob/main/tutorials/edge_detection.ipynb): Alternative abrupt-shift detector based on Bathiany et al. (2020) and Terpstra et al. (2025); TOAD implementation by [Sjoerd Terpstra](https://github.com/TerpstraS). See the tutorial for usage notes

## Documentation

Read the [documentation](https://toad-docs.pages.dev), which includes a short video introduction!

## About

TOAD is developed at the [Potsdam Institute for Climate Impact Research (PIK)](https://www.pik-potsdam.de/en) and the [Max Planck Institute of Geoanthropology](https://www.gea.mpg.de/). The project originated from early prototype work by [Sina Loriani](https://www.pik-potsdam.de/members/sinal) in 2022. Since 2024, [Jakob Harteg](https://www.pik-potsdam.de/members/jakobha) has led the full development of the package as part of his PhD project. Over time, numerous contributors have played important roles at various stages, including [Lukas Röhrich](https://www.pik-potsdam.de/members/lukasro) and [Fritz Kühlein](https://www.pik-potsdam.de/members/fritzku/homepage). The project has also benefited greatly from scientific advice and guidance from [Sina Loriani](https://www.pik-potsdam.de/members/sinal), [Jonathan Donges](https://www.pik-potsdam.de/members/donges), [Ricarda Winkelmann](https://www.pik-potsdam.de/members/ricardaw), and many others. Community contributions, such as feature suggestions, bug reporting, or even extensions like new detection algorithms, are very welcome. 

## Citation

If you use TOAD in your research, please cite:

**TOAD package:**
Harteg, J., Roehrich, L., De Maeyer, K., Garbe, J., Sakschewski, B., Klose, A. K., Donges, J., Winkelmann, R., and Loriani, S.: TOAD: Tipping and Other Abrupt events Detector, Zenodo, https://doi.org/10.5281/zenodo.18316437, 2026.

**TOAD methodology paper (preprint):**
Harteg, J., Röhrich, L., De Maeyer, K., Garbe, J., Sakschewski, B., Klose, A. K., Donges, J. F., Winkelmann, R., and Loriani, S.: TOAD v1.0: A Python Framework for Detecting Abrupt Shifts and Coherent Spatial Domains in Earth-System Data, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2026-356, 2026.

## License

TOAD is licensed under the BSD 2-Clause License. See [LICENSE.txt](LICENSE.txt) for details.

## References

- Boulton, C. A., & Lenton, T. M. (2019). A new method for detecting abrupt shifts in time series. F1000Research, 8, 746. [https://doi.org/10.12688/F1000RESEARCH.19310.1](https://doi.org/10.12688/F1000RESEARCH.19310.1)
- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. J. Open Source Softw., 2(11), 205. [https://doi.org/10.21105/joss.00205](https://doi.org/10.21105/joss.00205)
- The scikit-learn developers. (2025). scikit-learn. Zenodo. [https://doi.org/10.5281/zenodo.591564](https://doi.org/10.5281/zenodo.591564)
