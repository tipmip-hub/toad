# Test Data Attribution

This repository includes small subsets of data used for automated testing (CI), 
tutorials, and demonstrations of TOAD's capabilities. All data are used under 
permissive licenses that allow redistribution with attribution.

---

## 1. Antarctic Ice Sheet Simulations (PISM)

**Citation**:
Garbe, J., Albrecht, T., Levermann, A., Donges, J. F., and Winkelmann, R.: 
The hysteresis of the Antarctic Ice Sheet, Nature, 585(7826), 538–544, 
https://doi.org/10.1038/s41586-020-2727-5, 2020.

**Usage**: Tutorial demonstrating detection of ice-sheet thinning events and 
spatially coherent collapse regions

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Dataset details**:
- Model: PISM (Parallel Ice Sheet Model)
- Variable: Ice thickness (thk)
- Forcing: Prescribed GMST warming (0 to ~14°C above pre-industrial)
- Temporal resolution: Snapshots along warming trajectory

---

## 2. EC-Earth3-Veg Sea Ice Concentration (CMIP6)

**Citation**:
EC-Earth Consortium (EC-Earth3-Veg) (2020): EC-Earth-Consortium EC-Earth3-Veg 
model output prepared for CMIP6 CMIP 1pctCO2, Version v20200919. Earth System 
Grid Federation, https://doi.org/10.22033/ESGF/CMIP6.4524.

**Usage**: Testing TOAD's support for irregular ocean/sea-ice grids in CI tests

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Dataset details**:
- Model: EC-Earth3-Veg
- Experiment: 1pctCO2 (1% per year CO₂ increase)
- Variable: siconc (Sea Ice Area Percentage)
- Ensemble member: r1i1p1f1
- Version: v20200919
- Grid: native grid (irregular)

---

## 3. E3SM-1-0 Northern Hemisphere March Sea Ice (CMIP6)

**Citation**:
Bader, D. C., Leung, R., Taylor, M., and McCoy, R. B.: E3SM-Project E3SM1.0 
model output prepared for CMIP6 CMIP 1pctCO2, Version 20190718. Earth System 
Grid Federation, https://doi.org/10.22033/ESGF/CMIP6.4490, 2019.

**Model reference**:
Golaz, J.-C., Caldwell, P. M., Van Roekel, L. P., and co-authors: The DOE E3SM 
coupled model version 1: Overview and evaluation at standard resolution, JAMES, 
https://doi.org/10.1029/2018MS001603, 2019.

**Usage**: EDGE shift-detection tutorial (`tutorials/edge_detection.ipynb`). The 
same CMIP6 case is used in the Arctic sea-ice demonstration in the TOAD 
methodology paper (March Northern Hemisphere `siconc`, E3SM-1-0 1pctCO2, 
member r1i1p1f1), including qualitative comparison with Terpstra et al. (2025).

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Dataset details**:
- Model: E3SM-1-0 (E3SM-Project)
- Experiment: 1pctCO2 (1% per year CO₂ increase)
- Variable: siconc (Sea Ice Area Percentage)
- Table: SImon (monthly sea ice)
- Grid: gr (native atmospheric grid)
- Ensemble member: r1i1p1f1
- Version: v20190718
- Subset: March monthly values only; Northern Hemisphere (25.5–89.5°N); model 
  years 1–150 along the 1pctCO2 trajectory
- File: `tutorials/test_data/CMIP.E3SM-Project.E3SM-1-0.1pctCO2.SImon.gr_NH_march.nc`
- Processing: Derived from the CMIP6 archive (intake-esm download, March 
  selection, NH spatial subset). This is the raw tutorial subset; the TOAD paper 
  figure applies additional Gaussian smoothing and masking before ASDETECT 
  analysis.

---

## 4. Synthetic Spatio-Temporal Shifts

**Description**:
Synthetic dataset with known abrupt shifts generated programmatically to provide 
benchmark test cases with controllable spatio-temporal transitions.

**Usage**: 
- Tutorial demonstrating TOAD's ability to recover known shift regions
- CI testing with deterministic, reproducible test cases

**Dataset characteristics**:
- Grid: Regular lat-lon (90 × 180)
- Time steps: 200
- Background: White noise (magnitude 0.025)
- Shift 1: Sigmoid transition at t=50, blob-shaped region centered at (20°N, 75°W)
- Shift 2: Sigmoid transition at t=150, elliptical region centered at (0°, 10°W)
- Spatial correlation: Gaussian smoothing with varying correlation lengths

**License**: Generated synthetically for this project. No external data sources.

---

## General Acknowledgments

### CMIP6 Data
We acknowledge the World Climate Research Programme, which, through its Working 
Group on Coupled Modelling, coordinated and promoted CMIP6. We thank the 
EC-Earth-Consortium, E3SM-Project, and other modeling groups for producing and making available 
their model output, the Earth System Grid Federation (ESGF) for archiving the 
data and providing access, and the multiple funding agencies who support CMIP6 
and ESGF.

Full CMIP6 datasets available through ESGF: https://esgf-data.dkrz.de/

### License Summary
All test data included in this repository are licensed under Creative Commons 
Attribution 4.0 International (CC BY 4.0) or similar permissive licenses. 
Redistribution and use are permitted provided proper attribution is given to 
the original sources as specified above.

For questions about data usage or attribution, please contact the repository 
maintainers.