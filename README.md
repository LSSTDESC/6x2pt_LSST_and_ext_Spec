# LSST 6x2pt Fisher Matrix Analysis Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)
[![LSST DESC](https://img.shields.io/badge/LSST-DESC-blueviolet)](https://lsstdesc.org)

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [File Structure](#file-structure)
7. [Examples](#examples)
8. [Contributing](#contributing)
9. [License](#license)
10. [Citation](#citation)

## Introduction

This toolkit provides comprehensive Fisher matrix analysis for LSST 3x2pt probes combined with spectroscopic surveys (DESI, 4MOST). The package enables cosmological parameter forecasting and systematic error analysis for various survey combinations and binning strategies.

## Features

- _Multiple survey combinations_:
  - LSST 1x2pt (lensing or clustering only)
  - LSST 3x2pt (full combination)
  - 6x2pt (LSST + spectroscopic surveys)
- _Flexible configuration_:
  - Multiple binning strategies (0.05, 0.1, 0.2 redshift bins)
  - Various cosmological models (ΛCDM, w₀wₐ, mν)
  - With/without nuisance parameters
- _Comprehensive outputs_:
  - Fisher matrices
  - Covariance matrices
  - SACC files
  - Forecast visualizations

## Installation

### Requirements

- python > 3.11.11
- firecrown > 1.8
- pyccl > 3.0.2
- onecovariance > 1.0.1
- augur > 0.5.0

### Setup

```bash
git clone https://github.com/LSSTDESC/6x2pt_LSST_and_ext_Spec.git
cd 6x2pt_LSST_and_ext_Spec
```

## Usage

The main pipeline is executed via:

```bash
python run_pipeline.py general.yaml
```

Available repositories include:

- 1x2pt_lens_LSST
- 1x2pt_src_LSST
- 3x2pt_LSST
- 6x2pt*LSST_DESI*[BGS/ELG/LRG]\_0.2bin
- 6x2pt*LSST_DESI_FULL*[0.2/0.1/0.05]bin

## Configuration

The package uses YAML configuration files organized in three levels:

1. general.yaml - Controls the pipeline run (output path, pipeline stages, YAML configuration choices)
2. probes_properties.yaml - Sets survey and tracer properties (e.g inclusion RSD, Distribution, fsky, nuisance parameters, etc)
3. probes_combination.yaml - Defines which probes combination are include in the data vector
4. array_choices.yaml - Redshif and $\ell$ array, scale cuts
5. cosmology.yaml - Cosmological parameters definitions
6. prior_choices/\*.yaml - Prior values for parameters (for the nuisance parameters can be shared or tracer-specific)

example configuration hierarchy:

```bash
runs/
└── run_name/
    ├── config_yamls/
    │   ├── cosmology.yaml
    │   ├── probes_combination.yaml
    │   ├── probes_properties.yaml
    │   └── prior_choices
    │	   └── prior_choice.yaml
    └── general.yaml
```

## File strucutre

key components of the repository:

```bash
6x2pt_LSST_and_ext_Spec/
├── config_builder.py           # Central config manager
├── covariance.py               # Wrapper for OneCovariance (copied from OneCovariance)
├── fourrier_covariance_fsky.py # Gaussian covariance using Firecrown infraestructure (copied Tjpcov covariance builder)
├── likelihood_build.py         # Likelihood builder
├── OneCoveriance_builder.py    # Interface to OneCovariance (.ini creation, etc.)
├── sacc_generator.py           # SACC file generator
├── spec_dndz_config/           # Precomputed dN/dz files for spectroscopic surveys
├── utils.py                    # Utility functions to Sacc file generator
└── runs/                       # runs dictionary (1x2pt, 3x2pt, 6x2pt, etc.)
```

## Examples

Coming soon - will cinlude example runs and visual outputs

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

TODO!

## Citation

TODO!
