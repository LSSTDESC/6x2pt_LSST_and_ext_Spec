# LSST 6x2pt Fisher Matrix Analysis Toolkit

[![LSST DESC](https://img.shields.io/badge/LSST-DESC-blueviolet)](https://lsstdesc.org)

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [File Structure](#file-structure)
7. [Contributing](#contributing)

## Introduction

This toolkit provides comprehensive Fisher matrix analysis for LSST 3x2pt probes combined with spectroscopic surveys (DESI, 4MOST). The package enables cosmological parameter forecasting and systematic error analysis for various survey combinations and binning strategies.

## Features

- _Multiple survey combinations_:
  - LSST 3x2pt
  - 6x2pt (LSST + DESI/4MOST)
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

```bash
git clone https://github.com/LSSTDESC/6x2pt_LSST_and_ext_Spec.git
cd 6x2pt_LSST_and_ext_Spec
```

### Requirements

- [Firecrown](https://firecrown.readthedocs.io/en/latest/developer_installation.html) = 1.11  
  ⚠️ After installation, you must manually modify a regular expression in `firecrown/metadata_types.py` to support spectroscopic tracers.  
  Update the following line:

  ```diff
  - LENS_REGEX = re.compile(r"^lens\d+$")
  + LENS_REGEX = re.compile(r"^(lens\d+|spec\d+|spec_bgs\d+|spec_lrg\d+|spec_elg\d+)$")

  ```

- [Pyccl](https://github.com/LSSTDESC/CCL) > 3.0.2
- [Augur](https://github.com/LSSTDESC/augur) = 1.1.0

## Usage

The main pipeline is executed via:

```bash
cd runs/run_name/
python run_pipeline.py general.yaml
```

Available runs include:

- 3x2pt_LSST
- 6x2pt*LSST_DESI*[BGS/ELG/LRG]\_0.2bin
- 6x2pt*LSST_DESI_FULL*[0.2/0.1/0.05]bin
- 6x2pt\*LSST_4MOST_FULL_0.2bin

## Configuration

The package uses YAML configuration files to control all aspects of the pipeline. These are organized by

1. general.yaml - Controls the pipeline run (output path, pipeline choices, YAML configuration choices, etc)
2. probes_properties.yaml - Sets survey and tracer properties (e.g, inclusion RSD, Distribution path, fsky value, nuisance parameters, etc)
3. probes_combination.yaml - Defines which probe combinations are included in the data vector
4. array_choices.yaml - Redshif and $\ell$ arrays, scale cuts
5. cosmology.yaml - Cosmological parameters values
6. prior_choices/\*.yaml - Prior values for parameters (for the nuisance parameters can be shared or tracer-specific)
7. range_choices/\*.yaml - Range choices for Fisher cosmological derivatives

example configuration hierarchy:

```bash
runs/
└── run_name/
    ├── config_yamls/
    │   ├── array_choices.yaml
    │   ├── cosmology.yaml
    │   ├── probes_combination.yaml
    │   ├── probes_properties.yaml
    │   ├── range_choices
    │   │└── range_choice.yaml
    │   └── prior_choices
    │	   └── prior_choice.yaml
    └── general.yaml
```

## File structure

Key components of the repository:

```bash
6x2pt_LSST_and_ext_Spec/
├── functions_setup
│   ├── config_builder.py           # Central config manager
│   ├── fourrier_covariance_fsky.py # Gaussian covariance using Firecrown infrastructure (adapted from Tjpcov covariance_builder)
│   ├── likelihood_build.py         # Likelihood builder
│   ├── sacc_generator.py           # SACC file generator
│   ├── spec_dndz_config/           # Precomputed dN/dz files for DESI/4MOST surveys
│   ├── utils.py                    # Utility functions for the Sacc file generator
│   ├── runs/                       # runs dictionary (3x2pt, 6x2pt, etc.)
└── one_covariance_setup/       # dictionary with OneCovariance files (not yet implemented!!!!)
```

#### Note

To use the notebooks in the `spec_dndz_config/` directory for generating the redshift distribution of the DESI DR1/EDR sample, make sure the `.txt` histogram files—produced by the [desitutorials](https://github.com/desihub/tutorials)—are placed in the `auxiliar_files/` directory.

Additionally, create a sacc_files/ directory to store the generated SACC files.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
