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
- *Multiple survey combinations*:
  - LSST 1x2pt (lensing or clustering only)
  - LSST 3x2pt (full combination)
  - 6x2pt (LSST + spectroscopic surveys)
- *Flexible configuration*:
  - Multiple binning strategies (0.05, 0.1, 0.2 redshift bins)
  - Various cosmological models (ΛCDM, w₀wₐ, mν)
  - With/without nuisance parameters
- *Comprehensive outputs*:
  - Fisher matrices
  - Covariance matrices
  - SACC files
  - Forecast visualizations

## Installation

### Requirements
- 
- 

### Setup
```bash
git clone https://github.com/LSSTDESC/6x2pt_LSST_and_ext_Spec.git
cd 6x2pt_LSST_and_ext_Spec
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

## Usage
The main pipeline is executed via:

```bash
python run_pipeline.py general.yaml

Available repositories include:

* 1x2pt_lens_LSST
* 1x2pt_src_LSST
* 3x2pt_LSST
* 6x2pt_LSST_DESI_[BGS/ELG/LRG]_0.2bin
* 6x2pt_LSST_DESI_FULL_[0.2/0.1/0.05]bin

## Configuration
The package uses YAML configuration files organized in three levels:

1. Main Configuration (general.yaml):
	* controls pipeline configurations
	* set output directories and pipeline choices
	* specifies which yamls used in the analysis
2. Probes configurations (probes_properties.yaml)
	* Defines survey properties
	* Inclusion of rsd or not
	* Defines nuisance parameters
3. Probes combinations (probes_combination.yaml)
	* Define which tracer combination are being set in the analysis
4. array choices (array_choices.yaml)
	* Define redshift array (if you choose to change, please rerun the spectroscopic notebook for consistency)
	* Define ell bining arrays
	* Define scale cuts for each tracer
5. Cosmology (cosmology.yaml)
	* Define cosmological parameters
	* Defines other additional parameters to structure the cosmology object
6. Prior choices (prior_choices/.)
	* Define priors for the varying parameters in other yamls (for a list of parameters for example: lens{i}_delta_z parameters two options can be done: [x, y] for all i parameters or [[x1, y1], ...] for individual prior for each i parameters 

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
    
## File strucutre
key components of the repository:

```bash
6x2pt_LSST_and_ext_Spec/
├── config_builder.py        # Configuration management to run all pipeline
├── covariance.py            # OneCovariance covariance calculation (copied to run OneCovariance inside the pipeline)
├── fourrier_covariance_fsky.py  # Gaussian covariance calculation (tjpcov.covariance_gaussian_fsky code copy changing to accept new Firecrown infraestructure)
├── likelihood_build.py      # Likelihood construction
├── OneCoveriance_builder.py # OneCovariance interface script (create .ini file and reconstruct the OneCovariance covariance)
├── runs/                   # All survey configurations
│   ├── 1x2pt_*/           # Single probe analyses
│   ├── 3x2pt_LSST/        # LSST-only analysis
│   └── 6x2pt_*/           # Combined analyses
├── sacc_generator.py       # SACC file generation
├── spec_dndz_config/       # Spectroscopic survey n(z)'s
└── utils.py                # Utility functions to run Sacc generator

