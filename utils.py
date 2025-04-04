"""Utility functions using firecrown infraestructure to build the sacc file."""

# Standard library imports
from collections import defaultdict
from typing import Dict, List
import warnings
import re

# Third-party library imports
import numpy as np
import pyccl as ccl
import sacc

# Firecrown imports
from firecrown.ccl_factory import (
    CCLFactory,
    CAMBExtraParams,
    PoweSpecAmplitudeParameter
)
from firecrown.generators.inferred_galaxy_zdist import (
    LinearGrid1D,
    ZDistLSSTSRD,
    Y1_LENS_BINS,
    Y1_SOURCE_BINS,
    Y10_LENS_BINS,
    Y10_SOURCE_BINS,
)
from firecrown.metadata_types import (
    TwoPointXY,
    TwoPointHarmonic,
    Galaxies,
    InferredGalaxyZDist
)
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap

# Augur imports
from augur.utils.cov_utils import TJPCovGaus

def build_modeling_tools(config: Dict[str, Dict]) -> ModelingTools:
    """
    Create a ModelingTools object from the configuration.

    This function processes the configuration dictionary, adjusts cosmology
    parameters, and constructs a ModelingTools object with the specified
    cosmology and systematics parameters set in pyccl.

    The function performs the following steps:
    1. Extracts cosmology fiducial values and factory configuration from the
       input dictionary.
    2. Adjusts the cosmology parameters, including calculating Omega_c and
       handling neutrino mass with Omega_m are defined instead of Omega_c.
    3. Determines the amplitude parameter (A_s or sigma8) for the power
       spectrum based on the input configuration.
    4. Initializes CAMB extra parameters if provided in the configuration.
    5. Creates and prepares the ModelingTools object with the specified
       parameters.
    6. Returns the ModelingTools object with the cosmology parameters set.

    Args:
        config (Dict[str, Dict]): Dictionary containing cosmology and
                                  systematics parameters.

    Returns:
        ModelingTools: Modeling tools object with cosmology parameters.

    Raises:
        ValueError: If the amplitude parameter is neither A_s nor sigma8.
    """
    cosmo_config = config["cosmology"]
    # Adjust cosmology parameters.
    if "Omega_m" in cosmo_config:
        cosmo_config["Omega_c"] = cosmo_config["Omega_m"] - cosmo_config["Omega_b"]
        cosmo_config.pop("Omega_m", None)

        omega_nu = (
            cosmo_config["m_nu"].get("fid_value")
            / (93.14 * cosmo_config["h"].get("fid_value") ** 2)
        )
        cosmo_config["Omega_c"]["fid_value"] -= omega_nu

    # Determine the amplitude parameter for the power spectrum.
    amplitude_param = (
        PoweSpecAmplitudeParameter.AS if "A_s" in cosmo_config
        else PoweSpecAmplitudeParameter.SIGMA8
    )

    # Initialize CAMB extra parameters if provided.
    camb_extra_params = None
    if "extra_parameters" in cosmo_config:
        camb_extra_params = CAMBExtraParams(
            **cosmo_config.get("extra_parameters", {}).get("camb", {})
        )

    # Create and update the ModelingTools object.
    tools = ModelingTools(
        ccl_factory=CCLFactory(
            require_nonlinear_pk=True,
            amplitude_parameter=amplitude_param,
            mass_split=cosmo_config.get("mass_split"),
            camb_extra_params=camb_extra_params,
        )
    )

    cosmo_aux = {}
    for key, value in cosmo_config.items():
        if key in ("mass_split", "extra_parameters"):
                continue
        cosmo_aux[key] = value
    tools.update(ParamsMap(cosmo_aux))
    tools.prepare()

    return tools


def get_ext_dist_binned(
    distribution_path: str, tracer_name: str, measurements: Galaxies
) -> List[InferredGalaxyZDist]:
    """
    Import ext binned distribution to build the InferredGalaxyZDist objects.

    This function reads a .txt file containing binned redshift distributions
    and constructs InferredGalaxyZDist objects for each bin. The file should
    contain redshift values in the first row and corresponding dN/dz values
    in subsequent rows.

    Args:
        distribution_path (str): Path to the file containing the binned dN/dz
                                 values.
        tracer_name (str): Name of the tracer (e.g., 'lens', 'src').
        measurements (Galaxies): Type of measurements associated with each
                                 inferred distribution.

    Returns:
        List[InferredGalaxyZDist]: A list of InferredGalaxyZDist objects
                                   representing
        the binned redshift distributions.

    Raises:
        ValueError: If the input distribution file is invalid or contains
                    insufficient data.
    """
    # Load the binned redshift distribution from the file.
    dndz_binned = np.loadtxt(distribution_path)
    if dndz_binned.shape[0] < 2:
        raise ValueError(
            "Input distribution must have at least one redshift bin "
            "and one dndz value."
        )

    # Create InferredGalaxyZDist objects for each bin.
    z_array = dndz_binned[0]
    dndz_distributions = dndz_binned[1:]
    dist_list = [
        InferredGalaxyZDist(
            bin_name=f"{tracer_name}{i}",
            z=z_array,
            dndz=dndz,
            measurements={measurements},
        )
        for i, dndz in enumerate(dndz_distributions)
    ]

    return dist_list


def get_srd_dist_binned(
    z: np.ndarray, tracer_name: str, year: str
) -> List[InferredGalaxyZDist]:
    """
    Get the binned redshift distribution for a lens or source for SRD(Y1/Y10).

    This function retrieves the binned redshift distribution for a specified
    tracer (either lens or source) from the LSST Science Requirements Document
    (SRD) for Year 1 (Y1) or Year 10 (Y10). The function uses Firecrown repo's
    predefined redshift distributions and bin edges to create
    InferredGalaxyZDist objects.

    Args:
        z (np.ndarray): Array of redshift values.
        tracer_name (str): Name of the tracer (e.g., 'lens' for lens
                            photometric or 'src' for source photometric).
        year (str): Year of the survey ('1' for Year 1, '10' for Year 10).

    Returns:
        List[InferredGalaxyZDist]: List of InferredGalaxyZDist objects
                                    representing the binned redshift
                                    distributions.

    Raises:
        ValueError: If the tracer type is invalid or not recognized.
    """
    # auxiliary dictionary that maps tracer type to the corresponding
    # year-binned distributions information.
    year_binned_distributions = {
        "lens": {
            "1": (ZDistLSSTSRD.year_1_lens, Y1_LENS_BINS, Galaxies.COUNTS),
            "10": (ZDistLSSTSRD.year_10_lens, Y10_LENS_BINS, Galaxies.COUNTS),
        },
        "source": {
            "1": (
                ZDistLSSTSRD.year_1_source,
                Y1_SOURCE_BINS,
                Galaxies.SHEAR_E
            ),
            "10": (
                ZDistLSSTSRD.year_10_source,
                Y10_SOURCE_BINS,
                Galaxies.SHEAR_E
            ),
        },
    }

    # Check if the tracer type is valid.
    if "lens" in tracer_name:
        tracer_type = "lens"
    elif "src" in tracer_name or "source" in tracer_name:
        tracer_type = "source"
    else:
        raise ValueError(f"Invalid tracer type: {tracer_name}")

    # Extract the binned distribution information for the specified
    # year and type.
    zdist_func, bins, measurements = year_binned_distributions[
        tracer_type][year]
    zdist = zdist_func(use_autoknot=False)

    # Create InferredGalaxyZDist objects for each bin.
    zdist_list = [
        zdist.binned_distribution(
            zpl=bins['edges'][i],
            zpu=bins['edges'][i + 1],
            sigma_z=bins["sigma_z"],
            z=z,
            name=f"{tracer_name}{i}",
            measurements={measurements},
        )
        for i in range(len(bins['edges']) - 1)
    ]

    return zdist_list


def _process_distribution(
    tracer_name: str,
    distribution: str,
    z_arr: np.array
) -> List[InferredGalaxyZDist]:
    """
    Process the redshift distribution for a given tracer.

    This function determines the appropriate method to generate the redshift
    distribution based on the provided distribution name in config. It supports
    predefined SRD distributions and external distributions from .txt files.

    Args:
        tracer_name (str): The name of the tracer (e.g., 'lens', 'src',
                           'spec_bgs', 'spec_lrg', 'spec_elg' ).
        distribution (str): The type of distribution, either 'SRD_Y1',
                            'SRD_Y10', or a filename ending in '.txt'.
        z_arr (np.array): The redshift array used for binning the distribution.

    Returns:
        List[InferredGalaxyZDist]: A list of inferred galaxy redshift
                                   distributions.

    Raises:
        ValueError: If the distribution type is invalid or unsupported.
    """
    distr_list = []
    # Check if the distribution is predefined SRD or external file.
    if distribution in ("SRD_Y1", "SRD_Y10"):
        year = distribution[-1]  # Extract year from distribution string.
        tracer_type = "lens" if "lens" in tracer_name else "src"
        distr_list.extend(get_srd_dist_binned(z_arr, tracer_type, year))
    elif distribution.endswith(".txt"):
        tracer_map = {
            "lens": Galaxies.COUNTS,
            "src": Galaxies.SHEAR_E,
            "spec_bgs": Galaxies.COUNTS,
            "spec_lrg": Galaxies.COUNTS,
            "spec_elg": Galaxies.COUNTS
        }
        for key, galaxy_type in tracer_map.items():
            if key in tracer_name:
                distr_list.extend(
                    get_ext_dist_binned(
                        distribution, key, galaxy_type))

    # Raise an error if the distribution type is invalid.
    else:
        raise ValueError(
            f"Invalid distribution specified for tracer {tracer_name}. "
            "Use a valid file path for a lens/source tracer or choose "
            "'SRD_Y1'/'SRD_Y10' for predefined distributions."
        )
    # Check if the redshift array length matches the distribution length
    # for consistency.
    assert all(len(dist.z) == len(z_arr) for dist in distr_list), \
        f"z array length mismatch in {[dist.bin_name for dist in distr_list]}"

    return distr_list


def get_redshift_distribution(
        config_array: Dict[str, Dict],
        config_probes: Dict[str, Dict]) -> List[InferredGalaxyZDist]:
    """
    Generate the redshift distribution based on the configuration file.

    This function reads the configuration dictionary and constructs redshift
    distributions for tracers using either predefined SRD or external files.

    The function performs the following steps:
    1. Extracts the redshift array parameters from the config for consistency.
    2. Generates the redshift array using a linear grid.
    3. Iterates over the probes in the configuration to process each tracer'
       distribution.
    4. Uses predefined SRD distributions or external file to build the redshift
       distributions.
    5. Returns a list of inferred galaxy redshift distributions.

    Args:
        config (Dict[str, Dict]): A dictionary containing redshift
                                  distribution parameters.

    Returns:
        List[InferredGalaxyZDist]: A list of inferred galaxy redshift
                                   distributions.

    Raises:
        ValueError: If an invalid or unsupported distribution type is
                    encountered.
    """
    # Generate the redshift array using the configuration parameters.
    config_z = config_array["z_array"]
    z_ = LinearGrid1D(
        start=config_z["z_start"],
        end=config_z["z_stop"],
        num=config_z["z_number"]
    )
    z_arr = z_.generate()

    # Process the redshift distribution for each tracer.
    distribution_list = []
    for key, tracer_config in config_probes["probes"].items():
        if key == "overlap":
            continue  # Skip overlap handling if not needed.
        for tracer_name, tracer_data in tracer_config["tracers"].items():
            distribution_list.extend(
                _process_distribution(
                    tracer_name, tracer_data["distribution"], z_arr)
            )

    return distribution_list


def build_twopointxy_combinations(
    distribution_list: List[InferredGalaxyZDist],
    tracer_combinations: Dict[str, Dict]
) -> Dict[str, List[TwoPointXY]]:
    """
    Generate TwoPointXY combinations from tracer combinations and distribution.

    This function creates TwoPointXY objects by matching tracer combinations
    with their corresponding redshift distributions and measurements. It
    iterates over the tracer combinations, finds the matching redshift
    distributions, and constructs TwoPointXY objects for each valid
    combination.

    Args:
        distribution_list (List[InferredGalaxyZDist]): List of inferred galaxy
            redshift distributions, each containing redshift and measurement
            data.
        tracer_combinations (Dict[str, Dict]): Dictionary containing tracer
            combinations for each tracer. The keys are tracer names, and the
            values are dictionaries with combination details.

    Returns:
        Dict[str, List[TwoPointXY]]: A dictionary where the keys are tracer
        names and the values are lists of TwoPointXY objects representing the
        two-point correlations between the tracers.
    """
    # Create TwoPointXY objects for each tracer combination.
    all_two_point_combinations = {}
    for tracer, comb_data in tracer_combinations.items():
        all_two_point_combinations[tracer] = []
        # Iterate over the combinations and find the corresponding dist names.
        for comb in comb_data["combinations"]:
            x = f"{tracer.split('-')[0]}{comb[0]}"
            y = f"{tracer.split('-')[1]}{comb[1]}"
            x_measurement, y_measurement = None, None
            x_dist, y_dist = None, None
            # Find the corresponding distributions and measurament for the
            # tracer combination.
            for sample in distribution_list:
                if sample.bin_name == x:
                    x_dist = sample
                    x_measurement = next(iter(x_dist.measurements))
                if sample.bin_name == y:
                    y_dist = sample
                    y_measurement = next(iter(y_dist.measurements))
            if x_measurement and y_measurement:
                all_two_point_combinations[tracer].append(
                    TwoPointXY(
                        x=x_dist,
                        y=y_dist,
                        x_measurement=x_measurement,
                        y_measurement=y_measurement,
                    )
                )

    return all_two_point_combinations


def build_metadata_cells(
    config: Dict,
    two_point_comb: Dict[str, List[TwoPointXY]],
    cosmo: ccl.Cosmology,
    ells: np.ndarray,
) -> List[TwoPointHarmonic]:
    """
    Create a list of TwoPointHarmonic objects from the TwoPointXY objects.

    This function applies the scale cuts defined in the configuration file and
    computes the corresponding ells for each pair of tracers in the TwoPointXY
    objects. It iterates over the two point combinations, applies the scale
    cuts, and constructs TwoPointHarmonic objects with the computed ells.

    Args:
        config (Dict): Configuration dictionary containing scale cut
                       parameters.
        two_point_comb (Dict[str, List[TwoPointXY]]): Dictionary of TwoPointXY
                                                      objects
            indexed by tracer names.
        cosmo (ccl.Cosmology): Cosmology object from CCL.
        ells (np.ndarray): Array of ell values.

    Returns:
        List[TwoPointHarmonic]: List of TwoPointHarmonic objects with applied
                                scale cuts.
    """
    # Apply scale cuts and create TwoPointHarmonic objects.
    two_points_cells = []
    scale_cuts_config = config["scale_cuts"]
    for twp_ in two_point_comb.values():
        for xy in twp_:
            ells_cut_list = []
            # Apply scale cuts based on the configuration for each
            # tracer in the pair.
            x_name = re.sub(r'\d', '', xy.x.bin_name)
            y_name = re.sub(r'\d', '', xy.y.bin_name)
            for tracer in [x_name, y_name]:
                scale_cut = scale_cuts_config.get(tracer)
                if scale_cut == "None":
                    warnings.warn(
                        f"Scale cut not found for {tracer}. "
                        "Applying no scale cut."
                    )
                    ells_cut_list.append(np.max(ells))
                    continue
                if "kmax" in scale_cut:
                    z_avg = np.average(
                        xy.x.z, weights=xy.x.dndz / np.sum(xy.x.dndz)
                    )
                    a = 1.0 / (1 + z_avg)
                    ells_cut_list.append(
                        np.min(
                            scale_cut["kmax"]
                            * ccl.comoving_radial_distance(cosmo, a)
                            - 0.5
                        )
                    )
                elif "lmax" in scale_cut:
                    ells_cut_list.append(scale_cut["lmax"])
                else:
                    raise ValueError("Invalid scale cut configuration.")
            # Apply the minimum scale cut to the ells
            scale_cut_apply = np.min(ells_cut_list)
            ells_cut = ells[ells <= scale_cut_apply].astype(np.int32)
            two_points_cells.append(TwoPointHarmonic(XY=xy, ells=ells_cut))

    return two_points_cells


def build_sacc_file(
    tools: ModelingTools, distribution_list: list, two_point_functions: list
) -> sacc.Sacc:
    """
    Build the Sacc object by adding tracers and two-point functions.

    This function initializes a Sacc object and populates it with tracers and
    two-point functions. It first adds tracers from the provided distribution
    list, then groups and sorts the two-point functions by galaxy type, and
    finally adds the two-point functions to the Sacc object along with their
    computed theory vectors.

    Args:
        tools (ModelingTools): Modeling tools object containing cosmology and
                               related methods.
        distribution_list (list): List of InferredGalaxyZDist objects with
                                  redshift distributions.
        two_point_functions (list): List of TwoPointXY objects containing
                                    two-point function data.

    Returns:
        sacc.Sacc: The populated Sacc object with tracers and two-point
                   functions.
    """
    # Initialize the Sacc object
    sacc_data = sacc.Sacc()
    sacc_data.metadata["info"] = "Mock data vector and covariance matrix"

    # Add tracers to the Sacc object from the distribution list
    for sample in distribution_list:
        quantity = next(iter(sample.measurements))
        tracer_type = (
            "galaxy_density" if quantity.name == "COUNTS" else "galaxy_shear"
        )
        sacc_data.add_tracer(
            "NZ",
            sample.bin_name,
            quantity=tracer_type,
            z=sample.z,
            nz=sample.dndz
        )

    # Group two-point functions by galaxy type and sort by tracer names
    # to ensure consistent ordering.
    two_point_functions_dict = defaultdict(list)
    for tw in two_point_functions:
        galaxy_type = tw.sacc_data_type
        two_point_functions_dict[galaxy_type].append(
            (tw.sacc_tracers.name1, tw.sacc_tracers.name2, tw)
        )
  
    for tracer_pairs_ in two_point_functions_dict.values():
        tracer_pairs_.sort(key=lambda x: (x[0], x[1]))

    # check if the two-point functions dict are repeating some tracer pairs
    for galaxy_type, tracer_pairs in two_point_functions_dict.items():
        tracer_pairs_names = [f"{tracer0}-{tracer1}" for tracer0, tracer1, _ in tracer_pairs]
        if len(tracer_pairs_names) != len(set(tracer_pairs_names)):
            raise ValueError(f"Repeated tracer pairs in {galaxy_type} two-point functions")

    # Add the two-point functions to the Sacc object
    for galaxy_type, tracer_pairs in two_point_functions_dict.items():
        for tracer0, tracer1, tw in tracer_pairs:
            sacc_data.add_ell_cl(
                galaxy_type,
                tracer0,
                tracer1,
                tw.ells,
                tw.compute_theory_vector(tools)
            )

    return sacc_data


def build_tjpcov_dict(
    tools: ModelingTools,
    sacc_data: sacc.Sacc,
    config: Dict,
    ells_edges: np.ndarray,
) -> Dict:
    """
    Build the configuration dictionary for the TJPCov covariance calculation.

    This function constructs a configuration dictionary required for the
    covariance calculation using the Firecrown package. It processes the
    input configuration, extracts necessary parameters, and organizes them
    into a structured dictionary.
    
    Args:
 	    tools (ModelingTools): Modeling tools object containing cosmology and
                               related methods.
        sacc_data (sacc.Sacc): Sacc object containing the data and tracer
                               information.
        config (Dict): Configuration dictionary with parameters for fsky and
                       tracers.
        ells_edges (np.ndarray): Array of ell edges for the covariance
                                 calculation.

    Returns:
        Dict: Dictionary with the required configuration for the covariance
              calculation.
    """
    
    cov_config = {"tjpcov": {
        "cosmo": tools.ccl_cosmo,
        "sacc_file": sacc_data,
        "fsky": config.probes_config["probes"]["lsst"]["fsky"],
        "binning_info": {"ell_edges": ells_edges}
    }}

    # Update the dictionary with the tracer's parameters
    for tracer_name in sacc_data.tracers:
        if tracer_name.startswith("lens"):
            lens_config = (
                config.probes_config["probes"]
                ["lsst"]["tracers"]["lens"]
            )
            cov_config["tjpcov"].update(
                {
                    f"Ngal_{tracer_name}": (
                        lens_config["ngal"][int(tracer_name[-1])]
                        ),
                    f"bias_{tracer_name}": (
                        lens_config["bias"]["fid_value"]
                        [int(tracer_name[-1])]
                    ),
                }
            )

        if tracer_name.startswith("src"):
            src_config = (
                config.probes_config["probes"]
                ["lsst"]["tracers"]["src"]
            )
            cov_config["tjpcov"].update(
                {
                    f"Ngal_{tracer_name}": (
                        src_config["ngal"][int(tracer_name[-1])]
                    ),
                }
            )
            if "sigma_e" in src_config:
                cov_config["tjpcov"].update(
                    {f"sigma_e_{tracer_name}": (
                        src_config["sigma_e"][int(tracer_name[-1])]
                        )
                    }
                )
            if "ia" in src_config:
                cov_config["tjpcov"].update(
                    {"IA": src_config["ia_bias"]["fid_value"]}
                )   # TODO: Calculate the NLA for the TJPCov

    return cov_config


def build_tjpcov_covariance(
    tools: ModelingTools,
    sacc_data: sacc.Sacc,
    config: Dict,
    ells_edges: np.ndarray,
) -> np.ndarray:
    """
    Create a Gaussian fsky covariance matrix using TJPCov.

    This function calculates the Gaussian covariance matrix for a given set of
    tracers and cosmological parameters using the augur modification of the
    FourierGaussianFsky class from TJPCov. It constructs the necessary
    configuration dictionary, initializes thecovariance calculation object,
    and computes the covariance matrix by iterating over tracer combinations.

    Args:
        tools (ModelingTools): Modeling tools object containing cosmology and
                               related methods.
        sacc_data (sacc.Sacc): Sacc object containing the data and tracer
                               information.
        config (Dict): Configuration dictionary with parameters for fsky and
                       tracers.
        ells_edges (np.ndarray): Array of ell edges for the covariance
                                 calculation.
 
    Returns:
        np.ndarray: The calculated covariance matrix.
    """
    # Create the dictionary to be read by the tjpcov package
    tjpcov_config = build_tjpcov_dict(tools, sacc_data, config, ells_edges)
    cov_calc = TJPCovGaus(tjpcov_config)

    # Build the covariance matrix based on the tracers
    tracers_comb = sacc_data.get_tracer_combinations()
    ndata = len(sacc_data.mean)
    cov_matrix = np.zeros((ndata, ndata))

    for i, trs1 in enumerate(tracers_comb):
        ii = sacc_data.indices(tracers=trs1)

        for trs2 in tracers_comb[i:]:
            print(trs1, trs2)
            jj = sacc_data.indices(tracers=trs2)
            ii_all, jj_all = np.meshgrid(ii, jj, indexing="ij")

            cov_blocks = cov_calc.get_covariance_block(
                trs1, trs2, include_b_modes=False
            )
            cov_matrix[ii_all, jj_all] = cov_blocks[: len(ii), : len(jj)]
            cov_matrix[jj_all.T, ii_all.T] = cov_blocks[: len(ii), : len(jj)].T

    return cov_matrix