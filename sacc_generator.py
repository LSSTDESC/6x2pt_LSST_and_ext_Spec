"""Sacc generator Module to create a sacc file using the bellow infras."""

# Third-party library imports
import time
import numpy as np

#  Firecrown imports
from firecrown.utils import base_model_from_yaml
import firecrown.likelihood.number_counts as nc
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.two_point as tp
from firecrown.parameters import ParamsMap

# Importing the functions from config_builder.py
from config_builder import ConfigBuilder

# Importing the functions from utils.py
from utils import (
    build_modeling_tools,
    get_redshift_distribution,
    build_twopointxy_combinations,
    build_metadata_cells,
    build_sacc_file,
    build_tjpcov_covariance
)

# Importing covariance functions
from OneCoveriance_builder import BuildCovWithOneCovariance
from fourrier_covariance_fsky import FirecrownFourierGaussianFsky


def sacc_generator(cfg: ConfigBuilder):
    """Generate a Sacc file based on the provided configuration file.

    This function performs the following steps:
        1. Get a configuration dictionary and extract general settings
           (e.g., run name, file paths for cosmology, array choices, probes,
           and probe combinations).
        2. Build Modeling Tools object using the configuration cosmology and
           compute the non-linear power spectrum.
        3. Create a mapping of firecrown parameters for systematics and
           cosmology.
        4. Generate redshift distributions for each tracer.
        5. Construct the TwoPointXY objects for the probes combinations
           defined in the configuration.
        6. Define multipole (ell) bins, either logarithmically or linearly spaced,
           between configured start and stop values.
        7. Instantiate weak lensing and number counts factories via their
           respective configuration files.
        8. Build metadata for the two-point harmonic analysis with scale cuts
           and create the corresponding TwoPoint objects.
        9. Apply systematics parameters to the TwoPoint objects.
       10. Build the Sacc object using modeling tools, the redshift
           distribution, and the two-point functions.
       11. Compute the covariance matrix and integrate it into the Sacc object.
       12. Save the final Sacc file to disk with a runtime metadata record.

    Parameters:
        cfg (ConfigBuilder): Configuration object containing cosmology, array, and
                            probes settings.

    Returns:
        None

    Side Effects:.
        - Saves the generated Sacc file (including its covariance matrix) as a
          sacc file to the "./sacc_files/" directory.
        - Prints status messages to help track the progress of the Sacc file
          generation.
    """
    # Load configuration and print details
    start_time = time.time()

    sacc_file = cfg.config['general']['sacc_file']
    print(f"Starting {sacc_file} sacc file generation")
    for key in ["cosmology_file",
                "array_choices_file",
                "probes_file",
                "probe_combinations_file"
                ]:
        print(f"{key.replace('_', ' ').capitalize()}: "
              f"{cfg.config['general'][key]}")

    # Build modeling tools and compute non-linear powerspectrum
    print("Building Cosmology...")
    tools = build_modeling_tools(cfg.cosmo_config)
    tools.ccl_cosmo.compute_nonlin_power()

    # pylint: disable=protected-access
    print("Cosmology:")
    for key, value in tools.ccl_cosmo._params_init_kwargs.items():
        print(f"    {key}: {value}")

    # Map firecrown parameters
    params = ParamsMap(cfg.firecrown_params)

    # Build redshift distributions
    print("Building redshift distributions...")
    dists = get_redshift_distribution(cfg.array_config, cfg.probes_config)
    print(f"    {len(dists)} distributions were created")

    # Build TwoPointXY combinations
    twopoint_comb = build_twopointxy_combinations(dists,
                                                  cfg.probes_comb_config,
                                                  )

    # Create ℓ bins inline
    ells_edges = None
    if cfg.array_config["ell_array"]["type"] == "lin":
        if (cfg.config["general"]["cov_builder"].lower() == "tjpcov" or
                cfg.config["general"]["cov_builder"].lower() ==
                "fourriergaussianfsky"):
            ells_edges = np.linspace(
                cfg.array_config["ell_array"]["ell_start"],
                cfg.array_config["ell_array"]["ell_stop"],
                cfg.array_config["ell_array"]["ell_bins"] + 1,
                endpoint=True,
            ).astype(np.int32)
            # Linear average exclusive.
            ells = 0.5 * (ells_edges[:-1] + ells_edges[1:])
        else:
            ells = np.unique(np.linspace(
                cfg.array_config["ell_array"]["ell_start"],
                cfg.array_config["ell_array"]["ell_stop"],
                cfg.array_config["ell_array"]["ell_bins"],
                endpoint=True,
            ).astype(np.int32))
    elif cfg.array_config["ell_array"]["type"] == "log":
        if (cfg.config["general"]["cov_builder"].lower() == "tjpcov" or
                cfg.config["general"]["cov_builder"].lower() ==
                "fourriergaussianfsky"):
            ells_edges = np.unique(
                np.geomspace(
                    cfg.array_config["ell_array"]["ell_start"],
                    cfg.array_config["ell_array"]["ell_stop"],
                    cfg.array_config["ell_array"]["ell_bins"] + 1,
                    endpoint=True,
                )
            ).astype(np.int32)
            # Linear average exclusive.
            ells = 0.5 * (ells_edges[:-1] + ells_edges[1:])
        else:
            ells = np.unique(
                np.geomspace(
                    cfg.array_config["ell_array"]["ell_start"],
                    cfg.array_config["ell_array"]["ell_stop"],
                    cfg.array_config["ell_array"]["ell_bins"],
                    endpoint=True,
                ).astype(np.int32))

    # Build weak lensing, number counts factories and TwoPoint Objects
    print("Building TwoPoint objects...")
    if cfg.factories_config["nc_factory"] is None:
        factories = base_model_from_yaml(wl.WeakLensingFactory,
                                         str(cfg.factories_config["wl_factory"]
                                             ))
        two_point_objects = tp.TwoPoint.from_metadata(
            metadata_seq=build_metadata_cells(cfg.array_config,
                                              twopoint_comb,
                                              tools.ccl_cosmo,
                                              ells),
            wl_factory=factories,
        )
    elif cfg.factories_config["wl_factory"] is None:
        factories = base_model_from_yaml(nc.NumberCountsFactory,
                                         str(cfg.factories_config["nc_factory"]
                                             ))
        two_point_objects = tp.TwoPoint.from_metadata(
            metadata_seq=build_metadata_cells(cfg.array_config,
                                              twopoint_comb,
                                              tools.ccl_cosmo,
                                              ells),
            nc_factory=factories,
        )
    else:
        factories = [
            base_model_from_yaml(nc.NumberCountsFactory,
                                 str(cfg.factories_config["nc_factory"])),
            base_model_from_yaml(wl.WeakLensingFactory,
                                 str(cfg.factories_config["wl_factory"]))
        ]
        two_point_objects = tp.TwoPoint.from_metadata(
            metadata_seq=build_metadata_cells(cfg.array_config,
                                              twopoint_comb,
                                              tools.ccl_cosmo,
                                              ells),
            nc_factory=factories[0],
            wl_factory=factories[1],
        )
    # Apply firecrown parameters to TwoPoint objects
    two_point_objects.update(params)
    print(f"    {len(two_point_objects)} TwoPoint objects were created")

    print("Building Covariance Matrix...")
    # Build Sacc file without Covariance Matrix
    sacc = build_sacc_file(tools, dists, two_point_objects)
    sacc.save_fits(cfg.config['general']['sacc_file'], overwrite=True)

    # Build Covariance Matrix
    cov_builder = cfg.config['general']['cov_builder'].lower()
    cov_matrix = None  # Initialize cov_matrix
    if cov_builder == 'onecovariance':
        print("Calculating covariance with OneCovariance")
        one_cov_builder = BuildCovWithOneCovariance(
            cfg, tools, dists, params, sacc)
        one_cov_builder.build_ini_file()
        one_cov_builder.aux_necessary_files()
        one_cov_builder.build_cov_matrix()
        cov_matrix = one_cov_builder.load_cov_matrix()
    elif cov_builder == 'fourriergaussianfsky':
        print("    Calculating covariance with FourrierGaussianFsky")
        cov_config = FirecrownFourierGaussianFsky(
            tools,
            ell_edges=ells_edges,
            sacc_data=sacc,
            factories=cfg.factories_config,
            parameters=params,
            probes_cfg=cfg.probes_config,
        )
        cov_matrix = cov_config.get_covariance_matrix()
    elif cov_builder == 'tjpcov':
        print("    Calculating covariance with TJPCov")
        cov_matrix = build_tjpcov_covariance(tools, sacc, cfg, ells_edges)
    if cov_matrix is not None:
        # Add Covariance Matrix to Sacc file
        sacc.add_covariance(cov_matrix)
        print("    Covariance Matrix added to Sacc file")
    else:
        raise Warning("No Covariance Matrix added to Sacc file")

    # Save the Sacc file result
    sacc.metadata["total_time"] = time.strftime("%Hh:%Mm:%Ss",
                                                time.gmtime(time.time() -
                                                            start_time))
    sacc.save_fits(cfg.config['general']['sacc_file'], overwrite=True)
    print("Sacc file completed and saved")
    print("Total time:", sacc.metadata["total_time"])
