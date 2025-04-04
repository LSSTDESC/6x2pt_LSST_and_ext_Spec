"""Likelihood builder module for construct likelihood object from config."""

# Sacc import
import sacc

# Firecrown imports
from firecrown.parameters import ParamsMap
from firecrown.utils import base_model_from_yaml
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.weak_lensing import WeakLensingFactory
from firecrown.likelihood.number_counts import NumberCountsFactory
from firecrown.metadata_functions import extract_all_harmonic_metadata_indices

# Importing the functions from utils.py
from config_builder import ConfigBuilder


def build_likelihood(cfg: ConfigBuilder) -> ConstGaussian:
    """Build the likelihood object based on the Firecrown metadatas.

    This function loads the configuration and sacc files, extracts metadata
    from the sacc, and constructs the likelihood object using the specified
    systematics and factories.

    Args:
        cfg (ConfigBuilder): Configuration object containing cosmology, array, and
                            probes settings.

    Returns:
        ConstGaussian: The likelihood object constructed from the provided
        configuration and sacc data.
    """
    # Load configuration file
    sacc_filepath = cfg.config['general']['sacc_file']

    # Load the sacc file
    sacc_data = sacc.Sacc.load_fits(sacc_filepath)

    # Extract the metadata
    all_meta = extract_all_harmonic_metadata_indices(sacc_data)

    # Load systematics values from the configuration file
    param_values = cfg.firecrown_params
    params = ParamsMap(param_values)

    # Load WeakLensing and NumberCounts factories from the configuration
    # Create the two-point from the metadata index
    if cfg.factories_config["nc_factory"] is None:
        factories = base_model_from_yaml(WeakLensingFactory,
                                         str(cfg.factories_config[
                                             "wl_factory"]))
        two_point_list = TwoPoint.from_metadata_index(
            metadata_indices=all_meta,
            wl_factory=factories,
        )

    elif cfg.factories_config["wl_factory"] is None:
        factories = base_model_from_yaml(NumberCountsFactory,
                                         str(cfg.factories_config[
                                             "nc_factory"]))
        two_point_list = TwoPoint.from_metadata_index(
            metadata_indices=all_meta,
            nc_factory=factories,
        )
    else:
        factories = [
            base_model_from_yaml(NumberCountsFactory,
                                 str(cfg.factories_config["nc_factory"])),
            base_model_from_yaml(WeakLensingFactory,
                                 str(cfg.factories_config["wl_factory"]))
        ]
        two_point_list = TwoPoint.from_metadata_index(
            metadata_indices=all_meta,
            wl_factory=factories[1],
            nc_factory=factories[0],
        )

    # Create the likelihood object and update the systematics
    lk = ConstGaussian(two_point_list)
    lk.read(sacc_data)
    lk.update(params)
    return lk
