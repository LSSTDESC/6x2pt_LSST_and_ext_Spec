"""Likelihood builder module for construct likelihood object from config."""

# Sacc import
import sacc

# Firecrown imports
from firecrown.parameters import ParamsMap
from firecrown.utils import base_model_from_yaml
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.two_point import TwoPoint
import firecrown.likelihood.two_point as tp
from firecrown.likelihood.weak_lensing import WeakLensingFactory
from firecrown.likelihood.number_counts import NumberCountsFactory
from firecrown.metadata_functions import extract_all_harmonic_metadata_indices
from utils import build_modeling_tools
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
            tp_factory=tp.TwoPointFactory(
                    correlation_space= tp.TwoPointCorrelationSpace.HARMONIC,
                    weak_lensing_factories=[factories],
                )
            )

    elif cfg.factories_config["wl_factory"] is None:
        factories = base_model_from_yaml(NumberCountsFactory,
                                         str(cfg.factories_config[
                                             "nc_factory"]))
        two_point_list = TwoPoint.from_metadata_index(
            metadata_indices=all_meta,
            tp_factory=tp.TwoPointFactory(
                    correlation_space= tp.TwoPointCorrelationSpace.HARMONIC,
                    number_counts_factories=[factories],
                )
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
            tp_factory=tp.TwoPointFactory(
                    correlation_space= tp.TwoPointCorrelationSpace.HARMONIC,
                    number_counts_factories=[factories[0]],
                    weak_lensing_factories=[factories[1]],
                )
            )
    # Create the likelihood object and update the systematics
    lk = ConstGaussian(two_point_list)
    lk.read(sacc_data)
    lk.update(params)
    tools = build_modeling_tools(cfg.cosmo_config)
    tools.ccl_cosmo.compute_nonlin_power()
    print(lk.compute_chisq(tools))
    return lk
