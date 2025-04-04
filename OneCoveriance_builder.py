"""Module to build the covariance matrix using the OneCovariance pipeline."""

# Standard library imports
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import configparser

# Third-party library imports
import numpy as np

# Firecrown imports
from firecrown.metadata_types import TwoPointXY, TwoPointHarmonic
from firecrown.utils import base_model_from_yaml
import firecrown.likelihood.number_counts as nc
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.two_point as tp

# Constants for file paths
DIST_FILES_DIR = './One_cov_files/dist_files/'  # Directory for distribution files
BIAS_FILES_DIR = './One_cov_files/bias_files/'  # Directory for bias files
CELLS_FILES_DIR = './One_cov_files/C_ells_files/'  # Directory for C_ell files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildCovWithOneCovariance:
    """Class to build the covariance matrix using the OneCovariance pipeline.

    This class handles the extraction of tracer information, generation of
    input files, and construction of the configuration file required by the
    OneCovariance pipeline.

    Attributes:
        cfg (Dict): Configuration dictionary containing cosmology, array, and
                    probes settings.
        tools (ModelingTools): Modeling tools object for cosmology
                               calculations.
        dists (List[InferredGalaxyZDist]): List of inferred galaxy redshift
                                           distributions.
        params (ParamsMap): Parameters map for cosmology and systematics.
        sacc_file (sacc.Sacc): Sacc file object containing tracer and
                               measurement data.
        cosmology (Dict): Cosmology parameters extracted from the
                          configuration.
        ell_min (float): Minimum ell value for the power spectrum.
        ell_max (float): Maximum ell value for the power spectrum.
        ell_bins (int): Number of ell bins.
        ell_type (str): Type of ell binning ('linear' or 'log').
        tracers (List[str]): List of tracer names extracted from the
                             configuration.
        cosmicshear_factorie (bool): Flag indicating whether cosmic shear is
                                     enabled.
        clustering_factorie (bool): Flag indicating whether clustering is
                                    enabled.
        ggl_factorie (bool): Flag indicating whether galaxy-galaxy lensing is
                             enabled.
        sky_area_dict (Dict[str, float]): Dictionary of sky areas for each
                                          survey.
        n_eff_clust_ph (float): Effective number density for photometric
                                clustering.
        n_eff_lensing (float): Effective number density for lensing.
        ellipticity_dispersion (float): Ellipticity dispersion for lensing.
        n_eff_clust_spec (float): Effective number density for spectroscopic
                                  clustering.
        distributions_lens_files (List[str]): List of lens distribution file
                                              paths.
        distributions_spec_files (List[str]): List of spectroscopic
                                              distribution file paths.
        distributions_src_files (List[str]): List of source distribution file
                                             paths.
        bias_lens_files (List[str]): List of lens bias file paths.
        bias_spec_files (List[str]): List of spectroscopic bias file paths.
        src_config (Dict): Source configuration for intrinsic alignment
                           parameters.
        c_ells_files (Dict): Dictionary of C_ell file paths.
    """

    def __init__(self, cfg, tools, dists, params, sacc_file):
        """Initialize the BuildCovWithOneCovariance object.

        Args:
            cfg (Dict): Configuration dictionary containing cosmology, array,
                        and probes settings.
            tools (ModelingTools): Modeling tools object for cosmology
                                   calculations.
            dists (List[InferredGalaxyZDist]): List of inferred galaxy
                                               redshift distributions.
            params (ParamsMap): Parameters map for cosmology and systematics.
            sacc_file (sacc.Sacc): Sacc file object containing tracer and
                                   measurement data.
        """
        # Assign attributes
        self.cfg = cfg
        self.tools = tools
        self.dists = dists
        self.params = params
        self.sacc_file = sacc_file

        # Extract cosmology and ell binning settings
        self.cosmology = self.cfg.cosmo_config["cosmology"]
        self.ell_min = self.cfg.array_config["ell_array"]["ell_start"]
        self.ell_max = self.cfg.array_config["ell_array"]["ell_stop"]
        self.ell_bins = self.cfg.array_config["ell_array"]["ell_bins"]
        self.ell_type = self.cfg.array_config["ell_array"]["type"]

        # Extract tracers and factory flags
        self.tracers = self._extract_tracers()
        self.cosmicshear_factorie = (
            self.cfg.factories_config["wl_factory"] is not None
        )
        self.clustering_factorie = (
            self.cfg.factories_config["nc_factory"] is not None
        )
        self.ggl_factorie = (
            self.clustering_factorie and self.cosmicshear_factorie
        )

        # Extract sky areas for each survey
        self.sky_area_dict = {
            survey: self.cfg.probes_config["probes"][survey]["sky_area"]
            for survey in self.cfg.probes_config["probes"]
        }

        # Extract effective number densities and ellipticity dispersion
        (
            self.n_eff_clust_ph,
            self.n_eff_lensing,
            self.ellipticity_dispersion,
            self.n_eff_clust_spec,
        ) = self._extract_effective_numbers()

        # Extract distribution and bias files
        distribution_files_dict = self._extract_distribution_files_dict()
        self.distributions_lens_files = distribution_files_dict.get('lens', [])
        self.distributions_spec_files = distribution_files_dict.get('spec', [])
        self.distributions_src_files = distribution_files_dict.get('src', [])

        if "lens" or "spec" in self.tracers:
            bias_files_dict = self._extract_bias_files_dict()
            self.bias_lens_files = bias_files_dict.get('lens', [])
            self.bias_spec_files = bias_files_dict.get('spec', [])

        # Source configuration for intrinsic alignment
        if 'src' in self.tracers:
            self.src_config = {
                'A_IA': self.cfg.probes_config["probes"]["lsst"]["tracers"]["src"][
                    "ia_bias"]["fid_value"],
                'eta_IA': self.cfg.probes_config["probes"]["lsst"]["tracers"][
                    "src"]["alphaz"]["fid_value"],
                'z_pivot_IA': self.cfg.probes_config["probes"]["lsst"]["tracers"][
                    "src"]["z_piv"]["fid_value"]
            }

        # C_ell file paths
        self.c_ells_files = {
            'Cell_directory': CELLS_FILES_DIR,
            'Cgg_file': 'Cell_gg.ascii',
            'Cgm_file': 'Cell_gkappa.ascii',
            'Cmm_file': 'Cell_kappakappa.ascii'
        }

    def _extract_tracers(self) -> List[str]:
        """Extract tracer names from the configuration.

        Returns:
            List[str]: List of tracer names.
        """
        tracers = []
        for survey in self.cfg.probes_config["probes"]:
            if survey == 'overlap':
                continue
            for tracer in self.cfg.probes_config["probes"][survey]["tracers"]:
                tracers.append(tracer)
        return tracers

    def _extract_effective_numbers(self) -> Tuple[float, float, float, float]:
        """Extract effective number densities and ellipticity dispersion.

        Returns:
            Tuple[float, float, float, float]: A tuple containing:
                - n_eff_clust_ph: Effective number density for photometric
                                  clustering.
                - n_eff_lensing: Effective number density for lensing.
                - ellipticity_dispersion: Ellipticity dispersion for lensing.
                - n_eff_clust_spec: Effective number density for spectroscopic
                                    clustering.
        """
        n_eff_clust_ph, n_eff_lensing, ellipticity_dispersion, \
            n_eff_clust_spec = None, None, None, None
        for survey in self.cfg.probes_config["probes"]:
            if survey == 'overlap':
                continue
            for tracer in self.cfg.probes_config["probes"][survey]["tracers"]:
                if tracer.startswith('lens'):
                    n_eff_clust_ph = self.cfg.probes_config["probes"][survey][
                        "tracers"][tracer]["ngal"]
                elif tracer.startswith('src'):
                    n_eff_lensing = self.cfg.probes_config["probes"][survey][
                        "tracers"][tracer]["ngal"]
                    ellipticity_dispersion = self.cfg.probes_config["probes"][
                        survey]["tracers"][tracer]["sigma_e"]
                elif tracer.startswith('spec'):
                    n_eff_clust_spec = self.cfg.probes_config["probes"][survey]["tracers"][tracer]["ngal"]
        return n_eff_clust_ph, n_eff_lensing, ellipticity_dispersion, n_eff_clust_spec

    def _extract_bias_values(self) -> Tuple[float, float]:
        """Extract bias values for lens and spec tracers.

        Returns:
            Tuple[float, float]: A tuple containing bias values for lens and spec tracers.
        """
        bias_values = {}
        for survey in self.cfg.probes_config["probes"]:
            if survey == 'overlap':
                continue
            for tracer_name, tracer_config in self.cfg.probes_config["probes"][survey]["tracers"].items():
                if "bias" in tracer_config:
                    bias_list = tracer_config["bias"]["fid_value"]
                    for i, bias_value in enumerate(bias_list):
                        bias_values[f"{tracer_name}{i}"] = bias_value
        return bias_values

    def _extract_distribution_files_dict(self) -> Dict[str, List[str]]:
        """Extract distribution dictionary for tracers.

        This method creates a dictionary mapping tracer types (e.g., 'lens', 'spec', 'src')
        to their corresponding distribution files.

        Returns:
            Dict[str, List[str]]: Dictionary mapping tracer types to their distribution files.
        """
        dist_dict = {}
        for survey in self.cfg.probes_config["probes"]:
            if survey == 'overlap':
                continue
            for tracer in self.cfg.probes_config["probes"][survey]["tracers"]:
                dist_dict[tracer] = [f'distr_{tracer}{i}.ascii' for i in range(
                    len(self.cfg.probes_config["probes"][survey]["tracers"][
                        tracer]["ngal"]))]
        return dist_dict

    def _extract_bias_files_dict(self) -> Dict[str, List[str]]:
        """Extract bias dictionary for tracers.

        This method creates a dictionary mapping tracer types (e.g., 'lens', 'spec')
        to their corresponding bias files.

        Returns:
            Dict[str, List[str]]: Dictionary mapping tracer types to their bias files.
        """
        bias_dict = {}
        for survey in self.cfg.probes_config["probes"]:
            if survey == 'overlap':
                continue
            for tracer in self.cfg.probes_config["probes"][survey]["tracers"]:
                if "bias" in self.cfg.probes_config["probes"][survey][
                   "tracers"][tracer]:
                    bias_dict[tracer] = [
                        f'{BIAS_FILES_DIR}bias_{tracer}{i}.ascii' for i in
                        range(len(self.cfg.probes_config["probes"][survey]
                                  ["tracers"][tracer]["bias"]["fid_value"]))]
        return bias_dict

    def _write_redshift_distributions(self, distribution):
        """Write redshift distributions to files.

        This method writes the redshift distributions for each tracer to files in the
        specified directory.

        Args:
            distribution: List of redshift distribution objects.
        """
        for dist in distribution:
            file_path = f'{DIST_FILES_DIR}distr_{dist.bin_name}.ascii'
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write("# z dndz\n")
                    for z_val, dndz_val in zip(dist.z, dist.dndz):
                        file.write(f"{z_val:.18e} {dndz_val:.18e}\n")
                logger.info("Successfully wrote redshift distribution to %s", file_path)
            except IOError as e:
                logger.error("Failed to write redshift distribution to %s: %s", file_path, e)

    def _write_bias_files(self, distribution):
        """Write bias files for tracers.

        This method writes the bias values for each tracer to files in the specified
        directory.

        Args:
            distribution: List of distribution objects containing bias values.
        """
        bias_dict = self._extract_bias_values()
        for dist in distribution:
            if dist.bin_name.startswith('spec') or dist.bin_name.startswith('lens'):
                file_path = f'{BIAS_FILES_DIR}bias_{dist.bin_name}.ascii'
                try:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write("# z bias\n")
                        for z_val in distribution[0].z:
                            file.write(f"{z_val:.18e} {bias_dict[dist.bin_name]}\n")
                    logger.info("Successfully wrote bias file to %s", file_path)
                except IOError as e:
                    logger.error("Failed to write bias file to %s: %s", file_path, e)

    def _aux_write_c_ell_files(self, distribution, tools, params):
        """Generate and write C_ell files.

        This method creates all possible two-point combinations, computes the C_ell values,
        and writes them to files.

        Args:
            distribution: List of distribution objects.
            tools: Modeling tools for cosmology calculations.
            params: Parameters map for cosmology and systematics.

        Returns:
            Dict: Dictionary containing computed C_ell values.
        """
        all_two_point_xy = self._create_two_point_combinations(distribution)
        ells = self._generate_ells()
        all_two_point_cells = [TwoPointHarmonic(XY=xy, ells=ells) for xy in all_two_point_xy]
        two_point_objects = self._create_two_point_objects(all_two_point_cells)
        two_point_objects.update(params)
        return self._compute_cell_dictionary(two_point_objects, tools)

    def _create_two_point_combinations(self, distribution):
        """Create all possible two-point combinations.

        This method generates all possible pairs of tracers and their corresponding
        two-point combinations.

        Args:
            distribution: List of distribution objects.

        Returns:
            List[TwoPointXY]: List of two-point combinations.
        """
        all_two_point_xy = defaultdict()
        for sample1 in distribution:
            for sample2 in distribution:
                x_dist = sample1
                x_measurement = next(iter(x_dist.measurements))
                y_dist = sample2
                y_measurement = next(iter(y_dist.measurements))
                key = (sample1.bin_name, sample2.bin_name)
                all_two_point_xy[key] = TwoPointXY(
                    x=x_dist,
                    y=y_dist,
                    x_measurement=x_measurement,
                    y_measurement=y_measurement,
                )
        keys_to_remove = [key for key in all_two_point_xy
                          if key[0].startswith("src") and (key[1].startswith("lens") or key[1].startswith("spec")) or
                          (key[0].startswith("lens") and key[1].startswith("spec"))]
        for key in keys_to_remove:
            all_two_point_xy.pop(key)
        return [all_two_point_xy[key] for key in all_two_point_xy]

    def _generate_ells(self):
        """Generate ell values based on configuration.

        This method generates ell values for the power spectrum based on the
        configuration settings (linear or log binning).

        Returns:
            np.ndarray: Array of ell values.
        """
        if self.cfg.array_config["ell_array"]["type"] == "linear":
            return np.unique(np.linspace(
                self.cfg.array_config["ell_array"]["ell_start"],
                self.cfg.array_config["ell_array"]["ell_stop"],
                self.cfg.array_config["ell_array"]["ell_bins"],
                endpoint=True,
            ).astype(np.int32))
        elif self.cfg.array_config["ell_array"]["type"] == "log":
            return np.unique(np.geomspace(
                self.cfg.array_config["ell_array"]["ell_start"],
                self.cfg.array_config["ell_array"]["ell_stop"],
                self.cfg.array_config["ell_array"]["ell_bins"],
                endpoint=True,
            ).astype(np.int32))

    def _create_two_point_objects(self, all_two_point_cells):
        """Create two-point objects based on factories.

        This method creates two-point objects (e.g., clustering, weak lensing) based on
        the enabled factories.

        Args:
            all_two_point_cells: List of two-point combinations.

        Returns:
            TwoPoint: Two-point objects for the enabled factories.
        """
        if self.clustering_factorie and not self.cosmicshear_factorie:
            return tp.TwoPoint.from_metadata(
                metadata_seq=all_two_point_cells,
                nc_factory=base_model_from_yaml(nc.NumberCountsFactory, str(self.cfg.factories_config["nc_factory"])),
            )
        elif not self.clustering_factorie and self.cosmicshear_factorie:
            return tp.TwoPoint.from_metadata(
                metadata_seq=all_two_point_cells,
                wl_factory=base_model_from_yaml(wl.WeakLensingFactory, str(self.cfg.factories_config["wl_factory"])),
            )
        elif self.clustering_factorie and self.cosmicshear_factorie:
            return tp.TwoPoint.from_metadata(
                metadata_seq=all_two_point_cells,
                nc_factory=base_model_from_yaml(nc.NumberCountsFactory, str(self.cfg.factories_config["nc_factory"])),
                wl_factory=base_model_from_yaml(wl.WeakLensingFactory, str(self.cfg.factories_config["wl_factory"])),
            )

    def _compute_cell_dictionary(self, two_point_objects, tools):
        """Compute and store C_ell values in a dictionary.

        This method computes the C_ell values for each two-point combination and stores
        them in a dictionary.

        Args:
            two_point_objects: List of two-point objects.
            tools: Modeling tools for cosmology calculations.

        Returns:
            Dict: Dictionary containing computed C_ell values.
        """
        dict_aux = {}
        for i, ells_ in enumerate(two_point_objects[0].ells):
            if ells_ not in dict_aux:
                dict_aux[ells_] = {}
            for twp in two_point_objects:
                tracer1 = twp.sacc_tracers.name1
                tracer2 = twp.sacc_tracers.name2
                trc1_name = tracer1.split('_')[0] if tracer1.startswith('spec') else tracer1[0:-1]
                trc2_name = tracer2.split('_')[0] if tracer2.startswith('spec') else tracer2[0:-1]
                if twp.sacc_data_type not in dict_aux[ells_]:
                    dict_aux[ells_][twp.sacc_data_type] = {}
                tracer_key = (tracer1[-1], tracer2[-1])
                if tracer_key not in dict_aux[ells_][twp.sacc_data_type]:
                    dict_aux[ells_][twp.sacc_data_type][tracer_key] = {}
                dict_aux[ells_][twp.sacc_data_type][tracer_key][(trc1_name, trc2_name)] = twp.compute_theory_vector(tools)[i]
        return dict_aux

    def _write_c_ell_files(self, dict_aux):
        """Write C_ell values to output files.

        This method writes the computed C_ell values to output files in the specified
        directory. The files include C_ell values for galaxy clustering (gg),
        galaxy-galaxy lensing (gk), and cosmic shear (kk).

        Args:
            dict_aux (Dict): Dictionary containing computed C_ell values.
        """
        try:
            with open(f'{CELLS_FILES_DIR}/Cell_gg.ascii', 'w', encoding='utf-8') as file_gg, \
                 open(f'{CELLS_FILES_DIR}/Cell_gkappa.ascii', 'w', encoding='utf-8') as file_gk, \
                 open(f'{CELLS_FILES_DIR}/Cell_kappakappa.ascii', 'w', encoding='utf-8') as file_kk:

                file_gg.write("#ell\ttomo_i\ttomo_j\tCgsgs_ij\tCgsgp_ij\tCgpgp_ij\n")
                file_gk.write("#ell\ttomo_i\ttomo_j\tCgsm_ij\tCgpm_ij\n")
                file_kk.write("#ell\ttomo_i\ttomo_j\tCmm_ij\n")

                for ells_ in sorted(dict_aux.keys()):
                    self._write_cell_values(dict_aux[ells_], ells_, file_gg, file_gk, file_kk)
            logger.info("Successfully wrote C_ell files.")
        except IOError as e:
            logger.error("Failed to write C_ell files: %s", e)

    def _write_cell_values(self, cell_data, ells_, file_gg, file_gk, file_kk):
        """Write C_ell values for a specific ell.

        This method writes the C_ell values for a specific ell to the corresponding
        output files.

        Args:
            cell_data (Dict): Dictionary containing C_ell values for the current ell.
            ells_ (float): Current ell value.
            file_gg: File object for galaxy clustering C_ell values.
            file_gk: File object for galaxy-galaxy lensing C_ell values.
            file_kk: File object for cosmic shear C_ell values.
        """
        if 'galaxy_density_cl' in cell_data:
            self._write_galaxy_density_cl(cell_data['galaxy_density_cl'], ells_, file_gg)
        if 'galaxy_shearDensity_cl_e' in cell_data:
            self._write_galaxy_shear_density_cl(cell_data['galaxy_shearDensity_cl_e'], ells_, file_gk)
        if 'galaxy_shear_cl_ee' in cell_data:
            self._write_galaxy_shear_cl(cell_data['galaxy_shear_cl_ee'], ells_, file_kk)

    def _write_galaxy_density_cl(self, data, ells_, file_gg):
        """Write galaxy density C_ell values.

        This method writes the galaxy density C_ell values for the current ell to the
        galaxy clustering file.

        Args:
            data (Dict): Dictionary containing galaxy density C_ell values.
            ells_ (float): Current ell value.
            file_gg: File object for galaxy clustering C_ell values.
        """
        for (tomo_i, tomo_j), values in sorted(data.items()):
            file_gg.write(f"{ells_}\t{tomo_i}\t{tomo_j}")
            # FIXME: Not the right way to do this
            for sample_pair in [('lens', 'lens')]:
                value = values.get(sample_pair, 'NaN')
                file_gg.write(f"\t{value if value == 'NaN' else f'{value:.10e}'}")
            file_gg.write("\n")

    def _write_galaxy_shear_density_cl(self, data, ells_, file_gk):
        """Write galaxy shear density C_ell values.

        This method writes the galaxy shear density C_ell values for the current ell to the
        galaxy-galaxy lensing file.

        Args:
            data (Dict): Dictionary containing galaxy shear density C_ell values.
            ells_ (float): Current ell value.
            file_gk: File object for galaxy-galaxy lensing C_ell values.
        """
        for (tomo_i, tomo_j), values in sorted(data.items()):
            file_gk.write(f"{ells_}\t{tomo_i}\t{tomo_j}")
            # FIXME: Not the right way to do this
            for sample_pair in [('lens', 'src')]:
                value = values.get(sample_pair, 'NaN')
                file_gk.write(f"\t{value if value == 'NaN' else f'{value:.10e}'}")
            file_gk.write("\n")

    def _write_galaxy_shear_cl(self, data, ells_, file_kk):
        """Write galaxy shear C_ell values.

        This method writes the galaxy shear C_ell values for the current ell to the
        cosmic shear file.

        Args:
            data (Dict): Dictionary containing galaxy shear C_ell values.
            ells_ (float): Current ell value.
            file_kk: File object for cosmic shear C_ell values.
        """
        for (tomo_i, tomo_j), values in sorted(data.items()):
            file_kk.write(f"{ells_}\t{tomo_i}\t{tomo_j}")
            # FIXME: Not the right way to do this
            value = values.get(('src', 'src'), 'NaN')
            file_kk.write(f"\t{value if value == 'NaN' else f'{value:.10e}'}\n")

    def aux_necessary_files(self):
        """Write all necessary files for the analysis.

        This method writes the redshift distributions, bias files, and C_ell files
        required for the OneCovariance pipeline.
        """
        print("    Writing necessary files...")
        self._write_redshift_distributions(self.dists)
        self._write_bias_files(self.dists)
        dict_aux = self._aux_write_c_ell_files(self.dists, self.tools, self.params)
        self._write_c_ell_files(dict_aux)

    def build_ini_file(self):
        """Build the configuration file for the OneCovariance pipeline.

        This method constructs the `config.ini` file required by the OneCovariance
        pipeline. The file includes settings for covariance terms, observables,
        output settings, survey specifications, redshift distributions, cosmology,
        bias, intrinsic alignment, and other parameters.
        """
        config = configparser.ConfigParser()

        # Covariance terms
        config["covariance terms"] = {
            'gauss': 'True',
            'split_gauss': 'True',
            'nongauss': 'False',
            'ssc': 'False'
        }

        # Observables
        config['observables'] = {
            'cosmic_shear': self.cosmicshear_factorie,
            'est_shear': 'C_ell',
            'ggl': self.ggl_factorie,
            'est_ggl': 'C_ell',
            'clustering': self.clustering_factorie,
            'est_clust': 'C_ell',
            'cstellar_mf': 'False',
            'cross_terms': 'True',
            'unbiased_clustering': 'False'
        }

        # Output settings
        config['output settings'] = {
            'directory': './One_cov_files/output/',
            'file': 'covariance_list.dat, covariance_matrix.mat',
            'style': 'list, matrix',
            'list_style_spatial_first': 'True',
            'corrmatrix_plot': 'correlation_coefficient.pdf',
            'save_configs': 'save_configs.ini',
            'save_Cells': 'True',
            'save_trispectra': 'False',
            'save_alms': 'True',
            'use_tex': 'False'
        }

        # CovELLspace settings
        covellspace_settings = {
            'delta_z': '0.08',
            'tri_delta_z': '0.5',
            'integration_steps': '500',
            'nz_interpolation_polynom_order': '1',
            'ell_min': f'{self.ell_min}',
            'ell_max': f'{self.ell_max}',
            'ell_bins': f'{self.ell_bins}',
            'ell_type': f'{self.ell_type}',
        }
        if self.clustering_factorie:
            if 'spec' in self.tracers:
                covellspace_settings.update({
                    'n_spec': f'{len(self.distributions_spec_files)}'
                })
                covellspace_settings.update({
                    'ell_spec_min': f'{self.ell_min}',
                    'ell_spec_max': f'{self.ell_max}',
                    'ell_spec_bins': f'{self.ell_bins}',
                    'ell_spec_type': f'{self.ell_type}',
                })
                covellspace_settings.update({
                    'ell_photo_min': f'{self.ell_min}',
                    'ell_photo_max': f'{self.ell_max}',
                    'ell_photo_bins': f'{self.ell_bins}',
                    'ell_photo_type': f'{self.ell_type}',
                })
            covellspace_settings.update({
                'ell_min_clustering': f'{self.ell_min}',
                'ell_max_clustering': f'{self.ell_max}',
                'ell_bins_clustering': f'{self.ell_bins}',
                'ell_type_clustering': f'{self.ell_type}',
            })

        if self.cosmicshear_factorie:
            covellspace_settings.update({
                'ell_min_lensing': f'{self.ell_min}',
                'ell_max_lensing': f'{self.ell_max}',
                'ell_bins_lensing': f'{self.ell_bins}',
                'ell_type_lensing': f'{self.ell_type}',
            })
        covellspace_settings.update({
                'pixelised_cell': 'False',
                'limber': 'True',
        })
        config['covELLspace settings'] = covellspace_settings

        # Survey specs
        # FIXME: Check the consistency of this
        survey_specs = {}
        if self.clustering_factorie:
            try:
                survey_specs['survey_area_clust_in_deg2'] = \
                    ', '.join(map(str, [self.sky_area_dict["desi"], self.sky_area_dict["lsst"]]))
                survey_specs['n_eff_clust'] = \
                    ', '.join(map(str, self.n_eff_clust_spec + self.n_eff_clust_ph))
            except KeyError:
                survey_specs['survey_area_clust_in_deg2'] = \
                    f'{self.sky_area_dict["lsst"]}'
                survey_specs['n_eff_clust'] = \
                    ', '.join(map(str, self.n_eff_clust_ph))

        if self.cosmicshear_factorie:
            survey_specs['survey_area_lensing_in_deg2'] = \
                f'{self.sky_area_dict["lsst"]}'
            survey_specs['ellipticity_dispersion'] = \
                ', '.join(map(str, self.ellipticity_dispersion))
            survey_specs['n_eff_lensing'] = \
                ', '.join(map(str, self.n_eff_lensing))
        if self.clustering_factorie and self.cosmicshear_factorie:
            try:
                survey_specs['survey_area_ggl_in_deg2'] = \
                    ', '.join(map(str, [self.sky_area_dict["overlap"], self.sky_area_dict["lsst"]]))
            except KeyError:
                survey_specs['survey_area_ggl_in_deg2'] = \
                    f'{self.sky_area_dict["lsst"]}'
        config['survey specs'] = survey_specs

        # Redshift
        redshift_config = {
            'z_directory': './One_cov_files/dist_files',
        }

        if self.clustering_factorie:
            if self.distributions_spec_files == []:
                if self.distributions_lens_files == []:
                    print("No redshift distributions found.")
                    exit(-1)
                else:
                    redshift_config['zclust_file'] = ', '.join(self.distributions_lens_files)
            else:
                if self.distributions_lens_files == []:
                    redshift_config['zclust_file'] = ', '.join(self.distributions_spec_files)
                else:
                    redshift_config['zclust_file'] = ', '.join(self.distributions_spec_files + self.distributions_lens_files)
            redshift_config['value_loc_in_clustbin'] = 'mid'

        if self.cosmicshear_factorie:
            redshift_config['zlens_file'] = ', '.join(self.distributions_src_files)
            redshift_config['value_loc_in_lensbin'] = 'mid'

        config['redshift'] = redshift_config

        # Cosmo
        config['cosmo'] = {
            'sigma8': f'{self.cosmology["sigma8"]["fid_value"]}',
            'h': f'{self.cosmology["h"]["fid_value"]}',
            'omega_m': f'{self.cosmology["Omega_c"]["fid_value"] + self.cosmology["Omega_b"]["fid_value"] + self.cosmology["m_nu"]["fid_value"] / (93.14 * self.cosmology["h"]["fid_value"] ** 2)}',
            'omega_b': f'{self.cosmology["Omega_b"]["fid_value"]}',
            'w0': f'{self.cosmology["w0"]["fid_value"]}',
            'wa': f'{self.cosmology["wa"]["fid_value"]}',
            'ns': f'{self.cosmology["n_s"]["fid_value"]}',
            'neff': f'{self.cosmology["Neff"]["fid_value"]}',
            'm_nu': f'{self.cosmology["m_nu"]["fid_value"]}'
        }

        # Bias
        if self.clustering_factorie:
            config['bias'] = {}
            if self.bias_spec_files == []:
                if self.bias_lens_files == []:
                    print("No bias found.")
                    exit(-1)
                else:
                    config['bias']['bias_files'] = ', '.join(self.bias_lens_files)
            else:
                if self.bias_lens_files == []:
                    config['bias']['bias_files'] = ', '.join(self.bias_spec_files)
                else:
                    config['bias']['bias_files'] = ', '.join(self.bias_spec_files + self.bias_lens_files)

        # IA
        if self.cosmicshear_factorie:
            config['IA'] = {
                'A_IA': f'{self.src_config["A_IA"]}',
                'eta_IA': f'{self.src_config["eta_IA"]}',
                'z_pivot_IA': f'{self.src_config["z_pivot_IA"]}'
            }

        # HOD
        config['hod'] = {
            'model_mor_cen': 'double_powerlaw',
            'model_mor_sat': 'double_powerlaw',
            'dpow_logm0_cen': '10.51',
            'dpow_logm1_cen': '11.38',
            'dpow_a_cen': '7.096',
            'dpow_b_cen': '0.2',
            'dpow_norm_cen': '1.0',
            'dpow_norm_sat': '0.56',
            'model_scatter_cen': 'lognormal',
            'model_scatter_sat': 'modschechter',
            'logn_sigma_c_cen': '0.35',
            'modsch_logmref_sat': '13.0',
            'modsch_alpha_s_sat': '-0.858',
            'modsch_b_sat': '-0.024, 1.149'
        }

        # Halomodel evaluation
        config['halomodel evaluation'] = {
            'm_bins': '900',
            'log10m_min': '6',
            'log10m_max': '18',
            'hmf_model': 'Tinker10',
            'mdef_model': 'SOMean',
            'mdef_params': 'overdensity, 200',
            'disable_mass_conversion': 'True',
            'delta_c': '1.686',
            'transfer_model': 'CAMB',
            'small_k_damping_for1h': 'damped'
        }

        # Powspec evaluation
        config['powspec evaluation'] = {
            'non_linear_model': 'mead2020',
            'HMCode_logT_AGN': '7.3',
            'log10k_bins': '200',
            'log10k_min': '-3.49',
            'log10k_max': '2.15'
        }

        # Tabulated inputs files
        if self.clustering_factorie is True and self.cosmicshear_factorie is False:
            config['tabulated inputs files'] = {
                'Cell_directory': './One_cov_files/C_ells_files',
                'Cgg_file': f'{self.c_ells_files["Cgg_file"]}'
            }
        elif self.clustering_factorie is False and self.cosmicshear_factorie is True:
            config['tabulated inputs files'] = {
                'Cell_directory': './One_cov_files/C_ells_files',
                'Cmm_file': f'{self.c_ells_files["Cmm_file"]}'
            }
        elif self.clustering_factorie is True and self.cosmicshear_factorie is True:
            config['tabulated inputs files'] = {
                'Cell_directory': f'{self.c_ells_files["Cell_directory"]}',
                'Cgg_file': f'{self.c_ells_files["Cgg_file"]}',
                'Cgm_file': f'{self.c_ells_files["Cgm_file"]}',
                'Cmm_file': f'{self.c_ells_files["Cmm_file"]}'
            }

        # Misc
        config['misc'] = {
            'num_cores': '1'
        }

        # Write the configuration to a file
        with open('One_cov_files/config.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    def build_cov_matrix(self):
        """Execute the OneCovariance pipeline to compute the covariance matrix.

        This method calls the OneCovariance pipeline using the generated `config.ini`
        file to compute the covariance matrix.
        """
        print("    Calling OneCovariance pipeline...")
        os.system("python ../../covariance.py One_cov_files/config.ini")

    def load_cov_matrix(self):
        """Load and organize the resulting covariance matrix.

        This method loads the computed covariance matrix from the output file and
        organizes it into a format suitable for further analysis.

        Returns:
            np.ndarray: Covariance matrix.
        """
        full_data = np.loadtxt('One_cov_files/output/covariance_matrix.mat', delimiter=' ', skiprows=1)

        n_tomo_dictionary = dict()
        for survey in self.cfg.probes_config["probes"]:
            if survey == 'overlap':
                continue
            for tracer in self.cfg.probes_config["probes"][survey]["tracers"]:
                for syst_name, values in self.cfg.probes_config["probes"][survey]["tracers"][tracer].items():
                    if syst_name == 'ngal':
                        n_tomo_dictionary[tracer] = len(values)

        index_matrix = {}
        current_index = 0
        # Populate clustering pairs
        if self.clustering_factorie:
            for i_tomo in range(n_tomo_dictionary['lens']):
                for j_tomo in range(i_tomo, n_tomo_dictionary['lens']):
                    index_matrix[(f'lens{i_tomo}', f'lens{j_tomo}')] = list(range(current_index, current_index + self.ell_bins))
                    current_index += self.ell_bins
        # Populate ggl pairs
        if self.clustering_factorie and self.cosmicshear_factorie:
            for i_tomo in range(n_tomo_dictionary['lens']):
                for j_tomo in range(n_tomo_dictionary['src']):
                    index_matrix[(f'lens{i_tomo}', f'src{j_tomo}')] = list(range(current_index, current_index + self.ell_bins))
                    current_index += self.ell_bins

        # Populate weaklensing pairs
        if self.cosmicshear_factorie:
            for i_tomo in range(n_tomo_dictionary['src']):
                for j_tomo in range(i_tomo, n_tomo_dictionary['src']):
                    index_matrix[(f'src{i_tomo}', f'src{j_tomo}')] = list(range(current_index, current_index + self.ell_bins))
                    current_index += self.ell_bins

        # Initialize a dictionary to store the covariance blocks
        covariance_blocks = {}
        # Extract specific blocks using the index_matrix
        for key1, indices1 in index_matrix.items():
            for key2, indices2 in index_matrix.items():
                covariance_blocks[(key1, key2)] = full_data[np.ix_(indices1, indices2)]

        # Initialize the covariance needed for the sacc file
        matrix = np.zeros((len(self.sacc_file.mean), len(self.sacc_file.mean)))
        tracer_comb = self.sacc_file.get_tracer_combinations()
        for i, trs1 in enumerate(tracer_comb):
            ii = self.sacc_file.indices(tracers=trs1)
            for trs2 in tracer_comb[i:]:
                jj = self.sacc_file.indices(tracers=trs2)
                cov_block = covariance_blocks.get((trs1, trs2))
                ii_all, jj_all = np.meshgrid(ii, jj, indexing='ij')
                matrix[ii_all, jj_all] = cov_block[:len(ii), :len(jj)]
                matrix[jj_all.T, ii_all.T] = cov_block[:len(ii), :len(jj)].T
        return matrix
