"""Fourier Covariance Fsky Module for Covariance Matrix Computation.

Description:
This module computes the covariance matrix for Fourier modes under a
Gaussian approximation, considering the sky coverage fraction (`fsky`).
It adapts the FourrierGaussianFsky approach from TJPCOV
(fourrier_gaussian_fsky.py) to work with `firecrown` objects and
structures. This adaptation facilitates integration with more general
scenarios defined by firecrown factories.
"""

# Third-party library imports
import warnings
import numpy as np

# Firecrown imports
from firecrown.metadata_types import Galaxies, InferredGalaxyZDist
from firecrown.metadata_types import TwoPointXY, TwoPointHarmonic
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood.two_point import TwoPoint
from firecrown.utils import base_model_from_yaml


# pylint: disable=too-many-locals
# The following functions is copied from TJPCOV
def bin_cov(r, cov, r_bins):
    """Apply the binning operator.

    (Copied from TJPCOV)
    This function works on both one dimensional vectors and two dimensional
    covariance covrices.

    Args:
        r: theta or ell values at which the un-binned vector is computed.
        cov: Unbinned covariance. It also works for a vector of C_ell or xi
        r_bins: theta or ell bins to which the values should be binned.

    Returns:
        array_like: Binned covariance or vector of C_ell or xi
    """
    bin_center = 0.5 * (r_bins[1:] + r_bins[:-1])
    n_bins = len(bin_center)
    cov_int = np.zeros((n_bins, n_bins), dtype="float64")
    bin_idx = np.digitize(r, r_bins) - 1

    # this takes care of problems around bin edges
    r2 = np.sort(np.unique(np.append(r, r_bins)))
    dr = np.gradient(r2)
    r2_idx = [i for i in np.arange(len(r2)) if r2[i] in r]
    dr = dr[r2_idx]
    r_dr = r * dr
    cov_r_dr = cov * np.outer(r_dr, r_dr)

    for i in np.arange(min(bin_idx), n_bins):
        xi = bin_idx == i
        for j in np.arange(min(bin_idx), n_bins):
            xj = bin_idx == j
            norm_ij = np.sum(r_dr[xi]) * np.sum(r_dr[xj])
            if norm_ij == 0:
                continue
            cov_int[i][j] = np.sum(cov_r_dr[xi, :][:, xj]) / norm_ij
    return bin_center, cov_int


class FirecrownFourierGaussianFsky():
    """Class to adapt tjpcov Gaussian CellxCell covariance using Knox formula.

    Normalization for E-modes only. This class retrieves necessary data
    from a configuration file containing InferredZDist objects from firecrown,
    galaxy count per bin, ell edges, and `fsky`.

    Args:
        config (dict or str): Configuration in dictionary form or as a path to
            a YAML file.

    Attributes:
        tools: ModelingTools from firecrown module.
        fsky: Sky fraction.
        ell_edges: Binning edges for `ell`.
        sacc_data: Data from `sacc` file.
        factories: Systematic factories for number counts and weak lensing.
        parameters: Parameter mapping for cosmological and systematics
                    settings.
        sigma_e: Shape noise per tracer for shear.
        n_gal: Number of galaxies per tracer per square arcminute.
    """

    def __init__(self, tools, ell_edges, sacc_data, factories, parameters,
                 probes_cfg):
        """Initialize the class with the provided configuration dictionary."""
        config_ = self._get_configuration_file(tools, ell_edges, sacc_data,
                                               factories, parameters,
                                               probes_cfg)
        self.tools = self._get_config_value(config_, "tools")
        self.fsky = self._get_config_value(config_, "fsky")
        self.ell_edges = self._get_config_value(config_, "ell_edges")
        self.sacc_data = self._get_config_value(config_, "sacc_data")
        self.factories = self._get_config_value(config_, "factories")
        self.parameters = self._get_config_value(config_, "parameters")
        self.noise_data = {
            "sigma_e": config_.get("sigma_e", None),
            "n_gal": config_.get("n_gal", None),
        }

    @staticmethod
    def _get_configuration_file(tools, ell_edges, sacc_data,
                                factories, parameters, prbs_cfg):
        """Create the dictionary from the configuration file."""
        dict_ = {}
        dict_["tools"] = tools
        dict_["ell_edges"] = ell_edges
        dict_["sacc_data"] = sacc_data
        dict_["factories"] = factories
        dict_["parameters"] = parameters
        tracer_comb = sacc_data.tracers
        dict_["fsky"] = {}
        for tr1 in tracer_comb:
            for tr2 in tracer_comb:
                if "spec" in tr1 or "spec" in tr2:
                    if "spec" in tr1 and "spec" in tr2:
                        try:
                            dict_["fsky"][tr1, tr2] = prbs_cfg["probes"][
                                "desi"]["fsky"]
                        except KeyError:
                            dict_["fsky"][tr1, tr2] = prbs_cfg["probes"][
                                "4most"]["fsky"]
                    else:
                        dict_["fsky"][tr1, tr2] = prbs_cfg["probes"][
                            "overlap"]["fsky"]
                else:
                    dict_["fsky"][tr1, tr2] = prbs_cfg["probes"][
                        "lsst"]["fsky"]
        dict_["sigma_e"] = {}
        dict_["n_gal"] = {}
        for tr in sacc_data.tracers:
            # Split tracer name into base and number
            base = tr.rstrip('0123456789')  # Remove trailing numbers
            num = tr[len(base):]            # Get the number part
            tracer_num = int(num) if num else 0  # Convert to int (default 0 if no number)
            if "src" in tr:
                # Handle source tracers
                dict_["sigma_e"][tr] = prbs_cfg["probes"]["lsst"]["tracers"]["src"]["sigma_e"][tracer_num]
            # Handle all galaxy tracers (including spec)
            if "spec" in tr:
                survey = "desi"
            else:
                survey = "lsst"
            dict_["n_gal"][tr] = prbs_cfg["probes"][survey]["tracers"][base]["ngal"][tracer_num]
        return dict_

    def _get_config_value(self, config, key):
        """Get a config value and raise ValueError if None."""
        value = config.get(key)
        if value is None:
            raise ValueError(
                f"You need to set {key} for FirecrownFourierGaussianFsky"
            )
        return value

    def get_binning_info(self):
        """Retrieve binning information based on `ell` edges from config.

        (Adapted from TJPCOV).

        Returns:
            tuple:
                - ell (array): Array of `ell` values.
                - ell_edges (array): Array of `ell` bin edges.
        """
        ell_edges = self.ell_edges
        ell_min = np.min(ell_edges)
        ell_max = np.max(ell_edges)
        nbpw = (ell_max - ell_min).astype(np.int32)
        ell = np.linspace(ell_min, ell_max, nbpw+1).astype(np.int32)
        return ell, ell_edges

    def get_inferredzdist_from_sacc(self):
        """Extract dNdz and z arrays from the `sacc` file.

        Returns:
            list: List of InferredGalaxyZDist objects representing the
                  redshift distributions for each tracer.
        """
        sacc_file = self.sacc_data
        tracers = sacc_file.tracers
        inferredzdist = []
        for tracer in tracers:
            tracer_dat = sacc_file.get_tracer(tracer)
            tracer_name = tracer_dat.name
            z = tracer_dat.z
            dndz = tracer_dat.nz
            quantity = tracer_dat.quantity
            measurements = None
            if quantity == "galaxy_density":
                measurements = Galaxies.COUNTS
            elif quantity == "galaxy_shear":
                measurements = Galaxies.SHEAR_E

            infzdist_binned = InferredGalaxyZDist(
                bin_name=f"{tracer_name}",
                z=z,
                dndz=dndz,
                measurements={measurements},
            )
            inferredzdist.append(infzdist_binned)
        return inferredzdist

    def get_all_tracers_combinations(self):
        """Generate all combinations of tracer pairs.

        Returns:
            list: List of TwoPointXY objects for all tracer combinations.
        """
        inferredzdist_list = self.get_inferredzdist_from_sacc()
        all_two_point_combinations = []
        for trs1 in inferredzdist_list:
            for trs2 in inferredzdist_list:
                x_dist = trs1
                x_measurement = next(iter(x_dist.measurements))
                y_dist = trs2
                y_measurement = next(iter(y_dist.measurements))
                all_two_point_combinations.append(
                    TwoPointXY(
                        x=x_dist,
                        y=y_dist,
                        x_measurement=x_measurement,
                        y_measurement=y_measurement,
                    )
                )
        return all_two_point_combinations

    def get_cells(self):
        """Compute theoretical `C_ell` values for each tracer combination.

        Returns:
            dict: Dictionary containing computed `C_ell` for each tracer pair.
        """
        ell, _ = self.get_binning_info()
        combinations = self.get_all_tracers_combinations()
        all_metadata_cells = [
            TwoPointHarmonic(XY=xy, ells=ell) for xy in combinations
        ]

        # Define the systematic factories
        if self.factories["nc_factory"] is None:
            factories = base_model_from_yaml(wl.WeakLensingFactory,
                                             str(self.factories["wl_factory"]))
            all_cells = TwoPoint.from_metadata(
                metadata_seq=all_metadata_cells,
                wl_factory=factories,
            )
        elif self.factories["wl_factory"] is None:
            factories = base_model_from_yaml(nc.NumberCountsFactory,
                                             str(self.factories["nc_factory"]))
            all_cells = TwoPoint.from_metadata(
                metadata_seq=all_metadata_cells,
                nc_factory=factories,
            )
        else:
            factories = [
                base_model_from_yaml(nc.NumberCountsFactory,
                                     str(self.factories["nc_factory"])),
                base_model_from_yaml(wl.WeakLensingFactory,
                                     str(self.factories["wl_factory"]))
            ]
            all_cells = TwoPoint.from_metadata(
                metadata_seq=all_metadata_cells,
                nc_factory=factories[0],
                wl_factory=factories[1],
            )
        # Define the sys and cosmo parameters and update
        all_cells.update(self.parameters)

        # Create a dictionary with the C_ell
        all_cells_dict = {}
        for cell in all_cells:
            tracer_comb1 = cell.sacc_tracers.name1
            tracer_comb2 = cell.sacc_tracers.name2
            all_cells_dict[tracer_comb1, tracer_comb2] = cell
        return all_cells_dict

    def get_noise_info(self):
        """Compute shot noise signal for each tracer.

        Returns:
            dict: Dictionary with noise information for each tracer.
        """
        tracer_noise = {}
        sacc_file = self.sacc_data
        for tracer in sacc_file.tracers:
            tracer_dat = sacc_file.get_tracer(tracer)
            if tracer_dat.quantity == "galaxy_shear":
                conversion_factor = 180 / np.pi * 180 / np.pi
                arcmin2torad2 = 60*60 * conversion_factor
                if tracer in self.noise_data['sigma_e']:
                    tracer_noise[tracer] = self.noise_data['sigma_e'][tracer] ** 2 / self.noise_data['n_gal'][tracer] / arcmin2torad2
                else:
                    tracer_noise[tracer] = None

            elif tracer_dat.quantity == "galaxy_density":
                conversion_factor = 180 / np.pi * 180 / np.pi
                arcmin2torad2 = 60*60 * conversion_factor
                if tracer in self.noise_data['n_gal']:
                    tracer_noise[tracer] = 1.0 / self.noise_data['n_gal'][tracer] / arcmin2torad2
                else:
                    tracer_noise[tracer] = None
            if None in list(tracer_noise.values()):
                warnings.warn(
                    "Missing noise for some tracers in file. "
                    "You will have to pass it with the cache"
                )
                return None
        return tracer_noise

    def get_covariance_block(self, tracer_comb1, tracer_comb2):
        """Compute a single covariance matrix block for a given pair of C_ell.

        Args:
            tracer_comb1 (list): List of tracer names for the first C_ell pair.
            tracer_comb2 (list): List of tracer names for the second C_ell
            pair.

        Returns:
            array: Covariance block for the specified tracer pairs.
        """
        tools = self.tools
        ell, ell_edges = self.get_binning_info()
        all_c_ells = self.get_cells()
        all_noise = self.get_noise_info()
        if all_noise is None:
            raise ValueError("missing noise information")
        c_ell = {}
        c_ell[13] = all_c_ells[tracer_comb1[0],
                               tracer_comb2[0]].compute_theory_vector(
                                   tools)
        c_ell[24] = all_c_ells[tracer_comb1[1],
                               tracer_comb2[1]].compute_theory_vector(
                                   tools)
        c_ell[14] = all_c_ells[tracer_comb1[0],
                               tracer_comb2[1]].compute_theory_vector(
                                   tools)
        c_ell[23] = all_c_ells[tracer_comb1[1],
                               tracer_comb2[0]].compute_theory_vector(
                                   tools)
        noise = {}
        noise[13] = (
            all_noise[tracer_comb1[0]]
            if tracer_comb1[0] == tracer_comb2[0]
            else 0
        )

        noise[24] = (
            all_noise[tracer_comb1[1]]
            if tracer_comb1[1] == tracer_comb2[1]
            else 0
        )
        noise[14] = (
            all_noise[tracer_comb1[0]]
            if tracer_comb1[0] == tracer_comb2[1]
            else 0
        )
        noise[23] = (
            all_noise[tracer_comb1[1]]
            if tracer_comb1[1] == tracer_comb2[0]
            else 0
        )

        signal = {}
        signal[13] = (c_ell[13] + noise[13])
        signal[13] /= np.sqrt(
            self.fsky[(tracer_comb1[0], tracer_comb2[0])]
            * np.gradient(ell))
        signal[24] = (c_ell[24] + noise[24])
        signal[24] /= np.sqrt(
            self.fsky[(tracer_comb1[1], tracer_comb2[1])]
            * np.gradient(ell))
        signal[14] = (c_ell[14] + noise[14])
        signal[14] /= np.sqrt(
            self.fsky[(tracer_comb1[0], tracer_comb2[1])]
            * np.gradient(ell))
        signal[23] = (c_ell[23] + noise[23])
        signal[23] /= np.sqrt(
            self.fsky[(tracer_comb1[1], tracer_comb2[0])]
            * np.gradient(ell))

        cov = np.diag((signal[13] * signal[24]) + (signal[14] * signal[23]))
        norm = 2 * ell + 1
        cov /= norm

        _lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)
        assert _lb.all() == (0.5 * (ell_edges[:-1] + ell_edges[1:])).all()
        return cov

    def get_covariance_matrix(self):
        """Compute the full covariance matrix for all tracer pairs.

        Returns:
            array: Full covariance matrix for all tracer pairs.
        """
        # Initialize the covariance needed for the sacc file
        print("    Calling FirecrownFourierGaussianFsky pipeline...")
        matrix = np.zeros((len(self.sacc_data.mean), len(self.sacc_data.mean)))
        tracer_comb = self.sacc_data.get_tracer_combinations()
        for i, trs1 in enumerate(tracer_comb):
            ii = self.sacc_data.indices(tracers=trs1)
            for trs2 in tracer_comb[i:]:
                jj = self.sacc_data.indices(tracers=trs2)
                print("Computing covariance block for tracers: ",
                      trs1, trs2)
                cov_block = self.get_covariance_block(trs1, trs2)
                ii_all, jj_all = np.meshgrid(ii, jj, indexing='ij')
                matrix[ii_all, jj_all] = cov_block[:len(ii), :len(jj)]
                matrix[jj_all.T, ii_all.T] = cov_block[:len(ii), :len(jj)].T
        return matrix
