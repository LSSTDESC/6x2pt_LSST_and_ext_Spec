"""Module to build all the configuration files needed to run the pipeline."""
# Third-party library imports
from typing import Any, Dict
import re
import yaml


def load_yaml_file(yaml_file: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        Dict: Parsed YAML data as a dictionary. Returns an empty dictionary if
              the file is empty.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(yaml_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            return data if data is not None else {}
    except FileNotFoundError as error:
        raise FileNotFoundError(f"YAML file not found: {yaml_file}") from error
    except yaml.YAMLError as error:
        raise yaml.YAMLError(f"Error parsing YAML file {yaml_file}: {error}") from error


class ConfigBuilder:
    """
    A class to construct the configuration files required for the pipeline.

    This class stores all the configurations needed for the pipeline, including
    cosmology, probes, array choices, and probe combinations. It also builds
    configurations for Firecrown parameters, Firecrown factories, and
    Augur's Analyse class.

    Key Functionalities:
    - Processes systematic effects (e.g.,number counts bias, linear alignment).
    - Constructs factories for systematic effects (e.g., photo-z shifts,
      magnification bias, etc).
    - Validates and extracts tracer configurations.
    - Builds a Firecrown-compatible configuration dictionary.
    - Generates Fisher matrix configurations for Augur.

    Note:
    The .ini configuration for the covariance part is stored in a separate
    module because a different library is used to read .ini files.

    Attributes:
        config (Dict[str, Any]): Initial configuration dictionary.
        cosmo_config (Dict[str, Any]): Cosmology configuration.
        probes_config (Dict[str, Any]): Probes configuration.
        probes_comb_config (Dict[str, Any]): Probe combinations configuration.
        firecrown_params (Dict[str, Any]): Firecrown parameters.
        array_config (Dict[str, Any]): Array choices configuration.
        factories_config (Dict[str, Any]): Factories configuration.
        fisher_config (Dict[str, Any]): Fisher matrix configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ConfigBuilder with the initial configuration.

        Args:
            config (Dict[str, Any]): Initial configuration dictionary.
        """
        self.config = config
        self.cosmo_config = None
        self.probes_config = None
        self.probes_comb_config = None
        self.firecrown_params = None
        self.array_config = None
        self.factories_config = None
        self.fisher_config = None
        self.prior_config = None

    def _process_systematic_entries(
        self,
        syst_entry: dict,
        params: Dict[str, list],
        tracer_prefix: str,
        probes_aux_config: Dict[str, Any],
        firecrown_params: Dict[str, Any]
    ) -> None:
        """
        Process individual systematic entry and update Firecrown parameters.

        Args:
            syst_entry (dict): Dictionary containing the systematic entry
                               configuration.
            params (Dict[str, list]): Dictionary mapping systematic types to
                                      parameter names.
            tracer_prefix (str): Prefix for tracer type (e.g., 'lens', 'src').
            probes_aux_config (Dict[str, Any]): Dictionary containing
                                                auxiliary probe configurations.
            firecrown_params (Dict[str, Any]): Dictionary to be updated with
                                               Firecrown parameters.
        """
        for survey_data in probes_aux_config.values():
            tracers = survey_data.get("tracers", {})
            for tracer, tracer_data in tracers.items():
                if tracer.startswith(tracer_prefix):
                    syst_type = syst_entry["type"]
                    for factory_type, param_list in params.items():
                        if syst_type == factory_type:
                            for param in param_list:
                                values = tracer_data.get(param, {})
                                for i, value in enumerate(values):
                                    key = f"{tracer}{i}_{param}"
                                    firecrown_params[key] = value

    def _process_factory_entries(
        self,
        factory_config: Dict[str, Any],
        tracer_prefix: str,
        params: Dict[str, list],
        probes_aux_config: Dict[str, Any],
        firecrown_params: Dict[str, Any]
    ) -> None:
        """
        Iterate over the factory configuration and process systematic entries.

        Args:
            factory_config (Dict[str, Any]): Dictionary containing the factory
                                             configurations.
            tracer_prefix (str): Prefix for tracer type (e.g., 'lens', 'src').
            params (Dict[str, list]): Dictionary mapping systematic types to
                                      parameter names.
            probes_aux_config (Dict[str, Any]): Dictionary containing
                                                auxiliary probe configurations.
            firecrown_params (Dict[str, Any]): Dictionary to be updated with
                                               Firecrown parameters.
        """
        for syst in factory_config.values():
            if isinstance(syst, list):
                for syst_entry in syst:
                    if isinstance(syst_entry, dict) and (
                        syst_entry.get("type") in params
                    ):
                        self._process_systematic_entries(
                            syst_entry, params, tracer_prefix,
                            probes_aux_config, firecrown_params
                        )

    def _process_systematic_factories(
        self,
        factories_config: Dict[str, Any],
        probes_aux_config: Dict[str, Any],
        firecrown_params: Dict[str, Any]
    ) -> None:
        """
        Process systematic factories and update Firecrown parameters.

        Args:
            factories_config (Dict[str, Any]): Dictionary containing the
                                               factory configurations.
            probes_aux_config (Dict[str, Any]): Dictionary containing
                                                auxiliary probe configurations.
            firecrown_params (Dict[str, Any]): Dictionary to be updated with
                                               Firecrown parameters.
        """
        systematics_params = {
            "nc_factory": {
                "lens": {
                    "PhotoZShiftFactory": ["delta_z"],
                    "PhotoZShiftandStretchFactory": ["delta_z", "sigma_z"],
                    "ConstantMagnificationBiasSystematicFactory": ["mag_bias"],
                },
                "spec": {
                    "PhotoZShiftFactory": ["delta_z"],
                    "PhotoZShiftandStretchFactory": ["delta_z", "sigma_z"],
                    "ConstantMagnificationBiasSystematicFactory": ["mag_bias"],
                },
            },
            "wl_factory": {
                "src": {
                    "PhotoZShiftFactory": ["delta_z"],
                    "PhotoZShiftandStretchFactory": ["delta_z", "sigma_z"],
                    "MultiplicativeShearBiasFactory": ["mult_bias"],
                },
            },
        }

        for factory, tracer_dict in systematics_params.items():
            factory_config = factories_config.get(factory)
            if isinstance(factory_config, dict):
                for tracer_prefix, params in tracer_dict.items():
                    self._process_factory_entries(
                        factory_config, tracer_prefix, params,
                        probes_aux_config, firecrown_params
                    )

    def _process_number_counts_bias(
        self,
        probes_aux_config: Dict[str, Any],
        firecrown_params: Dict[str, Any]
    ) -> None:
        """
        Process NumberCountsBias and update Firecrown parameters.

        Args:
            probes_aux_config (Dict[str, Any]): Dictionary containing
                                                auxiliary probe configurations.
            firecrown_params (Dict[str, Any]): Dictionary to be updated with
                                               Firecrown parameters.
        """
        for survey_data in probes_aux_config.values():
            tracers = survey_data.get("tracers", {})
            for tracer in filter(
                lambda t: t.startswith("lens") or t.startswith("spec"), tracers
            ):
                tracer_data = tracers.get(tracer, {})
                values = tracer_data.get("bias", {})
                for i, value in enumerate(values):
                    key = f"{tracer}{i}_bias"
                    firecrown_params.update({key: value})

    def _process_linear_alignment_factory(
        self,
        factories_config: Dict[str, Any],
        probes_aux_config: Dict[str, Any],
        firecrown_params: Dict[str, Any]
    ) -> None:
        """
        Process LinearAlignmentSystematicFactory entries and update parameters.

        Args:
            factories_config (Dict[str, Any]): Dictionary containing the
                                               factory configurations.
            probes_aux_config (Dict[str, Any]): Dictionary containing
                                                auxiliary probe configurations.
            firecrown_params (Dict[str, Any]): Dictionary to be updated with
                                               Firecrown parameters.
        """
        if factories_config.get("wl_factory") is not None:
            for syst in factories_config.get("wl_factory", {}).values():
                if isinstance(syst, list) and any(
                    isinstance(syst_entry, dict)
                    and syst_entry.get("type") ==
                    "LinearAlignmentSystematicFactory"
                    for syst_entry in syst
                ):
                    for survey_data in probes_aux_config.values():
                        tracers = survey_data.get("tracers", {})
                        for tracer, tracer_data in tracers.items():
                            if tracer.startswith("src") and isinstance(
                                    tracer_data, dict):
                                for param in ["ia_bias", "alphaz", "z_piv"]:
                                    if param in tracer_data:
                                        firecrown_params[param] = tracer_data[param]
                                        

    @staticmethod
    def _process_tracer_bool(tracer_params_bool: Dict[str, Any]) -> Dict[str,
                                                                         bool]:
        """
        Process boolean parameters and return a dictionary of parameter flags.

        Args:
            tracer_params_bool (Dict[str, Any]): Dictionary of tracer values.

        Returns:
            Dict[str, bool]: Dictionary of parameter flags.
        """
        params = ["bias", "delta_z", "sigma_z", "mag_bias", "mult_bias",
                  "ia_bias", "alphaz", "z_piv"]
        return {param: param in tracer_params_bool for param in params}

    def _add_systematic_factories(
        self,
        factories_config: Dict[str, Any],
        fact_key: str,
        factorie_aux: Dict[str, Any],
    ) -> None:
        """
        Add systematics to the factories on the factorie_aux dictionary.

        Args:
            factories_config (Dict[str, Any]): Factories configuration
                                               dictionary.
            fact_key (str): Key for the factory ("wl_factory" or "nc_factory").
            factorie_aux (Dict[str, Any]): Dictionary of parameter flags for
                                           the factory.
        """
        if factorie_aux[fact_key] is None:
            return

        for param, param_bool in factorie_aux[fact_key].items():
            if not param_bool:  # Skip if the parameter is disabled
                continue

            if param == "bias":
                pass  # No action for bias
            elif param == "delta_z":
                factories_config[fact_key]["per_bin_systematics"].append({
                    "type": "PhotoZShiftFactory"})
            elif param == "sigma_z":
                if factorie_aux[fact_key].get("delta_z", False):
                    try:
                        (factories_config[fact_key]
                         ["per_bin_systematics"].remove(
                            {"type": "PhotoZShiftFactory"}
                        ))
                    except ValueError:
                        pass
                    factories_config[fact_key]["per_bin_systematics"].append(
                        {"type": "PhotoZShiftandStretchFactory"}
                    )
                else:
                    raise ValueError("Missing nuisance parameter delta_z. "
                                     "Photo-z shift is required to use "
                                     "Photo-z stretch.")
            elif param == "mag_bias":
                factories_config[fact_key]["per_bin_systematics"].append(
                    {"type": "ConstantMagnificationBiasSystematicFactory"}
                )
            elif param == "mult_bias":
                factories_config[fact_key]["per_bin_systematics"].append(
                    {"type": "MultiplicativeShearBiasFactory"}
                )

        # Add LinearAlignmentSystematicFactory only if ia_bias, alphaz,
        # and z_piv are all True
        if (
            factorie_aux[fact_key].get("ia_bias", False)
            and factorie_aux[fact_key].get("alphaz", False)
            and factorie_aux[fact_key].get("z_piv", False)
        ):
            alphag = self.probes_config["probes"]["lsst"]["tracers"]["src"].get("alphag")
            if alphag is None:
                raise ValueError("Missing nuisance parameter alphag. "
                                 "LinearAlignmentSystematicFactory requires "
                                 "alphag.")
            # Add the LinearAlignmentSystematicFactory to the global systematics
            # of the factory
            linear_alignment_config = {"type":
                                       "LinearAlignmentSystematicFactory",
                                       "alphag": alphag}
            factories_config[fact_key]["global_systematics"].append(
                linear_alignment_config)

    @staticmethod
    def _initialize_factories_config(include_rsd: bool) -> Dict[str, Any]:
        """
        Initialize the factories configuration dictionary.

        Args:
            include_rsd (bool): Whether to include RSD in the configuration.

        Returns:
            Dict[str, Any]: Initialized factories configuration dictionary.
        """
        return {
            "nc_factory": {
                "per_bin_systematics": [],
                "global_systematics": [],
                "include_rsd": include_rsd,
            },
            "wl_factory": {
                "per_bin_systematics": [],
                "global_systematics": [],
            },
        }

    def _validate_and_extract_factories(
        self, tracer_factorie_config: Dict[str, Any]
    ) -> tuple:
        """
        Validate the tracer configuration and extract factories.

        Args:
            tracer_factorie_config (Dict[str, Any]): Dictionary of tracer
                                                     configurations.

        Returns:
            tuple: A tuple containing wl_factorie_aux and nc_factorie_aux.

        Raises:
            ValueError: If 'src' key is missing or nuisance parameters do not
                        match.
        """
        if "src" in tracer_factorie_config.keys():
            wl_factorie_aux = {"wl_factory": tracer_factorie_config["src"]}
        else:
            wl_factorie_aux = {"wl_factory": None}

        other_items = {k: v for k, v in tracer_factorie_config.items() if k != "src"}

        if ("lens" in tracer_factorie_config.keys()) or (
            "spec" in tracer_factorie_config.keys()
        ):
            if other_items:
                first_key, first_value = next(iter(other_items.items()))
                for key, value in other_items.items():
                    if value != first_value:
                        raise ValueError(
                            f"Nuisance parameters \'{key}\' do not match "
                            f"the parameters of \'{first_key}\'."
                        )
                nc_factorie_aux = {"nc_factory": first_value}
            else:
                nc_factorie_aux = {"nc_factory": None}
        else:
            nc_factorie_aux = {"nc_factory": None}

        return wl_factorie_aux, nc_factorie_aux

    def _config_factories_auxiliar(
        self, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Auxiliar function for building the factories configuration dictionary.

        Args:
            config (Dict[str, Any]): Initial configuration dictionary.

        Returns:
            Dict[str, Any]: Probes dictionary containing the parameters for
                            factories configuration.

        Raises:
            KeyError: If any required key is missing in the initial
                    configuration dictionary.
        """
        try:
            probes_aux_config = config["probes"]
            include_rsd = probes_aux_config["include_rsd"]
            probes_aux_config.pop("include_rsd")
        except KeyError as exc:
            raise KeyError(
                "Missing required key: 'include_rsd' in config['probes']"
            ) from exc

        tracer_factorie_config = {}
        for survey_data in probes_aux_config.values():
            for tracer_data, tracer_params_bool in survey_data.get("tracers", {}).items():
                tracer_factorie_config[tracer_data] = (
                    self._process_tracer_bool(tracer_params_bool)
                )

        wl_factorie_aux, nc_factorie_aux = (
            self._validate_and_extract_factories(tracer_factorie_config)
        )
        factories_config = self._initialize_factories_config(include_rsd)

        for factorie_aux in [wl_factorie_aux, nc_factorie_aux]:
            fact_key = None
            if factorie_aux is None:
                continue  # Skip if factorie_aux is None

            if "wl_factory" in factorie_aux:
                fact_key = "wl_factory"
            elif "nc_factory" in factorie_aux:
                fact_key = "nc_factory"

            if fact_key is not None:
                if factorie_aux[fact_key] is None or all(
                    value is None for value in factorie_aux[fact_key].values()
                ):
                    factories_config[fact_key] = None
                else:
                    self._add_systematic_factories(factories_config, fact_key,
                                                   factorie_aux)

        return factories_config

    def _config_augur_fisher(self) -> Dict[str, Any]:
        """
        Build the Fisher matrix configuration for Augur.

        Returns:
            Dict[str, Any]: Dictionary containing the Fisher matrix
                            configuration.
        """

        def split_param_regex(param):
            """
            Split the parameter string using a regex pattern.

            Args:
                param (str): Parameter string to split.

            Returns:
                tuple: A tuple containing tracer_aux, enum_aux, and param_aux.
            """
            # Modified regex pattern with proper grouping
            match = re.match(r"^((spec)_(bgs|lrg|elg)|(lens|src))(\d+)_(.+)$", param)
            if not match:
                raise ValueError(f"Invalid parameter format: {param}")
            # Determine tracer type
            if match.group(2):  # Prefixed case (spec_xxx)
                tracer_aux = f"{match.group(2)}_{match.group(3)}"
            else:  # Non-prefixed case (lens/src)
                tracer_aux = match.group(4)
            enum_aux = int(match.group(5))
            param_aux = match.group(6)
            return tracer_aux, enum_aux, param_aux

        params_aux = {}
        for param in self.firecrown_params:
            if param in self.cosmo_config["cosmology"]:
                cosmo_param = self.cosmo_config["cosmology"].get(param)
                cosmo_prior = self.prior_config["priors"].get(param)
                if cosmo_prior is not None:
                    fid_value = cosmo_param
                    params_aux[param] = [cosmo_prior[0], fid_value, cosmo_prior[1]]
                continue  # Skip to the next parameter

            # Handle special parameters: ia_bias, alphaz, z_piv and other systematics
            elif param in {"ia_bias", "alphaz", "z_piv"}:
                tracer_dict = self.probes_config["probes"]["lsst"]["tracers"][
                    "src"]
                if tracer_dict is not None:
                    src_sys_fid_value = tracer_dict.get(param)
                    try:
                        src_sys_prior = self.prior_config["priors"]["lsst"]["src"][param]
                        if src_sys_prior is not None:
                            params_aux[param] = [src_sys_prior[0],
                                                src_sys_fid_value,
                                                src_sys_prior[1]]
                    except KeyError:
                        print(f"KeyError: {param} not found in prior_config")
                        continue
                continue  # Skip to the next parameter

            elif param.startswith("lens") or param.startswith("src"):
                tracer_aux, enum_aux, param_aux = split_param_regex(param)
                tracer_dict = self.probes_config["probes"]["lsst"]["tracers"][tracer_aux].get(param_aux)
                if tracer_dict is not None:
                    tracer_sys_fid_value = tracer_dict[enum_aux]
                    try:
                        tracer_sys_prior = self.prior_config["priors"]["lsst"][tracer_aux][param_aux]
                        if tracer_sys_prior is not None:
                            if isinstance(tracer_sys_prior, list) and all(isinstance(item, list) for item in tracer_sys_prior):
                                params_aux[param] = [tracer_sys_prior[enum_aux][0], tracer_sys_fid_value[enum_aux], tracer_sys_prior[enum_aux][1]]
                            else:
                                params_aux[param] = [tracer_sys_prior[0], tracer_sys_fid_value, tracer_sys_prior[1]]
                    except KeyError:
                        print(f"KeyError: {param} not found in prior_config")
                        continue
            elif param.startswith("spec"):
                for survey in self.probes_config["probes"]:
                    if survey == "overlap" or survey == "lsst":
                        continue
                    tracer_aux, enum_aux, param_aux = split_param_regex(param)
                    tracer_dict = self.probes_config["probes"][survey]["tracers"][tracer_aux].get(param_aux)
                    if tracer_dict is not None:
                        tracer_sys_fid_value = tracer_dict[enum_aux]
                        try:
                            tracer_sys_prior = self.prior_config["priors"]["desi"][tracer_aux][param_aux]
                            if tracer_sys_prior is not None:
                                if isinstance(tracer_sys_prior, list) and all(isinstance(item, list) for item in tracer_sys_prior):
                                    params_aux[param] = [tracer_sys_prior[enum_aux][0], tracer_sys_fid_value[enum_aux], tracer_sys_prior[enum_aux][1]]
                                else:
                                    params_aux[param] = [tracer_sys_prior[0], tracer_sys_fid_value, tracer_sys_prior[1]]
                        except KeyError:
                            print(f"KeyError: {param} not found in prior_config")
                            continue
        #FIXME: Not sure if this is correct
        bias_params_aux = {}
        # for bias_params in self.firecrown_params:
        #     if bias_params in self.cosmo_config["cosmology"]:
        #         if self.cosmo_config["cosmology"][bias_params].get("bias_value") is not None:
        #             cosmo_bias = self.cosmo_config["cosmology"][bias_params].get("bias_value")
        #             if cosmo_bias is not None:
        #                 bias_params_aux[bias_params] = cosmo_bias
        #     elif bias_params in {"ia_bias", "alphaz", "z_piv"}:
        #         tracer_dict = self.probes_config["probes"]["lsst"]["tracers"]["src"].get(bias_params)
        #         if tracer_dict is not None:
        #             systematic_bias_value = tracer_dict.get("bias_value")
        #             if systematic_bias_value is not None:
        #                 bias_params_aux[bias_params] = systematic_bias_value
        #     elif bias_params.startswith("lens") or bias_params.startswith("src"):
        #         tracer_aux, enum_aux, param_aux = split_param_regex(bias_params)
        #         tracer_dict = self.probes_config["probes"]["lsst"]["tracers"][tracer_aux].get(param_aux)
        #         if tracer_dict is not None:
        #             tracer_sys_bias_value = tracer_dict.get("bias_value")
        #             if tracer_sys_bias_value is not None:
        #                 bias_params_aux[bias_params] = tracer_sys_bias_value[enum_aux]
        #     elif bias_params.startswith("spec"):
        #         for survey in self.probes_config["probes"]:
        #             if survey == "overlap" or survey == "lsst":
        #                 continue
        #             tracer_aux, enum_aux, param_aux = split_param_regex(bias_params)
        #             tracer_dict = self.probes_config["probes"][survey]["tracers"][tracer_aux].get(param_aux)
        #             if tracer_dict is not None:
        #                 tracer_sys_bias_value = tracer_dict.get("bias_value")
        #                 if tracer_sys_bias_value is not None:
        #                     bias_params_aux[bias_params] = tracer_sys_bias_value[enum_aux]

        augur_fisher_dict = {
            "fisher": {
                "parameters": params_aux,
                "step": self.config["general"]["fisher_step"],
                "output": self.config["general"]["fisher_output"] + "/fisher.dat",
                "fid_output": self.config["general"]["fisher_output"] + "/fiducials.dat",
                "bias_output": self.config["general"]["fisher_output"] + "/bias.dat",
                "fisher_bias": {"bias_params": bias_params_aux},
            }
        }
        return augur_fisher_dict

    def show_augur_config_parameters(self):
        """
        Show the parameters for augur of the config_builder object.
        """
        for param in self.firecrown_params:
            if param in self.fisher_config["fisher"]["parameters"]:
                print(f"    {param}: {self.fisher_config['fisher']['parameters'][param]}")
            else:
                print(f"    {param}: {self.firecrown_params[param]}")
        print("Fisher bias parameters:")
        for key, value in self.fisher_config["fisher"]["fisher_bias"]["bias_params"].items():
            print(f"    {key}: {value}")

    def config_builder(self) -> Dict[str, Any]:
        """
        Build the configuration dict by filtering data from multiple YAML files.

        Returns:
            Dict[str, Any]: Updated configuration dictionary with combined and
                    processed data.

        Raises:
            KeyError: If any required key is missing in the initial
                  configuration dictionary.
        """
        required_keys = ["cosmology_file", "array_choices_file", "probes_file", "probe_combinations_file"]
        try:
            yaml_paths = {key: self.config["general"][key] for key in required_keys}
        except KeyError as err:
            raise KeyError(f"Missing required key in config['general']: {err}") from err

        self.cosmo_config = load_yaml_file(yaml_paths["cosmology_file"])
        cosmo_aux = {}
        for key, value in self.cosmo_config["cosmology"].items():
            if key in ("mass_split", "extra_parameters"):
                continue
            cosmo_aux[key] = value

        self.array_config = load_yaml_file(yaml_paths["array_choices_file"])
        self.probes_config = load_yaml_file(yaml_paths["probes_file"])
        self.probes_comb_config = load_yaml_file(yaml_paths["probe_combinations_file"])
        probes_aux_config = self.probes_config.get("probes", {})
        self.factories_config = self._config_factories_auxiliar(self.probes_config)

        firecrown_params = cosmo_aux
        self.prior_config = load_yaml_file(self.config["general"]["priors_file"])
        self._process_systematic_factories(self.factories_config, probes_aux_config, firecrown_params)
        self._process_number_counts_bias(probes_aux_config, firecrown_params)
        self._process_linear_alignment_factory(self.factories_config, probes_aux_config, firecrown_params)
        self.firecrown_params = firecrown_params

        if self.config["general"]["fisher_builder"] == "Augur":
            self.fisher_config = self._config_augur_fisher()

        return self.firecrown_params
