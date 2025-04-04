# (3x2pt) Fisher Forecasting with Augur and CosmoSIS

To execute the pipeline:

1.  Duplicate the `base_run` directory.
2.  Modify the configuration files within the `config_yamls` directory according to your specifications.
3.  Run the pipeline using the following command:

    ```bash
    python run_pipeline config_yamls/augur_config/augur_config.yaml
    ```

    This script will generate the necessary SACC file based on your configurations. Note: The `covariance.py` file is currently sourced from the OneCovariance repository for use within this pipeline. The Fisher matrix is then computed using Augur.

For CosmoSIS Fisher matrix computation, execute the following commands from within the `run` directory:

```bash
# Convert the Augur configuration file to a CosmoSIS-compatible values.ini file.
python cosmosis_analysis/augur_to_cosmosis_converter.py config_yamls/augur_config/augur_config.yaml cosmosis_analysis/values/values_3x2pt_strawberry.ini

# Compute the Fisher matrix using CosmoSIS.
cosmosis cosmosis_analysis/3x2pt_strawberry.ini

# Post-process the CosmoSIS output to generate the param_means file.
cosmosis-postprocess --no-plots -o cosmosis_analysis/fisher_output/3x2pt_strawberry/ cosmosis_analysis/fisher_output/3x2pt_strawberry/3x2pt_strawberry.txt
```

Once the Fisher matrix computation is complete, use the `contour_plots` notebook located in the `augur_analysis` directory to generate the desired contour plots.
