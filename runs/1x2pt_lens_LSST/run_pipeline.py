"""Brain script to run the analysis.

Description:
    This script will use the necessary modules to generate the sacc files,
    build the likelihood, and run the Fisher forecast.
"""

# Third-party library imports
import sys
import numpy as np
import os
import argparse

# augur and sacc import
import sacc
from augur.analyze import Analyze

# correcting the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../')))

#pylint: disable=wrong-import-position
#pylint: disable=import-error
# Importing the functions from likelihood_build.py
from likelihood_build import build_likelihood

# Importing the functions from utils.py
from utils import build_modeling_tools

# Importing the functions from config_builder.py
from config_builder import load_yaml_file, ConfigBuilder

# Importing the sacc_generator function
from sacc_generator import sacc_generator

if __name__ == "__main__":

    # Define the path to the general configuration file
    parser = argparse.ArgumentParser(description="Receive the path to the general config file")
    parser.add_argument("general_path", help="Path to genera yaml file")
    args = parser.parse_args()

    # Load the configuration file
    config = load_yaml_file(args.general_path)
    cfg = ConfigBuilder(config)
    cfg.config_builder()
       
    # Get the name of the sacc file
    sacc_file = cfg.config['general']['sacc_file']

    # Generate the sacc file
    try:
        sacc.Sacc.load_fits(sacc_file)
        print("Sacc file found: ", sacc_file)
        print("No need to generate a new one.")
    except FileNotFoundError:
        sacc_generator(cfg)

    # Build the likelihood
    print("Building likelihood...")
    likelihood = build_likelihood(cfg)

    # Starts Fisher forecast:
    # Define the parameters for the Fisher forecast
    params = cfg.firecrown_params
    print("Parameters for the Fisher forecast:")
    cfg.show_augur_config_parameters()

    # Define modeling tools
    tools = build_modeling_tools(cfg.cosmo_config)

    # Initialize the Analyze Class from augur
    # FIXME: Implement other sources of run Fisher matrix
    ao = Analyze(cfg.fisher_config,
                 likelihood=likelihood,
                 tools=tools,
                 req_params=params)

    # check if the Fisher matrix is already computed
    try:
        print("Fisher matrix already computed. Loading from file...")
        Fij = np.loadtxt(f'{config["general"]["fisher_output"]}/fisher.dat')
        print('Fisher matrix:')
        print(f"    {Fij}")
        exit(0)
    except FileNotFoundError:      
        # Compute the Fisher matrix
        print("Computing Fisher matrix...")
        Fij = ao.get_fisher_matrix(method='numdifftools')
        print('Fisher matrix:')
        print(f"    {Fij}")
        exit(0)


    #t =  np.linalg.inv(ij)
    # print(t)
    # exit(-1)

    # Compute the Fisher bias
    # BFij = ao.get_fisher_bias()
    # tab_out = Table(BFij, names=ao.var_pars)
    # tab_out.write(config['fisher']['bias_output'],
    #               format='ascii', overwrite=True)
    # print('Fisher bias:')
    # print(f"    {BFij}")
