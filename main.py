import argparse
import pandas as pd
import numpy as np

from fit_functions import run_aa, run_aa_flows, create_sankey_json
from make_figures import make_figures
from make_tables import make_table_data

def main(args):

    """ 
    Performs sector decompositions and other ML calculations described in the associated paper 

    Input:

        verbose - flag to print progress if desired

    Output:

        Saves decompositions and other calculations in 'aa-stocks/saves/'

        Produces figures in png format, table data in txt/LaTex format, and sankey data in json format
        Saves output to folder 'aa-stocks/figures/'
    """

    dataframe = pd.read_pickle('data/data.pkl')

    # Perform Generic Archetypal Analysis on stock dataset
    run_aa(dataframe, verbose=args.verbose)

    # Perform Archetypal Analysis on stock dataset with market mode removed 
    # to compare with fama and french factors
    run_aa(dataframe, ff=True, verbose=args.verbose)

    # Perform AA sliding across time with a gaussian weight
    run_aa_flows(dataframe, verbose=args.verbose)
    
    # Perform AA sliding across time with a gaussian weight
    # where certain companies are replaced with noise
    run_aa_flows(dataframe, noise=True, verbose=args.verbose)

    # Create Sankey Diagram JSON
    if args.verbose:
        print 'Creating json file for sankey diagram'
    create_sankey_json()

    # Make Figures    
    if args.verbose:
        print 'Generating and saving figures'
    make_figures()

    # Make Tables
    if args.verbose:
        print 'Generating and saving table data'
    make_table_data()

    return

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='Set verbosity')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    
    main(args)
