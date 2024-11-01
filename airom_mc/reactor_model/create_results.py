import matplotlib.pyplot as plt
import numpy as np
import os
from pv_utils import pv_calc_gas_outlet, pv_calc_biomass_outlet


def create_datfiles(case='base'):
    prefix = '../data/reactor_models/{0}/'.format(case)
    for idx in range(0,5):
        prefix_idx = os.path.join(prefix, "run_00{0}".format(idx))
        print(prefix_idx)
        infile = os.path.join(prefix_idx,'soln.foam')
        outfile_gas = os.path.join(prefix_idx,'gas.dat')
        outfile_biomass = os.path.join(prefix_idx,'biomass.dat')

        pv_calc_biomass_outlet(infile, outfile_biomass)
        pv_calc_gas_outlet(infile, outfile_gas)
        
        
if __name__ == "__main__":
    #create_datfiles('surrogate')
    #create_datfiles('base')
    create_datfiles('base_3d')
    #create_datfiles('surrogate_3d')
    

