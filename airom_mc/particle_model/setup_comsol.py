import argparse
import pandas as pd
import os
import glob
import numpy as np
import mph

def setup_lhs(idx):
    ddir = '../data/'
    df_ICs = pd.read_csv(os.path.join(ddir,"comsol_ICs.csv"))
    row = df_ICs.iloc[idx]

    
    outpath = os.path.join(ddir,f"particle_models/")

    client = mph.start()
    
    model = client.load(os.path.join(ddir,'particle_models/example.mph'))
    
    model.parameter('FL',f'{row.FL} [mm]')
    model.parameter('aspect',f'{row.aspect}')
    model.parameter('T_oven', f'{row.T_oven} [K]')
    

    model.save(os.path.join(outpath,f'model_{idx:03d}.mph'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('idx')
    args = parser.parse_args()

    setup_lhs(int(args.idx))
