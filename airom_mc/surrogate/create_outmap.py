import jax
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import os

from airom_mc.surrogate.load_models import load
from airom_mc.surrogate.dataloaders import CSVData
from airom_mc.common.global_cfg import SEED

def create_outmap(case,ddir):
    y_cols = ['tar_mol','char_mol']
    if case == 'differentiable':
        filename = os.path.join(ddir,'model.eqx')
    elif case == 'static':
        filename = os.path.join(ddir,'model_static.eqx')
    
    model,params = load(filename,return_params=True)
    
    ds = CSVData('../data',y_cols=y_cols, 
                 seed=SEED,
                 x_integrated=params['x_integrated'])
    
    dl = DataLoader(ds, batch_size=len(ds), shuffle=False)
    
    for X,Y,t,p in dl:
        Ea = jax.vmap(model.parameter)(X.numpy())*10
        df = pd.DataFrame(Ea,columns=[f'Ea_{i}' for i in range(10)])
        df.index = ds._sim_idxs 
        df.to_csv(f"../data/outmap_{case}.csv")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddir', default="../data/surrogate")

    
    args = parser.parse_args()

    create_outmap("differentiable",args.ddir)
    create_outmap("static",args.ddir)