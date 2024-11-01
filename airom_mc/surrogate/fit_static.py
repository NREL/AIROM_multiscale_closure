import numpy as np
import pandas as pd
from multiprocessing import Pool
import scipy
import argparse
from torch.utils.data import DataLoader
import os

import airom_mc.common.constants as c
from airom_mc.surrogate.dataloaders import CSVData
from airom_mc.common.global_cfg import SEED
from airom_mc.surrogate.utils import solve_model_scipy,calc_err
from airom_mc.common.utils import dotdict



x_integrated = False

def objective(EA, X,Y,t,p,y_cols,tscale_loss=False):
    #X,Y,t,p,y_cols = dat
   
    y_units = 'mol/mol'
    ypred = solve_model_scipy(X,p,t, y_cols,x_integrated=x_integrated, y_units=y_units,Ea=EA)
    if tscale_loss:
        err = calc_err(ypred,Y.numpy(),t.numpy())
    else:
        err = calc_err(ypred,Y.numpy())

    
    return err


def run_opt(grp):
    dat,idx,cfg,sim_idx = grp

    print(f'Optimizing {idx, sim_idx}')
    final_res = None
    best_fun = 1000
    for method in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'Powell']:
        try:
            res = scipy.optimize.minimize(objective, c.EA_0, args=(*dat,cfg.y_cols,cfg.tscale_loss),
                                            bounds=[[10,1000] for i in range(10)],
                                            method=method)
        except ValueError:
            continue
        if res.fun < best_fun:
            final_res = res
            best_fun = res.fun
    return np.r_[final_res.x,final_res.fun]
    
def fit_static(cfg):
    ds = CSVData(cfg.ddir,y_cols=cfg.y_cols, seed=cfg.SEED,x_integrated=x_integrated)
    idx = ds._sim_idxs
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    results = []
    pool = Pool(104)
    results = pool.map(run_opt,zip(dl, range(len(ds)),[cfg]*len(ds), idx))
    results = np.array(results)
    df = pd.DataFrame(results,columns=[f'Ea_{i}' for i in range(10)]+['loss'])
    df.index = ds._sim_idxs
    
    if cfg.tscale_loss:
        df.to_csv(os.path.join(cfg.ddir,"static_opt_tscale.csv"))
    else:
        df.to_csv(os.path.join(cfg.ddir,"static_opt.csv"))
        
        
        
if __name__=="__main__":
    x_cols = ['FL','aspect']
    y_units = 'mol/mol'
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_cols", nargs='+',type=str, default=['tar_mol','char_mol'])
    parser.add_argument("--tscale_loss",action='store_true')
    parser.add_argument('--ddir', default="../data/")

    
    args = parser.parse_args()

    cfg = dotdict(vars(args))
    cfg['SEED'] = SEED

    
    fit_static(cfg)