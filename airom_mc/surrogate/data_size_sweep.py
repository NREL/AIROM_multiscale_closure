import numpy as np
import argparse

from biofuels_particle_surrogate.common.utils import dotdict
from biofuels_particle_surrogate.surrogate.train_jax import train as train_NDE
from biofuels_particle_surrogate.surrogate.train_static import train as train_static
from biofuels_particle_surrogate.common.global_cfg import SEED

def hyperopt(idx,cfg):
    ds_size = np.logspace(-2,0,25)
    cfg['train_size_subset'] = ds_size[idx]
    
    if cfg.model_type == "differentiable":
        train_NDE(cfg)
    elif cfg.model_type == "static":
        train_static(cfg)
    else:
        raise(ValueError)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('idx')
    
    parser.add_argument('--batch_size', type=int, default=23)
    parser.add_argument('--N_epochs', type=int, default=25)
    parser.add_argument("--y_cols", nargs='+',type=str, default=['tar_mol','char_mol'])
    parser.add_argument("--loss_scale",action='store_true')
    parser.add_argument("--no_tracking",action='store_true')
    parser.add_argument("--experiment",type=str, default="biofuels_ds_size")
    parser.add_argument("--model_type",type=str, default="differentiable")
    parser.add_argument("--seed_shift",type=int,default=0)
    parser.add_argument('--N_hidden', type=int, default=2)
    parser.add_argument("--S_hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--lr_decay",type=float, default=None)
    parser.add_argument("--optimizer",default='adam')
    parser.add_argument("--shuffle",action='store_true')
    args = parser.parse_args()

    cfg = dotdict(vars(args))
    cfg['SEED'] = SEED 

    hyperopt(int(args.idx), cfg)