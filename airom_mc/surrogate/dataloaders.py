from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import airom_mc.common.constants as c
from airom_mc.common.utils import calc_volume, calc_surface_area


class CSVData(Dataset):
    
    def __init__(self, ddir, subset='all', y_cols=None, x_integrated=None,
                 seed=None, seed_add=0,add_params=False, include_static_opt=False,
                 train_size_subset=1.0,y_units="mol/mol", tscale_loss=False):
        
        self.ddir = ddir
        self.seed = seed
        self.seed_add = seed_add
        self.subset = subset
        self.y_units = y_units
        self.train_size_subset = train_size_subset
        
        if y_cols is None:
            y_cols = ['tar_mol']
        self.y_cols = y_cols
        
        self.x_integrated = x_integrated
        if self.x_integrated:
            self.x_cols = ['surface_area','volume']
        else:
            self.x_cols = ['FL','aspect']
            
        self._setup_data()
        self.include_static_opt = include_static_opt
        if include_static_opt:
            self._setup_static_data(tscale_loss)
        if add_params:
            self._add_params()
        
        self._create_subset(subset)
        
    def _setup_static_data(self, tscale_loss):
        if tscale_loss:
            fname = 'static_opt_tscale.csv'
        else:
            fname = 'static_opt.csv'
        df = pd.read_csv(os.path.join(self.ddir, fname),index_col=0)
        df = df[[c for c in df.columns if 'Ea' in c]]
        self.EA = df.loc[self._sim_idxs].values/10

        
    def _setup_data(self):
        self.output_df = pd.read_csv(os.path.join(self.ddir, "particle_eval_agg.csv"))
        
        self._sim_idxs = self.output_df.sim.unique()
        
        if self.subset == "validation":
            self.output_df = self.output_df.query("sim >= 250")
            self._sim_idxs = [i for i in self._sim_idxs if i >=250]
        elif self.subset in ("train",'test','tiny_train'):
            self.output_df = self.output_df.query("sim <250")
            self._sim_idxs = [i for i in self._sim_idxs if i <250]
        elif self.subset == 'all':
            pass
            
        self.input_df = pd.read_csv(os.path.join(self.ddir,"comsol_ICs.csv"))
        self.input_df = self.input_df.loc[self._sim_idxs]
        self.input_df['volume'] = calc_volume(self.input_df['FL'],self.input_df['aspect'])
        self.input_df['surface_area'] = calc_surface_area(self.input_df['FL'],self.input_df['aspect'])
        
        self.X = self.input_df[self.x_cols].values
        self.Y = np.array([self.output_df.query(f"sim == {idx}")[self.y_cols] for idx in self._sim_idxs])
        
        if self.y_units == "mol/mol":
            self.Y = self.Y/(self.input_df['volume'].values[:,np.newaxis,np.newaxis]*c.rho/c.MW_char)
        elif self.y_units == "mol":
            pass
        else:
            raise(ValueError)

        
        self.t = self.output_df.query(f"sim == {self._sim_idxs[0]}").t.values
        self.params = self.input_df['T_oven'].values
        
    def _create_subset(self, subset):
        idxs_tr, idxs_te = train_test_split(np.arange(self.X.shape[0]), 
                                            test_size=0.2, train_size=0.8, 
                                            random_state=self.seed)
        self.idxs_tr = idxs_tr
        self.idxs_te = idxs_te
        
        if self.train_size_subset < 1.0:
            idxs_tr, _ = train_test_split(idxs_tr,train_size=self.train_size_subset,
                                          random_state=self.seed+self.seed_add)
        
        if subset == "train":
            self.X = self.X[idxs_tr]
            self.Y = self.Y[idxs_tr]
            self.params = self.params[idxs_tr]
            if self.include_static_opt:
                self.EA = self.EA[idxs_tr]
        elif subset == 'test':
            self.X = self.X[idxs_te]
            self.Y = self.Y[idxs_te]
            self.params = self.params[idxs_te]
            if self.include_static_opt:
                self.EA = self.EA[idxs_te]
        elif subset == 'all' or subset == 'validation':
            pass
        elif subset == 'tiny_train':
            self.X = self.X[idxs_tr[:2]]
            self.Y = self.Y[idxs_tr[:2]]
            self.params = self.params[idxs_tr[:2]]
            if self.include_static_opt:
                self.EA = self.EA[idxs_tr[:2]]
        else:
            raise(ValueError)
        
    
    def __len__(self):
        return self.X.shape[0]            

    def __getitem__(self,i):
        x = self.X[i]
        y = self.Y[i]
        t = self.t
        p = self.params[i]
        if self.include_static_opt:
            EA = self.EA[i]
        
            return x, y, t, p, EA

        else:
            return x, y, t, p
        