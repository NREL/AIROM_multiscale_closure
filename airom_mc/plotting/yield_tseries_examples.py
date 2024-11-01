import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from torch.utils.data import DataLoader

from airom_mc.common.global_cfg import SEED
from airom_mc.surrogate.dataloaders import CSVData
from airom_mc.surrogate.utils import solve_model_scipy, solve_model_jax, setup_df
from airom_mc.surrogate.load_models import load
import airom_mc.common.constants as c
from airom_mc.plotting.plot_settings import palette_models,label_models, axis_lims

def create_plot(df,outdir):
    f,axes = plt.subplots(2, len(df.sim.unique())+1)
    for y_i, y in enumerate(['tar','char']):
        for x_i, sim in enumerate(df.sim.unique()):
            ax = axes[y_i,x_i]
            sns.lineplot(df.query(f"sim == {sim}"),x='t',
                         y=f'{y}_yield',hue='model',
                         palette=palette_models,ax=ax,
                         )
            #dashes=['','',(2,2),(2,2)]
            ax.set_ylim(0,axis_lims[f'{y}_yield'][1])
            if y_i == 0:
                ax.set_xlabel("")
                ax.set_xticks([])
                ax.set_title(f"Sim: {x_i}")
            if y_i == 1:
                ax.set_xlabel("Time [s]")
                ax.set_xticks([0,5,10,15,20])
            if x_i == 0:
                ax.set_ylabel(f"{y.capitalize()} Yield [mol/mol]")
            else:
                ax.set_ylabel("")
                ax.set_yticks([])

            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().set_visible(False)   
        
        axes[1, -1].legend(handles, labels, bbox_to_anchor=(1.1, .15), 
                           loc='lower right')

        axes[0,-1].set_axis_off()
        axes[1,-1].set_axis_off()
        
    plt.gcf().set_size_inches(12,4)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(os.path.join(outdir,'yield_tseries_examples.pdf'))
        

def plot_results(cfg):
    y_cols = ['tar_mol', 'char_mol']
    x_cols = ['FL','aspect']
    y_units = 'mol/mol'
    x_integrated=False
    ds = CSVData('../data', subset='test',y_cols=y_cols,seed=SEED,x_integrated=x_integrated)
    dl = DataLoader(ds, batch_size=6, shuffle=False)
    
    
    model_diff = load(os.path.join(cfg.ddir,'model.eqx'))
    model_static =  load(os.path.join(cfg.ddir,'model_static.eqx'))
    
    for X,Y,t,p in dl:
        
        df_comsol = setup_df(X,Y,t,p,ds._sim_idxs, name=label_models['comsol'])
        
        Y_base_0d = solve_model_scipy(X,p,t, y_cols,x_integrated=x_integrated, y_units=y_units,
                                     Ea=c.EA_0)
        Y_model_0d = solve_model_jax(model_diff, X, t, p)
        Y_smodel_0d = solve_model_jax(model_static, X, t, p)
        
        df_base0d = setup_df(X,Y_base_0d,t,p,ds._sim_idxs, name=label_models['baseline'])
        df_model0d = setup_df(X,Y_model_0d,t,p,ds._sim_idxs, name=label_models['differentiable'])
        df_smodel0d = setup_df(X,Y_smodel_0d,t,p,ds._sim_idxs, name=label_models['static'])
        
        
        df = pd.concat([df_comsol, df_base0d, df_model0d, df_smodel0d])

        create_plot(df, cfg.outdir)

        break
    
    

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddir', default="../data/surrogate")
    parser.add_argument('--outdir', default="../outputs")
    args = parser.parse_args()
    plot_results(args)