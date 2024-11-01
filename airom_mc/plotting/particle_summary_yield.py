import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import argparse
import os

from biofuels_particle_surrogate.common.global_cfg import SEED
from biofuels_particle_surrogate.surrogate.dataloaders import CSVData, setup_df
from biofuels_particle_surrogate.surrogate.solvers import solve_model_scipy, solve_model_jax
from biofuels_particle_surrogate.surrogate.load_models import load
import biofuels_particle_surrogate.common.constants as c

from biofuels_particle_surrogate.plotting.plot_settings import palette_models,label_models,axis_labels, axis_lims

def create_plot(df,x,title,outdir):
    f,axes = plt.subplots(2,1)
    sns.pointplot(df,x=x,y='tar_yield',hue='model',
                  palette =palette_models,ax=axes[0],native_scale=True)
    axes[0].set_ylabel(axis_labels['tar_yield'])
    axes[0].set_xticks([])
    axes[0].set_xlabel("")
    axes[0].set_ylim(axis_lims['tar_yield'])
    axes[0].get_legend().set_visible(False)
    
    sns.pointplot(df,x=x,y='char_yield',hue='model',
                  palette =palette_models,ax=axes[1],native_scale=True)
    axes[1].set_ylabel(axis_labels['char_yield'])
    axes[1].set_xlabel(axis_labels[x])
    axes[1].set_ylim(axis_lims['char_yield'])
    axes[0].set_title(title)
    f.set_size_inches(5,8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,f'particle_summary_yield_{x}.pdf'))
    plt.clf()

def plot_results(cfg):
    y_cols = ['tar_mol','char_mol']
    x_integrated = False
    y_units = 'mol/mol'
    ds = CSVData(cfg.ddir, subset='validation',y_cols=y_cols,seed=SEED, 
                 x_integrated=x_integrated)
    dl = DataLoader(ds, batch_size=len(ds), shuffle=False)
    model = load(os.path.join(cfg.modeldir,'model.eqx'))
    model_static = load(os.path.join(cfg.modeldir,'model_static.eqx'))
    
    for X,Y,t,p in dl:
        df_comsol = setup_df(X.numpy(),Y.numpy(),t.numpy(),p.numpy(),ds._sim_idxs, name=label_models['comsol'])
        
        Y_base_0d = solve_model_scipy(X,p,t, y_cols,x_integrated=x_integrated, y_units=y_units,
                                     Ea=c.EA_0)
        Y_model_0d = solve_model_jax(model, X, t, p)
        
        Y_smodel_0d = solve_model_jax(model_static, X, t, p)
        df_base0d = setup_df(X.numpy(),Y_base_0d,t.numpy(),p.numpy(),ds._sim_idxs, name=label_models['baseline'])
        df_model0d = setup_df(X.numpy(),Y_model_0d,t.numpy(),p.numpy(), ds._sim_idxs,name=label_models['differentiable'])
        
        df_smodel0d = setup_df(X.numpy(),Y_smodel_0d,t.numpy(),p.numpy(),ds._sim_idxs, name=label_models['static'])
        
        
        df = pd.concat([df_comsol, df_base0d, df_model0d,df_smodel0d]).reset_index(drop=True)
        df = df.loc[df.groupby(['sim','model']).t.idxmax()]

        create_plot(df.query("aspect == 4 & T_oven == 773.0 & FL > 1.0"),
                    "FL",
                    title="$T_{oven}$ = 773 [K], Aspect = 4",
                    outdir=cfg.outdir)
        

        create_plot(df.query("FL == 4.5 & T_oven == 773.0"),
                    "aspect",
                    title="$T_{oven}$ = 773 [K], FL = 4.5",
                    outdir=cfg.outdir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ddir', default="../data/")
    parser.add_argument('--modeldir', default="../data/surrogate")
    parser.add_argument('--outdir', default="../outputs")
    args = parser.parse_args()
    plot_results(args)