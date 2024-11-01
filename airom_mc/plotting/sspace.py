from biofuels_particle_surrogate.surrogate.load_models import load
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import biofuels_particle_surrogate.common.constants as c
import pandas as pd
import argparse
import os

from scipy.interpolate import griddata


ranges = {"FL":(0.9,10.1),
          "aspect":(1.9,8.1),
          "T_oven":(720,830)}
labels = {"FL":"FL [mm]",
          "aspect": "Aspect",
          "T_oven": r"$T_{oven}$ [K]"}

climits = {
    'Ea_0':25,
    'Ea_1':50,
    'Ea_2':25,
    'Ea_3':30,
    'Ea_4':20,
    'Ea_5':25,
    'Ea_6':25,
    'Ea_7':20,
    'Ea_8':30,
    'Ea_9':20,
        
}


def plot_ax(df, case, x, y, ax):
    
    xlim = ranges[x]
    ylim = ranges[y]
    
    xl = np.linspace(xlim[0],xlim[1],105)
    yl = np.linspace(ylim[0],ylim[1],100)

    xd = df[x].values
    yd = df[y].values
    zd = df[case].values

    zg = griddata((xd, yd), zd, (xl[None,:], yl[:,None]), method='nearest')
    im = ax.pcolormesh(xl, yl, zg-c.EA_0[int(case.split('_')[1])], shading='auto', rasterized=True,
                       #vmin=-1*climits[case],vmax=climits[case],
                        vmin=-25,vmax=25,
                       cmap='PiYG')
    
    ax.set_xlabel(labels[x])
    ax.set_ylabel(labels[y])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    return im

def create_sspace_viz(cfg,case):
    
    df_ICs = pd.read_csv(os.path.join(cfg.ddir,"comsol_ICs.csv"))
    if case == "static":
        df_EA = pd.read_csv(os.path.join(cfg.ddir,"static_opt_tscale.csv"))
        df = pd.merge(df_EA,df_ICs,left_on='Unnamed: 0',right_index=True)
    elif case == 'differentiable':
        df_EA = pd.read_csv(os.path.join(cfg.ddir,"outmap_differentiable.csv"))
        df = pd.merge(df_EA,df_ICs,left_on='Unnamed: 0',right_index=True)
    elif case == 'static_opt':
        df_EA = pd.read_csv(os.path.join(cfg.ddir,"outmap_static.csv"))
        df = pd.merge(df_EA,df_ICs,left_on='Unnamed: 0',right_index=True)
    
    f,axes = plt.subplots(3,10, height_ratios=[1.2,1,1])
    
    for i in range(10):
        im = plot_ax(df.query("T_oven == 773"),f'Ea_{i}','FL',"aspect",axes[0,i])
        im = plot_ax(df.query("aspect == 4"),f'Ea_{i}','FL',"T_oven",axes[1,i])
        im = plot_ax(df.query("FL == 5.6"),f'Ea_{i}','aspect',"T_oven",axes[2,i])

        if i != 0:
            for j in range(3):
                axes[j,i].set_yticks([])
                axes[j,i].set_ylabel("")

        plt.colorbar(im,ax=axes[0,i],location='top',orientation='horizontal')
    
    f.set_size_inches(12,4)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir,f"sspace_{case}.pdf"))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddir', default="../data/")
    parser.add_argument('--outdir', default="../outputs")
    args = parser.parse_args()
    
    create_sspace_viz(args,case='static')
    create_sspace_viz(args,case='differentiable')
    create_sspace_viz(args,case='static_opt')
