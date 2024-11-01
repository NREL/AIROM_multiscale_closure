import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

init_flow_rate = 8.33e-5

def plot_results(cfg):
    df = pd.read_csv(os.path.join(cfg.ddir,"reactor_results.csv"))
    
    df['tar_yield'] = 100*df['mflux_tr_tot']/init_flow_rate
    df['t'] = df['t[i]']
    
    df['FL'] = df.FL.astype(float)
    df.loc[df.case == 'base_3d','FL'] +=0.1
    df['case_label'] = df.replace({'case':{"base":"OpenFOAM 3D",
                                            "surrogate":"OpenFOAM 3D + Surrogate"}})['case']
    
    
    ax = sns.pointplot(df.query("t >4"), x='FL',y='tar_yield',
                  hue='case_label', errorbar='sd',capsize=1,
                  native_scale=True,linestyle='--',
                  palette = {"OpenFOAM 3D":'darkgrey',
                            'OpenFOAM 3D + Surrogate': 'tab:blue'})
    
    ax.lines[0].set_linestyle('--')
    ax.legend().get_lines()[0].set_linestyle('--')
    plt.xlabel("FL [mm]")
    plt.ylabel("Tar Yield [%]")
    plt.savefig(os.path.join(cfg.outdir,"reactor_summary_yield.pdf"))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ddir', default="../data/")
    parser.add_argument('--outdir', default="../outputs")
    args = parser.parse_args()
    plot_results(args)