import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mlflow
import os
import argparse
from biofuels_particle_surrogate.common.global_cfg import MLFLOW_URI
from biofuels_particle_surrogate.plotting.plot_settings import palette_models,label_models
from mlflow.tracking import MlflowClient



def plot_ds_size(cfg):
    
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Initialize MLflow client
    client = MlflowClient()
    
    experiment = mlflow.get_experiment_by_name("biofuels_ds_size")
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )
    #runs_df = runs_df[runs_df['params.model_type']!='differentiable']
    runs_df['train_size'] = runs_df["params.train_ds_size"].astype(int)
    runs_df['params.model_type'] = runs_df['params.model_type'].replace(label_models)
    
    ax = sns.lineplot(data=runs_df, x='train_size',
                  y='metrics.test_loss_best',
                  hue='params.model_type',
                  errorbar='sd',
                  estimator='mean',
                  palette=palette_models)

    
    ax.set_ylabel('Minimum Test Loss')
    ax.set_xlabel("# Training Examples")
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.legend()
    sns.move_legend(ax,"upper right",title='Model Type')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir,'ds_size.pdf'))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default="../outputs")
    args = parser.parse_args()
    
    plot_ds_size(args)