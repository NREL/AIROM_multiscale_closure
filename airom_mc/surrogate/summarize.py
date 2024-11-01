import mlflow
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from airom_mc.common.global_cfg import MLFLOW_URI, SEED
from airom_mc.surrogate.utils import calc_err, solve_model_scipy

from airom_mc.surrogate.dataloaders import CSVData
MLFLOW_URI =  "http://develo-devel-dgwauajbvdfe-0b87827112286405.elb.us-west-2.amazonaws.com/"


def get_best_model(model_type,metric):
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Load all runs from the experiment
    experiment = mlflow.get_experiment_by_name("biofuels_hyperopt")
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_type = '{model_type}'"
    )
    
    # get the best run
    best_run = runs_df.loc[runs_df[metric].idxmin()]

    metrics =  ['metrics.train_loss_best','metrics.test_loss_best']
    if model_type == 'static':
        metrics += ["metrics.train_loss_static_best","metrics.test_loss_static_best"]

    row =  best_run[metrics]
    row.rename({'metrics.train_loss_best':'train_loss',
                'metrics.test_loss_best':"test_loss",
                "metrics.train_loss_static_best":'train_loss_static',
                "metrics.test_loss_static_best":"test_loss_static"
                }, inplace=True)
    return row
    
def compute_baseline(cfg, optimized=False):
    ds_train = CSVData(cfg.ddir, subset='train',seed=SEED,
                 y_units='mol/mol',y_cols=['tar_mol','char_mol'],
                 include_static_opt=True, tscale_loss=True)
    ds_test = CSVData(cfg.ddir, subset='test',seed=SEED,
                 y_units='mol/mol',y_cols=['tar_mol','char_mol'],
                 include_static_opt=True, tscale_loss=True)
    dl_train = DataLoader(ds_train, batch_size=len(ds_train))
    dl_test = DataLoader(ds_test, batch_size=len(ds_test))
    
    
    for X,Y,t,p,EA in dl_train:
        if optimized:
            yp = solve_model_scipy(X,p,t, ['tar_mol','char_mol'], False, "mol/mol",Ea=EA*10)
        else:
            yp = solve_model_scipy(X,p,t, ['tar_mol','char_mol'], False, "mol/mol",Ea=None)
        train_err = calc_err(Y.numpy(),yp,t=t.numpy())
    
    for X,Y,t,p,EA in dl_test:
        if optimized:
            yp = solve_model_scipy(X,p,t, ['tar_mol','char_mol'], False, "mol/mol",Ea=EA*10)
        else:
            yp = solve_model_scipy(X,p,t, ['tar_mol','char_mol'], False, "mol/mol",Ea=None)
        test_err = calc_err(Y.numpy(),yp,t=t.numpy())

    return pd.Series({"train_loss":train_err,
                      "test_loss":test_err})

def summarize(cfg):
    
    r_diff = get_best_model("differentiable","metrics.test_loss_best")
    r_stat = get_best_model("static","metrics.test_loss_best")
    r_baseline = compute_baseline(cfg)
    r_baseline_opt = compute_baseline(cfg,optimized=True)

    df = pd.DataFrame(columns=r_stat.index)
    df.loc["A Priori"] = r_stat
    df.loc["A Posteriori"] = r_diff
    df.loc["Baseline"] = r_baseline
    df.loc["Baseline Optimized"] = r_baseline_opt


    print(df.to_latex(float_format='%.2E').replace('NaN','-'))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddir",default='../data/')
    args = parser.parse_args()
    
    summarize(args)