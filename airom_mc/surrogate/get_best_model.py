import mlflow
import os
from biofuels_particle_surrogate.common.global_cfg import MLFLOW_URI
import argparse

def get_best_model(model_type, ddir):
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment = mlflow.get_experiment_by_name("biofuels_hyperopt")

    # Load all runs from the experiment
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                 filter_string=f"params.model_type = '{model_type}'")
    best_run = runs_df.loc[runs_df["metrics.test_loss"].idxmin()]
    best_run_id = best_run['run_id']
    artifact_list = mlflow.artifacts.list_artifacts(run_id=best_run_id)

    # Print the list of artifacts
    for artifact in artifact_list:
        if artifact.path[-4:] == ".eqx":
            
            mlflow.artifacts.download_artifacts(run_id=best_run_id, 
                                                dst_path="../data/surrogate/",
                                                artifact_path=artifact.path)

            in_f = os.path.join(ddir,artifact.path.split("/")[-1])
            if model_type == "differentiable":
                out_f = os.path.join(ddir,"model.eqx")
            else:
                out_f = os.path.join(ddir,f"model_{model_type}.eqx")
                
            os.system(f"cp {in_f} {out_f}")
    

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddir', default="../data/surrogate")
    args = parser.parse_args()
    
    get_best_model("differentiable",args.ddir)
    get_best_model("static",args.ddir)