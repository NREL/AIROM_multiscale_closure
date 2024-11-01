
import jax.numpy as jnp
import jax
import optax
import time
import mlflow
import argparse
import equinox as eqx
from torch.utils.data import DataLoader
import os

from biofuels_particle_surrogate.surrogate.dataloaders import CSVData
from biofuels_particle_surrogate.surrogate.eqx_modules import NeuralODE
from biofuels_particle_surrogate.common.utils import dotdict
import biofuels_particle_surrogate.common.constants as c
from biofuels_particle_surrogate.common.global_cfg import MLFLOW_URI, SEED
from biofuels_particle_surrogate.surrogate.jax_utils import train_loop, test_loop
from biofuels_particle_surrogate.surrogate.load_models import save

def train(cfg):
    
    use_mlflow = not cfg.no_tracking
    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_URI)
        
        experiment = mlflow.set_experiment(cfg.experiment)
        run = mlflow.start_run(
            experiment_id=experiment.experiment_id,
            description="base",)
        mlflow.log_params(cfg)
        fmodel = f"../data/surrogate/model_{run.info.run_id}.eqx"
    else:
        fmodel = f"../data/surrogate/model_testing.eqx"
    
    key =  jax.random.key(cfg.SEED+cfg.seed_shift)
    
    ds_train = CSVData('../data', subset='train',y_cols=cfg.y_cols,seed=SEED,
                 y_units='mol/mol', x_integrated=cfg.x_integrated,
                 train_size_subset=cfg.train_size_subset,seed_add=cfg.seed_shift)
    ds_test = CSVData('../data', subset='test',y_cols=cfg.y_cols,seed=SEED,
                 y_units='mol/mol', x_integrated=cfg.x_integrated)
    mlflow.log_params({"train_ds_size":len(ds_train)})
    mlflow.log_params({"slurm_job_id":os.getenv("SLURM_JOB_ID")})
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    dl_test = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
    
    trainloader = dl_train
    testloader = dl_test
    print(len(testloader))
    
    hyper_params = {'N_hidden':cfg.N_hidden,
                    'S_hidden':cfg.S_hidden,
                    'y_axes':tuple([c.y_ax_map[yc] for yc in cfg.y_cols]),
                    'x_integrated':cfg.x_integrated}
    
    model = NeuralODE(key, **hyper_params)
    
    if cfg.optimizer == "adam":
        if cfg.lr_decay is None:
            optim = optax.adam(cfg.lr)
        else:
            base_optim = optax.scale_by_adam
    elif cfg.optimizer == 'sgd':
        if cfg.lr_decay is None:
            optim = optax.sgd(cfg.lr)
        else:
            base_optim = optax.sgd
    
    if cfg.lr_decay is not None:
        scheduler = optax.exponential_decay(
            init_value=cfg.lr,
            transition_steps=50,
            decay_rate=cfg.lr_decay)
        
        
        optim = optax.chain(
                    base_optim(cfg.lr),
                    optax.scale_by_schedule(scheduler),
                    optax.scale(-1.0)
                )
        

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))



    @eqx.filter_value_and_grad
    def grad_loss(model, Y,t,p):
        y_pred = jax.vmap(model,in_axes=(0, 0, None, 0))(t, X, c.S0, p)
        err = jnp.mean(((Y-y_pred)**2))
        
        return err
    
    @eqx.filter_jit
    def make_step(model, Y, t,p, opt_state):
        loss, grads = grad_loss(model,Y,t,p)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    
    @eqx.filter_jit
    def calc_loss(model, Y,t,p):
        y_pred = jax.vmap(model,in_axes=(0, 0, None, 0))(t, X, c.S0, p)
        err = jnp.mean(((Y-y_pred)**2))
        
        return err,y_pred
    
    @eqx.filter_value_and_grad
    def grad_loss_tscale(model, Y,t,p):
        y_pred = jax.vmap(model,in_axes=(0, 0, None, 0))(t, X, c.S0, p)
        err = jnp.mean(jax.vmap(jnp.multiply,in_axes=(2,None))((Y-y_pred)**2,t/20))
        
        return err
    
    @eqx.filter_jit
    def make_step_tscale(model, Y, t,p, opt_state):
        loss, grads = grad_loss_tscale(model,Y,t,p)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    
    @eqx.filter_jit
    def calc_loss_tscale(model, Y,t,p):
        y_pred = jax.vmap(model,in_axes=(0, 0, None, 0))(t, X, c.S0, p)
        err = jnp.mean(jax.vmap(jnp.multiply,in_axes=(2,None))((Y-y_pred)**2,t/20))
        
        
        return err,y_pred

    best_test_loss = 9999
    best_train_loss = 9999 
    for i in range(cfg.N_epochs):
        
        s0 = time.time()
        
        train_loss = 0
        for X,Y,t,p in trainloader:
            X,Y,t,p = X.numpy(), Y.numpy(),t.numpy(),p.numpy()
            
            start = time.time()
            if cfg.loss_scale:
                loss, model, opt_state = make_step_tscale(model, Y, t,p, opt_state)
            else:
                loss, model, opt_state = make_step(model, Y, t,p, opt_state)
            end = time.time()
            
            print(f"Loss: {loss}, Computation time: {end - start}")
            train_loss += loss
            
        train_loss = train_loss / len(trainloader)
        
        
        test_loss = 0
        y_all = []
        for X,Y,t,p in testloader:
            X,Y,t,p = X.numpy(), Y.numpy(),t.numpy(),p.numpy()
            
            if cfg.loss_scale:
                loss,y_pred = calc_loss_tscale(model, Y,t,p)
            else:
                loss,y_pred = calc_loss(model, Y,t,p)
            
            
            test_loss += loss
            y_all.append(y_pred)
            
        test_loss = test_loss / len(testloader)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_loss = train_loss
        
        s1 = time.time()
        
        if i % 5 == 0:
            save(fmodel, hyper_params, model)
        
        if use_mlflow:
            mlflow.log_metric("train_loss", train_loss,i)
            mlflow.log_metric("test_loss", test_loss,i)
            mlflow.log_metric("train_loss_best", best_train_loss,i)
            mlflow.log_metric("test_loss_best", best_test_loss,i)
            mlflow.log_metric('epoch_time', s1-s0,i)
            mlflow.log_metric("epoch",i,i)
            if i % 5== 0 and test_loss == best_test_loss:
                mlflow.log_artifact(fmodel)
        

        s2 = time.time()
        print(f"Epoch: {i}, test loss:{test_loss}, epoch time: {s1-s0}, save time: {s2-s1} -----------------------")

    if use_mlflow:
        mlflow.end_run()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=23)
    parser.add_argument('--N_hidden', type=int, default=2)
    parser.add_argument("--S_hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument('--N_epochs', type=int, default=25)
    parser.add_argument("--y_cols", nargs='+',type=str, default=['tar_mol', 'char_mol'])
    parser.add_argument("--loss_scale",action='store_true')
    parser.add_argument("--x_integrated",type=bool,default=False)
    parser.add_argument("--no_tracking",action='store_true')
    parser.add_argument("--experiment",type=str, default="biofuels_testing")
    parser.add_argument("--train_size_subset",type=float, default=1.0)
    parser.add_argument("--lr_decay",type=float, default=None)
    parser.add_argument("--seed_shift",type=int,default=0)
    parser.add_argument("--optimizer",default='adam')
    parser.add_argument("--shuffle",action='store_true')
    
    args = parser.parse_args()

    cfg = dotdict(vars(args))
    cfg['SEED'] = SEED
    cfg['model_type'] = 'differentiable'

    train(cfg)