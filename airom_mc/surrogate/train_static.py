
import jax.numpy as jnp
import jax
import optax
import time
import mlflow
import argparse
from torch.utils.data import DataLoader
import equinox as eqx
import os

from biofuels_particle_surrogate.surrogate.dataloaders import CSVData
from biofuels_particle_surrogate.surrogate.eqx_modules import NeuralODE
from biofuels_particle_surrogate.common.utils import dotdict
import biofuels_particle_surrogate.common.constants as c
from biofuels_particle_surrogate.common.global_cfg import MLFLOW_URI, SEED
from biofuels_particle_surrogate.surrogate.jax_utils import train_loop_static, test_loop_static
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
        fmodel = f"../data/surrogate/model_static_{run.info.run_id}.eqx"
    else:
        fmodel = f"../data/surrogate/model_static_testing.eqx"
    
    key =  jax.random.key(cfg.SEED+cfg.seed_shift)
    

    ds_train = CSVData('../data', subset='train',y_cols=cfg.y_cols,seed=SEED,
                 y_units='mol/mol', x_integrated=cfg.x_integrated,include_static_opt=True,
                 train_size_subset=cfg.train_size_subset, tscale_loss=cfg.loss_scale,
                 seed_add=cfg.seed_shift)
    ds_test = CSVData('../data', subset='test',y_cols=cfg.y_cols,seed=SEED,
                 y_units='mol/mol', x_integrated=cfg.x_integrated,include_static_opt=True, tscale_loss=cfg.loss_scale)
    mlflow.log_params({"train_ds_size":len(ds_train)})
    mlflow.log_params({"slurm_job_id":os.getenv("SLURM_JOB_ID")})
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)
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
            scheduler = optax.exponential_decay(
                init_value=cfg.lr,
                transition_steps=50,
                decay_rate=cfg.lr_decay)
        
        
            optim = optax.chain(
                        base_optim(),
                        optax.scale_by_schedule(scheduler),
                        optax.scale(-1.0)
                    )
    elif cfg.optimizer == 'sgd':
        if cfg.lr_decay is None:
            optim = optax.sgd(cfg.lr)
        else:
            base_optim = optax.sgd(cfg.lr)
            scheduler = optax.exponential_decay(
                init_value=cfg.lr,
                transition_steps=50,
                decay_rate=cfg.lr_decay)
        
            optim = optax.chain(optax.scale_by_schedule(scheduler),  # Apply the learning rate schedule
                                base_optim  # SGD optimizer
                            )

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


    @eqx.filter_jit
    def calc_loss_tscale(model, Y,t,p):
        y_pred = jax.vmap(model,in_axes=(0, 0, None, 0))(t, X, c.S0, p)
        err = jnp.mean(((Y-y_pred)**2)*jnp.expand_dims(t,axis=-1)/20)
        
        return err
    
    @eqx.filter_jit
    def calc_loss(model, Y,t,p):
        y_pred = jax.vmap(model,in_axes=(0, 0, None, 0))(t, X, c.S0, p)
        err = jnp.mean(((Y-y_pred)**2))
        
        return err
    
    @eqx.filter_jit
    def calc_static_loss(pmodel, X, EA_opt):
        EA_pred = jax.vmap(pmodel.parameter)(X)
        err = jnp.mean(((EA_opt-EA_pred)**2))
        
        return err
    
    
    @eqx.filter_value_and_grad
    def grad_loss(pmodel, X, EA_opt):
        EA_pred = jax.vmap(pmodel.parameter)(X)
        err = jnp.mean(((EA_opt-EA_pred)**2))
        
        return err
    
    @eqx.filter_jit
    def make_step(model, X, EA_opt, opt_state):
        loss, grads = grad_loss(model,X, EA_opt)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    best_test_loss = 9999
    best_train_loss = 9999 
    for i in range(cfg.N_epochs):
        
        s0 = time.time()
        #model, train_loss, train_loss_static, opt_state = train_loop_static(model, dl_train, optim, opt_state, tscale_loss=cfg.loss_scale)
        
        avg_static_loss = 0
        avg_NDE_loss = 0
        for X,Y,t,p,EA in dl_train:
            #print('train',i)
            X,Y,t,p,EA = X.numpy(), Y.numpy(),t.numpy(),p.numpy(), EA.numpy()
            
            start = time.time()
            
            loss_static, model, opt_state = make_step(model, X, EA, opt_state)
            mid = time.time()
            if cfg.tscale_loss:
                loss_NDE = calc_loss_tscale(model, Y,t,p)
            else:
                loss_NDE = calc_loss(model, Y,t,p)
            end = time.time()
            
            
            print(f"Train loop Loss Static: {loss_static}, Computation time: {end - start}")
            avg_static_loss += loss_static
            avg_NDE_loss += loss_NDE
            
        train_loss = avg_NDE_loss / len(dl_train)
        train_loss_static = avg_static_loss / len(dl_train)

        # _, test_loss, test_loss_static = test_loop_static(model, dl_test,tscale_loss=cfg.loss_scale)
        
        
        avg_static_loss = 0
        avg_NDE_loss = 0
        for X,Y,t,p,EA in dl_test:
            X,Y,t,p,EA = X.numpy(), Y.numpy(),t.numpy(),p.numpy(), EA.numpy()
            
            start = time.time()
            loss_static = calc_static_loss(model, X, EA)
            if cfg.tscale_loss:
                loss_NDE = calc_loss_tscale(model, Y,t,p)
            else:
                loss_NDE = calc_loss(model, Y,t,p)
            end = time.time()
            
            #print(f"Loss SSStatic: {loss_static}, Computation time: {end - start}")
            avg_static_loss += loss_static
            avg_NDE_loss += loss_NDE
            
        test_loss = avg_NDE_loss / len(dl_test)
        test_loss_static = avg_static_loss / len(dl_test)
  
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_loss = train_loss
            best_test_loss_static = test_loss_static
            best_train_loss_static = train_loss_static
        
        s1 = time.time()
        
        if i % 5 == 0:
            save(fmodel, hyper_params, model)
        
        if use_mlflow:
            mlflow.log_metric("train_loss", train_loss,i)
            mlflow.log_metric("test_loss", test_loss,i)
            mlflow.log_metric("train_loss_static", train_loss_static,i)
            mlflow.log_metric("test_loss_static", test_loss_static,i)
            mlflow.log_metric("train_loss_best", best_train_loss,i)
            mlflow.log_metric("test_loss_best", best_test_loss,i)
            mlflow.log_metric("train_loss_static_best", best_train_loss_static,i)
            mlflow.log_metric("test_loss_static_best", best_test_loss_static,i)
            mlflow.log_metric('epoch_time', s1-s0,i)
            mlflow.log_metric("epoch",i,i)
            if i % 5== 0 and test_loss == best_test_loss:
                mlflow.log_artifact(fmodel)
        
            
        print(f"Epoch: {i},test loss static:{test_loss_static},train loss static:{train_loss_static},  test loss:{test_loss},train loss:{train_loss}, epoch time: {s1-s0} -----------------------")
        
    if use_mlflow:
        mlflow.end_run()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=23)
    parser.add_argument('--N_hidden', type=int, default=2)
    parser.add_argument("--S_hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument('--N_epochs', type=int, default=25)
    parser.add_argument("--y_cols", nargs='+',type=str, default=['tar_mol','char_mol'])
    parser.add_argument("--loss_scale",action='store_true')
    parser.add_argument("--no_tracking",action='store_true')
    parser.add_argument("--experiment",type=str, default="biofuels_testing")
    parser.add_argument("--train_size_subset",type=float, default=1.0)
    parser.add_argument("--seed_shift",type=int,default=0)
    parser.add_argument("--lr_decay",type=float, default=None)
    parser.add_argument("--optimizer",default='adam')
    args = parser.parse_args()

    cfg = dotdict(vars(args))
    cfg['SEED'] = SEED
    cfg['model_type'] = 'static'

    train(cfg)