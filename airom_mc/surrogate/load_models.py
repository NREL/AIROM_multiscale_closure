import json
import equinox as eqx
import jax.numpy as jnp
import jax
from airom_mc.surrogate.eqx_modules import NeuralODE
import airom_mc.common.constants as c

def save(filename, hyperparams, model):
    print(f"Saving model: {filename}")
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load(filename,return_params=False):
    with open(filename, "rb") as f:
        hyperparams = {'ds_mean': jnp.array([5,4]), 
                    'ds_std':jnp.array([5,4])}
        add_params = json.loads(f.readline().decode())
        if add_params is not None:
            add_params = {k:v for k,v in add_params.items() if k not in ('ds_mean','ds_std','A_0')}
            hyperparams.update(add_params)
        key =  jax.random.key(2424)
        model = NeuralODE(key,**add_params)
        if return_params:
            return (eqx.tree_deserialise_leaves(f, model), add_params)
        return eqx.tree_deserialise_leaves(f, model)