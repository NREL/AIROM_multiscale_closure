import numpy as np
import pandas as pd
import os
from pathlib import Path
import biofuels_particle_surrogate.common.constants as c
from biofuels_particle_surrogate.surrogate.load_models import load
import jax.numpy as jnp

def modify_files(params, out_dir):
    
    rx_file = os.path.join(out_dir, "constant/reactions.particles")
    # Read in the file
    with open(rx_file, 'r') as file:
        filedata = file.read()

    # Replace the target string
    for i in range(10):
        Ea_i = params[f'Ea_{i}']
        filedata = filedata.replace(f'Ta{i}', f'{Ea_i/c.R}')

    # Write the file out again
    with open(rx_file, 'w') as file:
        file.write(filedata)

    p_file = os.path.join(out_dir, "constant/globalVars")
    
    # Read in the file
    with open(p_file, 'r') as file:
        filedata = file.read()

    FL_idx = params['FL']*1e-3
    filedata = filedata.replace(f'partSize   5.2e-4;', f'partSize   {FL_idx};')

    # Write the file out again
    with open(p_file, 'w') as file:
        file.write(filedata)

def setup_reactor_models(case='base'):
    # Setup simulation FL case
    FL = np.array([1,2,5,8,10])
    A = 4.0
    
    dat = np.array([FL,A*np.ones_like(FL)])
    
    df_params = pd.DataFrame({"FL":FL,
                              "aspect":np.ones_like(FL)*A})
    
    
    if case in ("surrogate",'surrogate_3d'):
        model = load('../data/surrogate/model.eqx')
    
    prefix = "../data/reactor_models/"
    if '3d' in case:
        ex_dir = os.path.join(prefix,"example_3d")
    else:
        ex_dir = os.path.join(prefix,"example")
    
    Path(os.path.join(prefix,case)).mkdir(exist_ok=True)
    
    for idx in df_params.index:
        
        if case in ('surrogate', 'surrogate_3d'):
            # calculate new Ea from params
            Ea = model.parameter(jnp.array(dat[:,idx]))*10
            df_params.loc[idx, [f'Ea_{i}' for i in range(10)]] = Ea
        elif case in ('base', 'base_3d'):
            Ea = c.EA_0
            df_params.loc[idx, [f'Ea_{i}' for i in range(10)]] = Ea
        
        # copy base case - base
        out_dir = os.path.join(prefix,case,f"run_{idx:03d}")
        print(f"Copying: {ex_dir} to {out_dir}")
        os.system(f"cp -r {ex_dir} {out_dir}")

        modify_files(df_params.loc[idx], out_dir)        

    df_params.to_csv(f'../data/reactor_params_{case}.csv')
    
    

if __name__ == "__main__":
    setup_reactor_models("surrogate")
    setup_reactor_models("base")
