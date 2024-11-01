import pandas as pd
import os
from scipy.stats import qmc
import numpy as np

def main():
    fdir = '../data/'

    sampler = qmc.LatinHypercube(d=3, seed=2424)
    sample = sampler.random(n=250)
    
    l_bounds = [0.1,2,723]
    u_bounds = [10,8,823]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    df_ICs = pd.DataFrame(sample_scaled,columns=['FL','aspect','T_oven'])

    # # add the sweeps
    N = 10
    FL_l = np.linspace(l_bounds[0], u_bounds[0],N)
    aspect_l = np.linspace(l_bounds[1],u_bounds[1],N)
    T_oven_l = np.linspace(l_bounds[2],u_bounds[2],N)
    
    FL_c = 4
    aspect_c = 4
    T_oven_c = 773
    
    FL_g, aspect_g = np.meshgrid(FL_l,aspect_l)
    sweep_1 = pd.DataFrame({"FL":FL_g.flatten(),"aspect":aspect_g.flatten(),
               "T_oven":np.ones(N**2)*T_oven_c})
    
    
    FL_g, T_oven_g = np.meshgrid(FL_l,T_oven_l)
    sweep_2 = pd.DataFrame({"FL":FL_g.flatten(),"aspect":np.ones(N**2)*aspect_c,
               "T_oven":T_oven_g.flatten()})
    
    
    aspect_g, T_oven_g = np.meshgrid(aspect_l,T_oven_l)
    sweep_3 = pd.DataFrame({"FL":np.ones(N**2)*FL_c,"aspect":aspect_g.flatten(),
               "T_oven":T_oven_g.flatten()})
    
    
    df_ICs = pd.concat([df_ICs, sweep_1, sweep_2, sweep_3])
    df_ICs = df_ICs.drop_duplicates().reset_index(drop=True)
    
    df_ICs.to_csv(os.path.join(fdir,"comsol_ICs.csv"))



if __name__ == "__main__":
    main()
