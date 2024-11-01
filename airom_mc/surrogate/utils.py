import pandas as pd
import torch
import jax
import scipy
import numpy as np

from biofuels_particle_surrogate.common.utils import calc_surface_area, calc_volume
import biofuels_particle_surrogate.common.constants as c

def setup_df(X,Y,t,p,sim_idxs, name=None, select=None):
    dfs = []
    for i in range(Y.shape[0]):
        df = pd.DataFrame({'tar_yield':Y[i,:,0],
                           'char_yield':Y[i,:,1]})
        df['T_oven'] = p[i]
        df['FL'] = X[i,0]
        df['aspect'] = X[i,1]
        df['sim'] = sim_idxs[i]
        df['t'] = t[i]
        df['model'] = name
        dfs.append(df)
        
    return pd.concat(dfs)


def solve_model_jax(model, X, t,p):
    if type(X) == torch.Tensor:
        X = X.numpy()
        t = t.numpy()
        p = p.numpy()

    y_pred = jax.vmap(model,in_axes=(0, 0, None, 0))(t, X, c.S0, p)
    return y_pred


def arrhenius(A, Ea, T):
    return A*np.exp(-1*Ea/(c.R*T))
    

def dS_dt(t, S,EA, A, surface_area, volume, T_oven):
    T = S[0]
    m = S[1:]
    
    dm = np.zeros(9)

    k_1c = arrhenius(A[0], EA[0], T)
    k_2c = arrhenius(A[1], EA[1], T)
    k_3c = arrhenius(A[2], EA[2], T)
    k_1h = arrhenius(A[3], EA[3], T)
    k_2h = arrhenius(A[4], EA[4], T)
    k_3h = arrhenius(A[5], EA[5], T)
    k_1l = arrhenius(A[6], EA[6], T)
    k_2l = arrhenius(A[7], EA[7], T)
    k_3l = arrhenius(A[8], EA[8], T)
    k_4 = arrhenius(A[9], EA[9], T)
    
    
    # mass
    dm[0] = -1*k_1c*m[0]
    dm[1] = -1*k_1h*m[1]
    dm[2] = -1*k_1l*m[2]
    dm[3] = k_1c*m[0] - (k_2c + k_3c)*m[3]
    dm[4] = k_1h*m[1] - (k_2h + k_3h)*m[4]
    dm[5] = k_1l*m[2] - (k_2l + k_3l)*m[5]
    
    dm[6] = k_2c*m[3] + k_2h*m[4] + k_2l*m[5] - k_4*m[6] 
    dm[7] = (1-c.Y_c)*k_3c*m[3] + (1 - c.Y_h)*k_3h*m[4] + (1-c.Y_l)*k_3l*m[5] + k_4*m[6]
    dm[8] = c.Y_c*k_3c*m[3] + c.Y_h*k_3h*m[4] + c.Y_l*k_3l*m[5]
    

    # factor of 1e3 because surface area volume in mm -> m
    Q_in = surface_area*c.h*(T_oven-T)
    dT = Q_in/(volume*c.rho*c.cp)

    dS = np.zeros(10)
    dS[0] = dT
    dS[1:] = dm

    return dS




def solve_model_scipy(X,p,t, y_cols,x_integrated, y_units,Ea=None):

    if Ea is None:
        Ea = c.EA_0
    
    t_eval = t[0].numpy()

    
    if x_integrated:
        surface_area = X[:,0]
        volume = X[:,1]
    else:
        surface_area = calc_surface_area(X[:,0],X[:,1])
        volume = calc_volume(X[:,0],X[:,1])
    
        
    sol = []
    for i in range(X.shape[0]):

        if Ea.ndim== 1:
            sol_i = scipy.integrate.solve_ivp(dS_dt, t_span=(0,20.001), y0=c.S0,
                                    args=(Ea, c.A_0, surface_area[i], volume[i], p[i]),
                                    t_eval=t_eval,method='BDF',atol=1e-8,rtol=1e-8)
        elif Ea.ndim == 2:
            sol_i = scipy.integrate.solve_ivp(dS_dt, t_span=(0,20.001), y0=c.S0,
                            args=(Ea[i], c.A_0, surface_area[i], volume[i], p[i]),
                            t_eval=t_eval,method='BDF',atol=1e-8,rtol=1e-8)
        sol.append(sol_i.y.T)
        
    y_axes = [c.y_ax_map[yc] for yc in y_cols]
    ypred = np.array(sol)[...,y_axes]
    
    
    if y_units == 'mol/mol':
        ypred = ypred/(c.m0.sum())
    else:
        raise(ValueError)
    
    return ypred

def calc_err(y,yp,t=None):
    if t is None:
        return np.mean((y-yp)**2)
    else:
        return np.mean((y-yp)**2*t[...,np.newaxis]/20)