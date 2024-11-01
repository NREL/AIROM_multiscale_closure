import equinox as eqx  
import jax.numpy as jnp
import jax
import diffrax
import biofuels_particle_surrogate.common.constants as c
from biofuels_particle_surrogate.common.utils import calc_surface_area, calc_volume

class ParameterizationNN(eqx.Module):
    layers: list
    x_integrated: bool
    
    def __init__(self,key, N_hidden, S_hidden, x_integrated):
        key1, key2,key3,key4,key5 = jax.random.split(key, 5)
        layer_0 = eqx.nn.Linear(2, S_hidden, key=key1)
        activation = jax.nn.sigmoid
        final_layer = eqx.nn.Linear(S_hidden, 10, key=key2)
 
        add_keys = [key3,key4,key5]
        add_layers = []
        for i in range(1,N_hidden):
            add_layers.append(eqx.nn.Linear(S_hidden, S_hidden, 
                                            key=add_keys[i-1]))
            add_layers.append(activation)
 
        new_bias = jnp.array(c.EA_0/10)
        where = lambda l: l.bias
        final_layer = eqx.tree_at(where, final_layer, new_bias)

        self.layers = [layer_0,activation]+add_layers+[final_layer]
        self.x_integrated = x_integrated
        
    def __call__(self,x):
        if self.x_integrated:
            x = (x -  c.ds_mean_integrated)/(c.ds_std_integrated)
        else:
            x = (x -  c.ds_mean)/(c.ds_std)
            
        for layer in self.layers:
            x = layer(x)
        return x

class ChemistryODE(eqx.Module):

    def dS_dt(self,s,Ea, T_oven, surface_area, volume):
        def arrhenius(A, Ea, T):
            return A*jnp.exp(-1*Ea/(c.R*T))
        
        k = arrhenius(c.A_0, Ea, s[...,[0]])

        dS = jnp.zeros_like(s)
      
        Q_in = surface_area*c.h*(T_oven-s[...,0])
        dT = Q_in/(volume*c.rho*c.cp)

        # mass
        dS1 = -1*k[...,0]*s[...,1]
        dS2 = -1*k[...,3]*s[...,2]
        dS3 = -1*k[...,6]*s[...,3]
        dS4 = k[...,0]*s[...,1] - (k[...,1] + k[...,2])*s[...,4]
        dS5 = k[...,3]*s[...,2] - (k[...,4] + k[...,5])*s[...,5]
        dS6 = k[...,6]*s[...,3] - (k[...,7] + k[...,8])*s[...,6]
        
        dS7 = k[...,1]*s[...,4] + k[...,4]*s[...,5] + \
                k[...,7]*s[...,6] - k[...,9]*s[...,7] 
        dS8 = (1-c.Y_c)* k[...,2]*s[...,4] + \
                    (1 - c.Y_h)*k[...,5]*s[...,5] + \
                    (1-c.Y_l)*k[...,8]*s[...,6] + k[...,9]*s[...,7]
        dS9 = c.Y_c* k[...,2]*s[...,4] + \
                    c.Y_h*k[...,5]*s[...,5] + \
                    c.Y_l*k[...,8]*s[...,6]
                    
        dS = jnp.stack([dT, dS1,dS2,dS3,dS4,dS5,dS6,dS7,dS8,dS9])
        return dS

    def __call__(self, t, s, args):
        EA, T_oven, surface_area, volume = args

        dS = self.dS_dt(s, EA, T_oven, surface_area, volume)
        return dS
    
    

    

class NeuralODE(eqx.Module):
    integrator: ChemistryODE
    parameter: ParameterizationNN
    y_axes: tuple = eqx.static_field()
    x_integrated: bool
    

    def __init__(self, key, N_hidden,S_hidden,
                 y_axes,x_integrated, **kwargs):
        super().__init__(**kwargs)
        self.integrator = ChemistryODE()
        self.parameter = ParameterizationNN(key, N_hidden,S_hidden,x_integrated)
        self.y_axes = y_axes
        self.x_integrated = x_integrated
        #self.x_cols = x_cols



    def __call__(self, ts, X, S0, T_oven):
        EA = self.parameter(X)*10
        
        if self.x_integrated:
            surface_area = X[0]
            volume = X[1]
        else:
            surface_area = calc_surface_area(X[0],X[1])
            volume = calc_volume(X[0],X[1])

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.integrator),
            diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=1e-4,
            y0=S0,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
            saveat=diffrax.SaveAt(ts=ts),
            args=(EA, T_oven, surface_area, volume)
        )
        return solution.ys[:,self.y_axes]/c.m0.sum()
    