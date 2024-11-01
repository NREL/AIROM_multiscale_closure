import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def calc_surface_area(FL,aspect):
    """
    Compute surface area for filleted cylindrical
    particle. Takes FL in mm returns surface area in m^2
    """
    FL = FL*1e-3
    rad = FL/aspect
    
    A = 2*np.pi*(3*rad/4)**2
    B = (4*np.pi**2*(3*rad/4)*(rad/4))/4
    C = 2*np.pi*rad*(FL/2-rad/4)
    
    surface_area =2*(A+B+C)
    
    return surface_area

def calc_volume(FL,aspect):
    """
    Compute volume area for filleted cylindrical
    particle. Takes FL in mm returns volume in m^3
    """
    FL = FL*1e-3
    rad = FL/aspect
    
    A = np.pi*(3*rad/4)**2*rad/4
    B = (2*np.pi**2*(rad/4)**2*(3*rad/4))/4
    C = np.pi*rad**2*(FL/2-rad/4)
    
    volume = 2*(A+B+C)
    
    return volume

