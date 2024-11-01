import argparse
import pandas as pd
import os
import glob
import numpy as np
import mph
from scipy import integrate


def main(idx):
    ddir = '../data/'    
    outpath = os.path.join(ddir,f"particle_models/")

    client = mph.start()
    
    model = client.load(os.path.join(outpath,f'model_{idx:03d}.mph'))
    
    evals = model/'evaluations'
    eval1 = evals/'Surface Average 1'
    eval2 = evals/'Surface Integration 1'
    
    eval1.property('data','dset1')
    eval1.property('expr',['t','T'])
    eval1.property('unit',['s','K'])
    
    eval2.property('data','dset1')
    eval2.property('expr',['RCHAR',"RTAR", "RCELL","RACELL",
                           "RHCELL","RAHCELL","RLIG","RALIG","RSG","CELL_0","LIG_0","HCELL_0"])
    eval2.property('unit',["mol/s","mol/s","mol/s","mol/s","mol/s",
                           "mol/s","mol/s","mol/s","mol/s","mol","mol","mol"])
    
    r1 = np.array([eval1.java.computeResult()])
    r2 = np.array([eval2.java.computeResult()])
    
    df1 = pd.DataFrame(r1[0,0],columns=['t','T_avg'])
    
    
    df2 = pd.DataFrame(r2[0,0],columns=['RCHAR',"RTAR", "RCELL","RACELL",
                           "RHCELL","RAHCELL","RLIG","RALIG","RSG","CELL_0","LIG_0","HCELL_0"])
    
    df = pd.merge(df1,df2,left_index=True,right_index=True)
    rates =  ['RCHAR',"RTAR", "RCELL","RACELL","RHCELL","RAHCELL","RLIG","RALIG","RSG",]
    for rate in rates:
        if rate == "RCELL":
            init = df.iloc[0]['CELL_0']
        elif rate == "RHCELL":
            init = df.iloc[0]['HCELL_0']
        elif rate == "RLIG":
            init = df.iloc[0]['LIG_0']
        else:
            init = 0
            
        df[rate[1:]] = np.r_[init,integrate.cumulative_trapezoid(df[rate], df.t)+init]
        
    
    
    df.to_csv(os.path.join(outpath,f'model_eval_{idx:03d}.csv'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('idx')
    args = parser.parse_args()

    main(int(args.idx))