import pandas as pd
import glob
import os

def main():
    ddir = '../data/'
    outpath = os.path.join(ddir,f"particle_models/")
    df_ICs = pd.read_csv(os.path.join(ddir,"comsol_ICs.csv"))
    df_ICs['sim'] = df_ICs.index.values
    all_df = []

    print(df_ICs)
    print('--------------')
    for idx in df_ICs.index:    
        try:
            df = pd.read_csv(os.path.join(outpath,f"model_eval_{idx:03d}.csv"))
        except:
            continue
        df['sim'] = idx
        all_df.append(df)
        
    df = pd.concat(all_df)
    
    df = pd.merge(df,df_ICs, left_on='sim',right_on='sim',how='inner')
    
    df = df.rename(columns={
                    "CHAR":"char_mol",
                    "TAR":"tar_mol",
                    "SG":"sg_mol",
                    "CELL":"cell_mol",
                    "ACELL":"acell_mol",
                    "HCELL":"hcell_mol",
                    "AHCELL":"ahcell_mol",
                    "LIG":"lig_mol",
                    "ALIG":"alig_mol"
                })
    
    df.to_csv(os.path.join(ddir,'particle_eval_agg.csv'))
    
if __name__ =="__main__":
    main()
    
