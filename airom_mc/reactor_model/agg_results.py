import pandas as pd
import os

def agg_results():
    
    res_df = []
    for case in ['base_3d', 'surrogate_3d']:
        
        all_df = []
        df_params = pd.read_csv(f"../data/reactor_params_{case}.csv")
        prefix = f'../data/reactor_models/{case}/'
        
        for run in df_params.index:
            prefix_idx = os.path.join(prefix, f"run_{run:03d}")
            try:
                df_gas = pd.read_csv(os.path.join(prefix_idx, "gas.dat"),delimiter='\t')
                df_biomass = pd.read_csv(os.path.join(prefix_idx, "biomass.dat"),delimiter='\t')
            except FileNotFoundError:
                print(f"Not found, skipping: {prefix_idx}")
                continue
            
            df = pd.merge(df_gas,df_biomass, left_on="t[i]",right_on="t[i]")
            
            df['run'] = run
            df['case'] = case
            
            all_df.append(df)

        df = pd.concat(all_df,ignore_index=True)
        df = pd.merge(df,df_params, left_on='run',right_index=True)
        res_df.append(df)
    
    df = pd.concat(res_df)
    print(df.columns)
    df.to_csv("../data/reactor_results.csv")
    
if __name__ == "__main__":
    agg_results()