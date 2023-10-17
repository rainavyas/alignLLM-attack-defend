from typing import List, Dict
import pandas as pd


def load_data(data_name:str, lim:int=None):
    data_ret = {
        'advbench'  :   _load_advbench
    }
    return data_ret[data_name](lim)

def _load_advbench(lim:int=None)->List[Dict['prompt', 'target']]:
    data_path = 'src/data/advbench.csv'
    df = pd.read_csv(f'{data_path}')
    df = df.rename(columns={'goal':'prompt', 'target':'adv_target'})
    return df.to_dict('records')[:lim]