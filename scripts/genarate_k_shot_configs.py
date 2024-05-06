from typing import List, Optional, Union
import yaml
import numpy as np
from pathlib import Path
import shutil

# config_path = Path('/home/kabird/lfads-torch-fewshot-benchmark/configs/callbacks/fewshot10and50.yaml')
# config_path = Path('/home/kabird/lfads-torch-fewshot-benchmark/configs/callbacks/sklearn_fewshot20and50and100.yaml')
config_path = Path('/home/kabird/lfads-torch-fewshot-benchmark/configs/callbacks/fewshot10and20and50and100and500and1000.yaml')

# k_list = [10] * 20 + [50] * 5
k_list = [10] * 20 + [20] * 10 + [50] * 5 + [100]*2 + [500]*2 + [1000]
seed_list = list(np.arange(len(k_list),dtype=int))
k_seed_list = list(zip(k_list,seed_list))

base_config_single = {
    'fewshot_head_model'  : '${fewshot_head_model_lightning}',
    'fewshot_trainer'     : '${fewshot_trainer}',
    'K'                   : 500,
    'seed'                : 0,
    'log_every_n_epochs'  : 1,
    'fewshot_trainer_epochs': 150,
    'use_recon_as_targets' : True,
    'eval_type'            : 'post_run'
}

full_config = {
    f'fewshot_k{k}_seed{seed}':{
        **base_config_single,
        'K':int(k),
        'seed':int(seed), 
        # 'fewshot_head_model':'${fewshot_head_model_sklearn}'
    } for k,seed in k_seed_list
}


# print(k_seed_list)
# print(full_config[f'fewshot_k{10}_seed{0}'])
yaml_string = yaml.dump(full_config,default_flow_style=False)
yaml_string =  '\n'.join(["defaults:"]+[f"  - fewshot_callbacks@fewshot_k{k}_seed{seed}: fewshot_base" for (k,seed) in k_seed_list]+['  - _self_','',yaml_string])  
with open(config_path,'w+') as f:
    f.write(yaml_string)