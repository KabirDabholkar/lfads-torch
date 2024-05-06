from nlb_tools.fewshot_utils import result_dict_to_pandas
from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import save_to_h5, make_train_input_tensors, h5_to_dict    

import torch
import numpy as np
from lfads_torch.tuples import SessionBatch

from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor

from os import path as osp
import os
import h5py, json
from pathlib import Path

data_path_base = '/home/kabird/datasets'
DATA_DIR = Path(f"{data_path_base}/")

def fit_poisson_parallel(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0):
    """Fit Poisson GLM from factors to spikes and return rate predictions"""

    pr = MultiOutputRegressor(
        estimator=PoissonRegressor(alpha=alpha, max_iter=500),
        n_jobs=-1
    )
    

    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    
    pr.fit(train_in, train_out)
    train_rates_s = pr.predict(train_factors_s)
    eval_rates_s = pr.predict(eval_factors_s)
    return train_rates_s, eval_rates_s


def run_model_on_numpy(model,spikes_heldin,batch_size=8):
    batches = []
    for encod_data in np.array_split(
        spikes_heldin,
        int(np.maximum(spikes_heldin.shape[0]//batch_size,1)),
        axis=0):
        encod_data_torch = torch.tensor(encod_data).to(model.device)
        n_samps, n_steps, _ = encod_data.shape
        ext_input = torch.zeros(n_samps, n_steps, 0).to(model.device)
        batch = SessionBatch(**{
            "encod_data": encod_data_torch,
            "recon_data": None,  # Fill with appropriate data if available
            "ext_input": ext_input,   # Fill with appropriate data if available
            "truth": None,       # Fill with appropriate data if available
            "sv_mask": None,     # Fill with appropriate data if available
        })
        batches.append(batch)
    outputs = [model(batch) for batch in batches]
    output_params = np.concatenate([o[0].output_params.detach().cpu().numpy() for o in outputs])
    factors = np.concatenate([o[0].factors.detach().cpu().numpy() for o in outputs])
    return output_params,factors

def run_nlb_fewshot(model,variant='dmfc_rsg'):
    target_path = osp.join(DATA_DIR, f"{variant}_target.h5")

    with h5py.File(target_path, 'r') as h5file:
        target_dict = h5_to_dict(h5file)

    
    train_path = osp.join(DATA_DIR, f"{variant}_train.h5")
    with h5py.File(train_path, 'r') as h5file:
        train_dict = h5few_shot_metadata = h5_to_dict(h5file)
    
    val_path = osp.join(DATA_DIR, f"{variant}_val.h5")
    with h5py.File(val_path, 'r') as h5file:
        val_dict = h5_to_dict(h5file)

    train_path_json = osp.join(DATA_DIR, f"{variant}_train.json")
    with open(train_path_json, 'r') as file:
        few_shot_metadata = json.load(file)

    
    # print(few_shot_metadata)

    # model()
    batch_size = 8

    eval_spikes_heldin = val_dict['eval_spikes_heldin']
    eval_spikes_heldout = val_dict['eval_spikes_heldout']
    
    eval_pred, eval_latents = run_model_on_numpy(model,eval_spikes_heldin)
    eval_latents = eval_latents[:,:eval_spikes_heldin.shape[1]]
    eval_latents_s = eval_latents.reshape(-1,eval_latents.shape[-1])
    

    eval_rates = eval_pred
    spikes = eval_spikes_heldin
    heldout_spikes = eval_spikes_heldout

    eval_rates, eval_rates_forward = np.split(eval_rates, [spikes.shape[1]], axis=1)
    eval_rates_heldin_forward, eval_rates_heldout_forward = np.split(eval_rates_forward, [spikes.shape[-1]], axis=-1)
    # train_rates, _ = np.split(train_rates, [spikes.shape[1], train_rates.shape[1] - spikes.shape[1]], axis=1)
    eval_rates_heldin, eval_rates_heldout = np.split(eval_rates, [spikes.shape[-1]], axis=-1)
    # train_rates_heldin, train_rates_heldout = np.split(train_rates, [spikes.shape[-1], heldout_spikes.shape[-1]], axis=-1)


    output_dict = {
        variant: {
            # 'train_rates_heldin': train_rates_heldin.cpu().numpy(),
            # 'train_rates_heldout': train_rates_heldout.cpu().numpy(),
            'eval_rates_heldin': eval_rates_heldin,
            'eval_rates_heldout': eval_rates_heldout,
            'eval_rates_heldin_forward': eval_rates_heldin_forward,
            'eval_rates_heldout_forward': eval_rates_heldout_forward
        }
    }



    fewshot_output_dict = {}
    k_range = few_shot_metadata["Kvalues_applicable"] #2**np.arange(4,11)[:1].astype(int)
    # k_range = [int(k) for k in k_range]
    for k in k_range[3:]:
        for shot_id in few_shot_metadata[f'{k}shot_ids']:
            fewshot_train_spikes_heldin = train_dict[f'{k}shot_id{shot_id}_train_spikes_heldin']
            fewshot_train_spikes_heldout = train_dict[f'{k}shot_id{shot_id}_train_spikes_heldout']
            fewshot_train_spikes_heldout_s = fewshot_train_spikes_heldout.reshape(-1,fewshot_train_spikes_heldout.shape[-1])
            
            fewshot_train_pred, fewshot_train_latents = run_model_on_numpy(model,fewshot_train_spikes_heldin)
            fewshot_train_latents = fewshot_train_latents[:,:fewshot_train_spikes_heldin.shape[1]]
            fewshot_train_latents_s = fewshot_train_latents.reshape(-1,fewshot_train_latents.shape[-1])
            fewshot_train_rates_s, eval_rates_s = fit_poisson_parallel(fewshot_train_latents_s,eval_latents_s,fewshot_train_spikes_heldout_s)
            eval_rates = eval_rates_s.reshape(*heldout_spikes.shape[:2],-1)
            fewshot_output_dict [f'{k}shot_id{shot_id}_eval_rates_heldout'] = eval_rates

    output_dict[variant] = {
                    **output_dict[variant],
                    **fewshot_output_dict
                }
    
    fewshot_code_name = 'sklearn_parallel'

    result_data = evaluate(target_dict, output_dict)
    print('result_dict',result_data)
    df = result_dict_to_pandas(
        result_data,
        fewshot_learner=fewshot_code_name,
        # path=ckpt_path
    )

    # eval_report.append(df)
    
    # D = result_data.reset_index()
    # D.to_csv()
    return df

if __name__=="__main__":
    run_nlb_fewshot("")