import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import scipy
import pandas as pd
import pickle as pkl
from multiprocessing import Pool
import numpy as np
from itertools import product
from tqdm import tqdm
from sklearn.decomposition import PCA
import scipy
from typing import Optional
# from hydra.utils import instantiate
from functools import partial
from hydra.utils import instantiate
import seaborn as sns
import torch

from copy import deepcopy

import matplotlib as mpl


mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


all_models_configs = [
    # {
    #     'dataset' : 'nlb_mc_maze',
    #     'num_runs' : 20,
    #     'architecture' : 'lfads-torch',
    #     'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240318_144734_MultiFewshot',
    # },
    {
        'dataset' : 'nlb_mc_maze',
        'num_runs' : 200,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240319_085230_MultiFewshot',
    },
    # {
    #     'dataset' : 'nlb_mc_rtt',
    #     'num_runs' : 20,
    #     'architecture' : 'lfads-torch',
    #     'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240329_201611_MultiFewshot',
    # },
    {
        'dataset' : 'nlb_mc_rtt',
        'num_runs' : 200,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240331_130041_MultiFewshot',
    },
    {
        'dataset' : 'nlb_dmfc_rsg',
        'num_runs' : 20,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240402_110106_MultiFewshot',
    },
    {
        'dataset' : 'nlb_dmfc_rsg',
        'num_runs' : 200,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240404_111929_MultiFewshot',
    },
]

def load_model_datas(path_to_models):
    all_files = os.listdir(path_to_models)

    model_files = [f for f in all_files if f.startswith('run_model')]
    # print(model_files)
    model_datas = []
    for f in model_files:
        full_path = os.path.join(path_to_models, f,'csv_logs','metrics.csv')
        # print(full_path,os.path.exists(full_path))
        if os.path.exists(full_path):
            model_data = pd.read_csv(full_path,index_col=0) 
            model_data = model_data.iloc[0:1]
            model_data['model_id'] = f
            model_datas.append(model_data)
    model_datas_ = pd.concat(model_datas,axis=0).reset_index()
    
    # print(model_datas_)
    csv_path = os.path.join(path_to_models, 'concat_model_data.csv')
    model_datas_.to_csv(csv_path)
    return model_datas_


def main():
    all_model_data = []
    for conf in all_models_configs:
        model_data = load_model_datas(conf['path_to_models'])
        model_data['dataset'] = [conf['dataset']]*model_data.shape[0]
        model_data['path_to_models'] = [conf['path_to_models']]*model_data.shape[0]
        all_model_data.append(model_data)

    all_model_data = pd.concat(all_model_data,axis=0)
    
    # fig, ax = plt.subplots()
    
    # sns.regplot(
    #     x='valid/co_bps',
    #     # y='valid/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps',
    #     y='post_run/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps',
    #     data=all_model_data,
    #     ax=ax,
    # )
    # ax.set_ylabel(r'$k=100$-shot co_bps to held out')
    # ax.set_xlabel('co_bps')
    # ax.set_ylim(0.0,0.4)
    # ax.set_xlim(0.2,0.4)
    # r = scipy.stats.pearsonr(x=all_model_data['valid/co_bps'], 
    #                          y=all_model_data['post_run/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps'])[0]
    # ax.plot([0,1],[0,1],ls='dashed',c='black')
    # ax.set_title('Pearsonr:{:1.2f}'.format(r))

    g = sns.FacetGrid(all_model_data, col="dataset", hue='path_to_models')
    g.map(sns.scatterplot, 'valid/co_bps','post_run/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps', alpha=0.4)
    g.add_legend()
    fig = g.figure
    
    fig.savefig(os.path.join('/home/kabird/lfads-torch-fewshot-benchmark/plots', 'co_bps_vs_100shot_heldout.png'), dpi=250)

    

if __name__ == '__main__':
    main()