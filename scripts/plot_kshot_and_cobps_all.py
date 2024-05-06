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
    {
        'dataset' : 'nlb_mc_maze',
        'num_runs' : 200,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240407_002905_MultiFewshot',
    },
    {
        'dataset' : 'nlb_mc_rtt',
        'num_runs' : 200,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240331_130041_MultiFewshot',
    },
    {
        'dataset' : 'nlb_mc_rtt',
        'num_runs' : 200,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240404_153551_MultiFewshot',
        
    },
    {
        'dataset' : 'nlb_mc_rtt',
        'num_runs' : 200,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_010359_MultiFewshot',
        
    },
    {
        'dataset' : 'nlb_mc_rtt',
        'num_runs' : 1,
        'architecture' : 'lfads-torch',
        # 'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_025110_MultiFewshot',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_124609_MultiFewshot',
        
    },
    {
        'dataset' : 'nlb_mc_rtt',
        'num_runs' : 1,
        'architecture' : 'lfads-torch',
        # 'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_025445_MultiFewshot',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_124746_MultiFewshot',
    },
    {
        'dataset' : 'nlb_mc_rtt',
        'num_runs' : 1,
        'architecture' : 'lfads-torch',
        # 'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_031342_MultiFewshot',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_135825_MultiFewshot',
    },
    {
        'dataset' : 'nlb_mc_rtt',
        'num_runs' : 1,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_135216_MultiFewshot',
    },
    {
        'dataset' : 'nlb_dmfc_rsg',
        'num_runs' : 20,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240406_030151_MultiFewshot',
    },
    # {
    #     'dataset' : 'nlb_dmfc_rsg',
    #     'num_runs' : 200,
    #     'architecture' : 'lfads-torch',
    #     'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240404_111929_MultiFewshot',
    # },
    {
        'dataset' : 'nlb_dmfc_rsg',
        'num_runs' : 200,
        'architecture' : 'lfads-torch',
        'path_to_models' : '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240405_204157_MultiFewshot',
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

    # kshot_need_averaging = [c for c in all_model_data.columns if 'shot_seed' in c]
    
    # if len(kshot_need_averaging)>0:
    #     Kvals = [int(c.split('/')[1].split('shot')[0]) for c in kshot_need_averaging ]
    #     unique_Kvals = np.unique(Kvals)
    #     for K in unique_Kvals:
    #         cols = [c for c in kshot_need_averaging if f'{K}shot_seed' in c]
    #         # all_model_data[] all_model_data[cols]
    

    kshot_need_averaging_bool = all_model_data.columns.str.contains('shot_seed')
    kshot_need_averaging_cols = all_model_data.columns[kshot_need_averaging_bool]
    other_columns = all_model_data.columns[~kshot_need_averaging_bool]
    melted_df = pd.melt(
        all_model_data,
        id_vars=other_columns,
        value_vars=kshot_need_averaging_cols,
        var_name='column',
        value_name=r'$k$-shot cobps'
    )

    # Extract 'shot', 'seed', 'model', and 'target' from the 'column' names
    melted_df['column'] = melted_df['column'].str.replace('lfads_torch.post_run.fewshot_analysis.LinearLightning','LinearLightning')
    print(melted_df['column'].unique())
    split_columns = melted_df['column'].str.split('_', expand=True)
    melted_df['K'] = split_columns[0].str.extract(r'(\d+)').astype(int)
    melted_df['seed'] = split_columns[1].str.extract(r'seed(\d+)').astype(int)
    melted_df['model'] = split_columns[2]
    melted_df['target'] = split_columns[3]

    # Drop unnecessary columns
    melted_df.drop(columns=['column'], inplace=True)

    # Reorder columns as needed
    # melted_df = melted_df[['K', 'seed', 'target', r'$k$-shot cobps']]

    # melted_df = pd.concat([melted_df.reset_index(drop=True), all_model_data[other_columns].reset_index(drop=True)], axis=1)
    all_model_data_melted = melted_df
    print(np.isnan(
        all_model_data_melted[
            all_model_data_melted.model=='sklearnPoissonRegressor.alpha0.1.solverlbfgs'
        ][
            all_model_data_melted.K==50
        ][
            r'$k$-shot cobps'
        ].values).mean())

    print(
        np.isnan(all_model_data_melted[
            all_model_data_melted.model=='sklearnPoissonRegressor.alpha0.1.solverlbfgs'
        ][
            all_model_data_melted.K==20
        ][
            r'$k$-shot cobps'
        ].values).mean()
    )

    all_model_data_melted = all_model_data_melted[all_model_data_melted.target=='recon']

    g = sns.FacetGrid(all_model_data_melted, row='dataset',col='K', hue='model') #, hue='path_to_models')
    # g.map(sns.pointplot, 'valid/co_bps', r'$k$-shot cobps',  alpha=0.7, linestyle='none',errorbar='se',markersize=1,err_kws={'linewidth': 0.7}) #, s=10)
    g.map(sns.scatterplot, 'valid/co_bps',r'$k$-shot cobps', s=10)
    g.add_legend()
    # g.set(xticks=range(5))
    # for ax in g.axes[:,0]:
    #     ax.set_ylabel(rf'$k={100}$-shot cobps')
    axes = g.axes
    for ax in g.axes[-1,:]:
        ax.set_xlabel(rf'cobps')
    fig = g.figure
    

    fig.savefig(os.path.join('/home/kabird/lfads-torch-fewshot-benchmark/plots', 'co_bps_vs_k-shot_heldout.png'), dpi=250)
    fig.savefig(os.path.join('/home/kabird/lfads-torch-fewshot-benchmark/plots', 'co_bps_vs_k-shot_heldout.pdf'))
    plt.close()

    g = sns.FacetGrid(all_model_data_melted, row='dataset',col='K', hue='path_to_models')
    # g = sns.FacetGrid(all_model_data_melted)
    g.map(sns.pointplot, 'valid/co_bps', r'$k$-shot cobps',  alpha=0.7, linestyle='none',errorbar='se',markersize=1,err_kws={'linewidth': 0.7}) #, s=10)
    # g.map(sns.pointplot, 'valid/co_bps', r'$k$-shot cobps') #, s=10)
    # g.map(sns.scatterplot, 'valid/co_bps',r'$k$-shot cobps', s=10)
    g.add_legend()
    # g.set(xticks=range(5))
    for ax in g.axes[:,0]:
        ax.set_ylabel(rf'$k={100}$-shot cobps')
    for ax in g.axes[-1,:]:
        ax.set_xlabel(rf'cobps')
    for ax,ax_old in zip(g.axes.flatten(),axes.flatten()):
        ax.set_xticks(ax_old.get_xticks())
        ax.set_xticklabels(ax_old.get_xticklabels())

    fig = g.figure

    # fig,ax = plt.subplots()
    # sns.scatterplot(data=all_model_data_melted,x='valid/co_bps', y=r'$k$-shot cobps',ax=ax)

    fig.savefig(os.path.join('/home/kabird/lfads-torch-fewshot-benchmark/plots', 'co_bps_vs_mean_k-shot_heldout.png'), dpi=250)
    fig.savefig(os.path.join('/home/kabird/lfads-torch-fewshot-benchmark/plots', 'co_bps_vs_mean_k-shot_heldout.pdf'))
    plt.close()



    g = sns.FacetGrid(all_model_data,col="dataset", hue='path_to_models')
    # g.map(sns.pointplot, 'valid/co_bps',f'post_run/{100}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps', alpha=0.4, linestyle='none')#, s=10)
    g.map(sns.scatterplot, 'valid/co_bps',f'post_run/{100}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps', s=10)
    g.add_legend()
    for ax in g.axes[:,0]:
        ax.set_ylabel(rf'$k={100}$-shot cobps')
    for ax in g.axes[-1,:]:
        ax.set_xlabel(rf'cobps')
    fig = g.figure
    
    fig.savefig(os.path.join('/home/kabird/lfads-torch-fewshot-benchmark/plots', 'co_bps_vs_100shot_heldout.png'), dpi=250)

    

if __name__ == '__main__':
    main()