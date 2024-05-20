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
from hydra.utils import instantiate
import seaborn as sns
import torch

from copy import deepcopy

import matplotlib as mpl


mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

CONFIG_PATH = "../configs"
# CONFIG_NAME = "config"
# CONFIG_NAME = "config_cohmm_mc_maze"
# path_to_models = '/home/kabird/ray_results/all_models_validated_v2/teacher_state4_poisson_partial_eps0.01_length35/combined_traintrials1600'

CONFIG_NAME = "comparative_config"
# path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240316_144215_MultiFewshot'
# path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240318_144734_MultiFewshot'
path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240319_085230_MultiFewshot'
# path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240328_171607_MultiFewshot'
# path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240329_193308_MultiFewshot'
# path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240329_201611_MultiFewshot'
# path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_rtt/240331_130041_MultiFewshot'
# path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240402_110106_MultiFewshot' # 20 models
# path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240402_111015_MultiFewshot' # 200 models
dataframe_file_name = 'latents_dataframe.pkl'
threshold = 2e-3

test_train_split = 0.7

def plot_scatter_with_lines(
        x: Optional[str],
        y: Optional[str],
        data: pd.DataFrame,
        save_path: str,
        hue: Optional[str] = None,
        data_lines: Optional[pd.DataFrame] = None,
        func1 = sns.scatterplot,
        func2 = None,
        sortby = [],
        xlabel=None,
        ylabel=None,
        xlim=[],
        ylim=[],
        hlines=[],
        print_corrcoef = False,
        zoom_inset: Optional[dict] = None,
):
    fig,axs = plt.subplots()
    all_axes = [axs]
    if zoom_inset is not None:
        # ax_ins = inset_axes(axs, **zoom_inset)
        ax_ins = axs.inset_axes(**zoom_inset)
        all_axes.append(ax_ins)
    for ax in all_axes:
        data_ = data.sort_values(by=sortby)
        func1(x=x, y=y, hue=hue, data=data_, ax=ax)
        if func2:
            # print(func2)
            # print(data[[x,y,hue]].sort())
            # data_ = data.sort_values(by=hue)
            func2(x=x,y=y,data = data_,ax=ax)
        if data_lines is not None:
            # print('here',data_lines[y].values[0])
            for color,(name_,x_,y_) in zip(['black','blue','red'],data_lines[['model_name',x,y]].values):
                # print(x_,y_)
                # l = ax.axhline(data_lines[y].values[0], ls='dashed', color='black')
                # l = ax.axvline(data_lines[x].values[0], ls='dashed', color='black')
                l = ax.axhline(y_, ls='dashed', color=color,label=name_)
                ax.axvline(x_, ls='dashed', color=color)
        for i,ls in enumerate(hlines):
            ax.axhline(ls,color='C%d'%i,ls='dashed',lw=1)
        handles, labels = ax.get_legend_handles_labels()
        # if data_lines is not None:
        #     handles += [l]
        #     labels  += ['Ground-truth']
    axs.legend(handles,labels,fontsize=7,framealpha=0.3)
    axs.set_xlim(*xlim)
    axs.set_ylim(*ylim)
    axs.set_xlabel(x if xlabel is None else xlabel)
    axs.set_ylabel(y if ylabel is None else ylabel)
    if print_corrcoef:
        a,b = data[[x,y]].dropna().values.T
        print(a,b)
        corrcoef = np.corrcoef(a,b)[0,1]
        print('corrcoef',corrcoef)
        ax.set_title('Corrcoef=%.2f'%corrcoef)
    if zoom_inset:
        # ax_ins.set_xlim(*zoom_xlim)
        # ax_ins.set_ylim(*zoom_ylim)
        ax_ins.set_xlabel(None)
        ax_ins.set_ylabel(None)
        legend = ax_ins.legend()
        legend.remove()
        axs.indicate_inset_zoom(ax_ins, edgecolor="black")
    fig.tight_layout()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig.savefig(save_path,dpi=300)
    plt.close()


def load_and_filternan_models_with_csvs():
    saveloc = os.path.join(path_to_models, dataframe_file_name)
    with open(saveloc, 'rb') as f:
        latents_dataframe = pkl.load(f)
    is_best = latents_dataframe.co_bps > (latents_dataframe.co_bps.max() - threshold)
    print('Latents dataframe len', len(latents_dataframe))
    print(latents_dataframe.columns)
    print('Num best models', is_best.sum())

    all_files = os.listdir(path_to_models)

    model_files = [f for f in all_files if (f[-4] != '.' and f[-3] != '.')]
    # model_files = model_files[:4]

    models = []
    model_datas = []
    for f in model_files:
        full_path = os.path.join(path_to_models, f)
        with open(full_path, 'rb') as _f:
            models.append(pkl.load(_f))

        full_path = full_path + '.csv'

        model_data = pd.read_csv(full_path) if os.path.exists(full_path) else None

        model_datas.append(model_data)
    print('Loaded', len(models), 'models.')
    filtered_models, model_datas = zip(
        *[(m, d) for m, d in zip(models, model_datas) if not np.any(np.isnan(m.transmat_))])
    print('Removed', len(models) - len(filtered_models), 'models with NaN params.')
    models = filtered_models

    print('Num models:', len(model_datas))
    model_datas = pd.concat(model_datas, axis=0).reset_index().drop(columns=['iterations'])  # ['6-shot co-smoothing']
    model_datas['is_best'] = is_best
    csv_path = os.path.join(path_to_models, 'concat_model_data.csv')
    model_datas.to_csv(csv_path)


def load_model_datas():
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
    # model_datas_ = model_datas_.T.T
    # print(model_datas_.shape,len(model_datas))
    # print(pd.concat(model_datas).shape)
    # print(model_datas_.T.shape)
    # print(model_datas_['valid/1000shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps'])
    print(model_datas_)
    csv_path = os.path.join(path_to_models, 'concat_model_data.csv')
    model_datas_.to_csv(csv_path)

def load_latents(dataframe):
    # csv_path = os.path.join(path_to_models, 'concat_model_data.csv')
    # D = pd.read_csv(csv_path)

    all_files = dataframe['model_id'] #os.listdir(path_to_models)

    model_files = [f for f in all_files if f.startswith('run_model')]
    
    
    model_datas = []
    for f in model_files:
        full_path = os.path.join(path_to_models, f,'model_outputs_valid')
        if os.path.exists(full_path):
            numpy_array = torch.load(full_path).cpu().numpy()
            model_datas.append(numpy_array)
        else:
            model_datas.append(None)
    
    # total_trials = model_datas[0].shape[0]
    # train_trials = int(test_train_split * total_trials)
    # test_trials = total_trials-train_trials
    
    dataframe['train_latents'] = [(m[:int(model_datas[0].shape[0] * 0.7)] if m is not None else None) for m in model_datas]
    dataframe['test_latents']  = [(m[int(model_datas[0].shape[0] * 0.7):]  if m is not None else None) for m in model_datas]
    return dataframe


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def load_models_and_store_latents(cfg):
    omegaconf_resolvers()

    all_files = os.listdir(path_to_models)

    model_files = [f for f in all_files if f[-4] != '.']
    # model_files = model_files[:4]

    models = []
    for f in model_files:
        full_path = os.path.join(path_to_models, f)
        with open(full_path, 'rb') as _f:
            models.append(pkl.load(_f))

    print('Loaded', len(models), 'models.')
    filtered_models = [m for m in models if not np.any(np.isnan(m.transmat_))]
    print('Removed', len(models) - len(filtered_models), 'models with NaN params.')
    models = filtered_models

    ### generating data
    instantiate(cfg.numpy_seed)
    if cfg.data_mode == 'student-teacher':
        teacher = instantiate(cfg.teacher)
        data = instantiate(cfg.generate_all_data_dictmodule, _convert_='partial')(hmm_model=teacher)
    else:
        datamodule = instantiate(cfg.datamodule, _convert_="all")
        data_numpy = lfads_torch_datamodule_to_numpy(datamodule)[:, :35, :].astype(int)
        # data_numpy[data_numpy>=1] = 1
        print(data_numpy.shape)
        data = instantiate(cfg.numpy_to_xarray_with_breakdownlabels, _convert_='partial')(data=data_numpy)

    all_model_data = []
    for model in tqdm(models):
        model_data = {}

        ### co-smoothing
        student = model
        bits_per_spike = instantiate(cfg.bits_per_spike_func)
        test_student = deepcopy(student)
        (
            test_student_in,
            test_student_out
        ) = split_model_emission(
            test_student,
            split_indices=instantiate(cfg.neurons_split_indices)[:1]
        )
        split_student = CoHMM(test_student_in, test_student_out)
        # print('transmat sum',split_student.encoder.transmat_.sum(-1))
        test_pred_out = split_student.predict(data.select(**cfg.breakups.cosmoothing.input), mode3d=True)
        test_pred_out = test_pred_out.reshape(*data.select(**cfg.breakups.cosmoothing.target).shape)
        test_pred_out[np.isnan(test_pred_out)] = 0
        co_bps = bits_per_spike(test_pred_out, data.select(**cfg.breakups.cosmoothing.target).to_numpy())
        model_data['co_bps'] = co_bps

        #### full prediction
        split_student2 = CoHMM(test_student_in, test_student)
        # print('transmat sum',split_student.encoder.transmat_.sum(-1))
        test_pred_full = split_student2.predict(data.select(**cfg.breakups.cosmoothing_full.input), mode3d=True)
        target_full = data.select(**cfg.breakups.cosmoothing_full.target)
        test_pred_full = test_pred_full.reshape(*target_full.shape)
        model_data['test_pred_full'] = test_pred_full
        model_data['target_full'] = target_full

        #### get latents for cross-decoding
        train_input_data = data.select(**cfg.breakups.decoding_full.fit.input).values
        test_input_data = data.select(**cfg.breakups.decoding_full.test.input).values
        student_inout = split_model_emission(student, split_indices=instantiate(cfg.neurons_split_indices)[1:2])[0]
        student_inout_dynamax, params, param_props = hmmlearn_to_dynamaxhmm(student_inout)
        # train_latents = student_inout.predict_proba(train_input_data,mode3d=True)
        # test_latents = student_inout.predict_proba(test_input_data, mode3d=True)
        smooth = vmap(student_inout_dynamax.smoother, (None, 0), 0)
        train_latents = smooth(params, jnp.asarray(train_input_data)).smoothed_probs
        test_latents = smooth(params, jnp.asarray(test_input_data)).smoothed_probs
        train_latents, test_latents = [np.array(thing) for thing in [train_latents, test_latents]]
        model_data['train_latents'] = train_latents
        model_data['test_latents'] = test_latents

        all_model_data.append(model_data)

    D = pd.DataFrame(all_model_data)
    saveloc = os.path.join(path_to_models, dataframe_file_name)
    with open(saveloc, 'wb') as f:
        pkl.dump(D, f)


def train_model(model, dataset):
    """Function to train a regression model"""
    X, y = dataset
    model.fit(X, y)
    return model


def score_model(model, dataset, metric, predict_method):
    X, y = dataset
    pred_y = getattr(model, predict_method)(X)
    score = np.stack([metric(
        y[sample_id],
        pred_y[sample_id]
    ) for sample_id in range(pred_y.shape[0])]).mean()
    return score


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def cross_decoding(cfg):
    # omegaconf_resolvers()

    saveloc = os.path.join(path_to_models, 'concat_model_data.csv')
    latents_dataframe = pd.read_csv(saveloc)
    latents_dataframe = load_latents(latents_dataframe)

    best_latents_dataframe = latents_dataframe[latents_dataframe['valid/co_bps'] > (latents_dataframe['valid/co_bps'].max() - 2e-2)]
    
    # best_latents_dataframe = best_latents_dataframe.head(2)

    n_models = len(best_latents_dataframe)

    print('n_models:', n_models)

    train_latents = best_latents_dataframe['train_latents'].values
    train_latents_r = [thing.reshape(-1, thing.shape[-1]) for thing in train_latents]
    if hasattr(cfg.decoding, 'preprocess_target'):
        preprocess = instantiate(cfg.decoding.preprocess_target)
        train_latents_r = [preprocess(thing) for thing in train_latents_r]
    train_datasets = []
    for i, j in tqdm(list(product(range(n_models), range(n_models)))):
        train_datasets.append(
            (train_latents_r[i], train_latents_r[j])
        )
    test_latents = best_latents_dataframe['test_latents'].values
    test_latents_r = [thing.reshape(-1, thing.shape[-1]) for thing in test_latents]
    test_datasets = []
    for i, j in tqdm(list(product(range(n_models), range(n_models)))):
        test_datasets.append(
            (test_latents_r[i], test_latents_r[j])
        )
    models = [
        instantiate(cfg.decoding.regression_model)
        for m in range(len(train_datasets))
    ]
    metric = instantiate(cfg.decoding.metric)
    with Pool() as p:
        trained_models = p.starmap(train_model, zip(models, train_datasets))
        scores = p.starmap(
            score_model,
            zip(
                trained_models,
                test_datasets,
                [metric] * len(test_datasets),
                [cfg.decoding.predict_method] * len(test_datasets)
            )
        )

    print(len(scores))
    score_dataframe = pd.DataFrame({
        'from_to_index' : list(product(range(n_models), range(n_models))),
        'score' : scores,
    })
    score_dataframe[['from','to']]=pd.DataFrame(score_dataframe['from_to_index'].to_list(),index=score_dataframe.index)
    print(score_dataframe)
    score_dataframe['from_id'] = best_latents_dataframe['model_id'].iloc[score_dataframe['from']].values
    score_dataframe['to_id'] = best_latents_dataframe['model_id'].iloc[score_dataframe['to']].values
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores.csv')
    score_dataframe.to_csv(saveloc)
    
    scores = np.array(scores).reshape(n_models, n_models)
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores_parallel')
    np.save(saveloc, scores)

def plotting_histogram():
    saveloc = os.path.join(path_to_models, 'concat_model_data.csv')
    latents_dataframe = pd.read_csv(saveloc)

    fig, ax = plt.subplots()
    metric_name = 'valid/co_bps'
    ax.hist(latents_dataframe[metric_name], bins=20) #(np.arange(0, 0.4, 0.01))
    ax.set_xlabel(metric_name)
    fig.savefig(os.path.join(path_to_models, 'co_bps_hist.png'), dpi=250)

    
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'cobps_heldin_colsums.png'), dpi=300)

    fig, ax = plt.subplots()
    sns.regplot(
        x='valid/co_bps',
        # y='valid/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps',
        y='post_run/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps',
        data=latents_dataframe,
        ax=ax,
    )
    ax.set_ylabel(r'$k=100$-shot co_bps to held out')
    ax.set_xlabel('co_bps')
    # ax.set_ylim(0.0,0.4)
    # ax.set_xlim(0.2,0.4)
    r = scipy.stats.pearsonr(x=latents_dataframe['valid/co_bps'], 
                             y=latents_dataframe['post_run/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps'])[0]
    # ax.plot([0,1],[0,1],ls='dashed',c='black')
    ax.set_title('Pearsonr:{:1.2f}'.format(r))
    fig.savefig(os.path.join(path_to_models, 'co_bps_vs_100shot_heldout.png'), dpi=250)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x='valid/co_bps',
        y=f'debugging/val_{500}shot_co_bps_recon_truereadout',
        hue='hp/dropout_rate',
        data=latents_dataframe,
        ax=ax
    )
    # ax.plot([0.3,0.425],[0.3,0.425],ls='dashed',c='black')
    ax.set_aspect('equal')
    ax.set_ylabel('co_bps on held-in neurons')
    ax.set_xlabel('co_bps on held-out neurons')
    fig.savefig(os.path.join(path_to_models, 'co_bps_heldin_heldout.png'), dpi=250)

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def plotting(cfg):
    omegaconf_resolvers()

    saveloc = os.path.join(path_to_models, dataframe_file_name)
    with open(saveloc, 'rb') as f:
        latents_dataframe = pkl.load(f)

    fig, ax = plt.subplots()
    ax.hist(latents_dataframe['co_bps'], bins=np.arange(0, 0.4, 0.01))
    fig.savefig(os.path.join(path_to_models, 'co_bps_hist.png'), dpi=250)

    best_latents_dataframe = latents_dataframe[latents_dataframe.co_bps > (latents_dataframe.co_bps.max() - 1e-2)]
    print('Plotting', len(best_latents_dataframe), 'models.')

    fig, axs = plt.subplots(2, 3, sharex=True)
    for i, ax in enumerate(axs.flatten()):
        x = best_latents_dataframe['train_latents'].iloc[i][0].T
        # x = np.log10(x)
        print(x.shape)
        ax.imshow(x, aspect='auto')
        ax.set_yticklabels([])
        xticks = np.arange(0, x.shape[1], 10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    axs[-1, 0].set_xlabel('time', fontsize=8)
    axs[-1, 0].set_ylabel('HMM states', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'latents_samples.png'), dpi=250)

    fig, axs = plt.subplots(1, 4, sharex=True)
    target = best_latents_dataframe['target_full'].iloc[0][0].T
    prop = {
        'aspect': 'equal',
        'interpolation': 'none',
    }
    lineprop = {
        'color': 'red',
        'ls': 'dashed',
        'lw': 1,
    }
    axs[0].imshow(target, **prop)
    axs[0].axhline(cfg.num_neurons_heldin + 0.5, **lineprop)
    for i in range(1, len(axs)):
        pred = best_latents_dataframe['test_pred_full'].iloc[i][0].T
        # target = best_latents_dataframe['target_full'].iloc[i][0].T
        ax = axs[i]
        ax.imshow(pred, **prop)
        ax.axhline(cfg.num_neurons_heldin + 0.5, **lineprop)
        ax.set_yticklabels([])
        xticks = np.arange(0, x.shape[1], 10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    axs[0].set_xlabel('time', fontsize=8)
    axs[0].set_ylabel('channels', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'targets_and_predictions.png'), dpi=300)


def plot_cross_decoding_scores():
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores_parallel.npy')
    scores = np.load(saveloc)
    print(scores.shape)

    fig, ax = plt.subplots()
    im = ax.imshow(1-scores)
    ax.set_ylabel('input model')
    ax.set_xlabel('target model')
    fig.colorbar(im, ax=ax, label=r'$1-R^2$')
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'cross_decoding.png'), dpi=300)

    plt.close(fig)

    # fig, ax = plt.subplots()
    # ax.hist(np.log10(scores.flatten()), bins=30)
    # fig.savefig(os.path.join(path_to_models, 'scores_histogram.png'))


def plot_kshot_and_crossdecoding():
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores_parallel.npy')
    scores = np.load(saveloc)
    
    saveloc = os.path.join(path_to_models, 'concat_model_data.csv')
    latents_dataframe = pd.read_csv(saveloc)
    latents_dataframe['model_id'] = latents_dataframe['model_id'].str.split('_').str[-1]
    latents_dataframe.index = latents_dataframe['model_id']
    
    # additionally condition on good heldin co_bps
    # latents_dataframe = latents_dataframe[latents_dataframe[f'debugging/val_{1000}shot_co_bps_recon_truereadout']>0.36]
    latents_dataframe = latents_dataframe[latents_dataframe['valid/co_bps'] > (latents_dataframe['valid/co_bps'].max() - 1e-2)]
    # latents_dataframe = latents_dataframe[latents_dataframe.co_bps > (latents_dataframe.co_bps.max() - 1e-2)]

    saveloc = os.path.join(path_to_models, 'cross_decoding_scores.csv')
    score_dataframe = pd.read_csv(saveloc)
    score_dataframe['from_id'] = score_dataframe['from_id'].str.split('_').str[-1]
    score_dataframe['to_id']   = score_dataframe['to_id'].str.split('_').str[-1]
    square_score_dataframe = score_dataframe.drop(['from_to_index','from','to'],axis=1).pivot(
        index = 'from_id',
        columns = 'to_id',
        values = 'score'
    )
    print(len(latents_dataframe.index.values))
    select_ids = latents_dataframe.index.values
    selection = square_score_dataframe.index.isin(select_ids)
    print(square_score_dataframe)
    square_score_dataframe = square_score_dataframe.loc[selection,:].loc[:,selection]
    square_dist_dataframe = 1 - np.array(square_score_dataframe.values)
    col_sums = square_dist_dataframe.mean(axis=0)
    

    select_latents_dataframe = latents_dataframe.loc[list(square_score_dataframe.index.values)]
    # k1000shot_reallyheldout = select_latents_dataframe['valid/1000shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps'].values
    k100shot_reallyheldout = select_latents_dataframe['valid/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps'].values
    select_latents_dataframe['col_sums'] = col_sums
    kshot_heldout = select_latents_dataframe['post_run/100shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps']
    co_bps = select_latents_dataframe['valid/co_bps'].values

    co_bps_heldin = select_latents_dataframe[f'debugging/val_{100}shot_co_bps_recon_truereadout']
    dropout_rate = select_latents_dataframe['hp/dropout_rate']

    Krange = [100,500]  #[100,500,1000]
    Kshot_vs_Krange = np.stack([
        select_latents_dataframe[f'valid/{k}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps'].values
        for k in Krange
    ])
    Kshot_vs_Krange = np.stack([
        select_latents_dataframe[f'post_run/{k}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps'].values
        for k in Krange
    ])

    # print(kshot)
    # print(col_sums.shape,kshot.shape,square_score_dataframe.index.values.shape,latents_dataframe.loc[list(square_score_dataframe.index.values)[:2]].shape)
    # print(square_score_dataframe.index.values)
    # print(latents_dataframe.loc[list(square_score_dataframe.index.values)[:2]])
    # print(latents_dataframe.shape)
    fig,ax = plt.subplots()
    ax.scatter(col_sums,k100shot_reallyheldout)

    ax.set_xlabel(r'Column sum of $1-R^2$')
    ax.set_ylabel(r'$k=100$-shot co_bps to really held out')
    ax.set_ylim(0.2,0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'k100shot_colsums.png'), dpi=300)
    plt.close()

    # fig,ax = plt.subplots()
    # ax.scatter(col_sums,k1000shot_reallyheldout)

    # ax.set_xlabel(r'Column sum of $1-R^2$')
    # ax.set_ylabel(r'$k=1000$-shot co_bps to really held out')
    
    # fig.tight_layout()
    # fig.savefig(os.path.join(path_to_models, 'k1000shot_colsums.png'), dpi=300)
    # plt.close()

    fig,ax = plt.subplots()
    ax.scatter(col_sums,co_bps)
    # ax.scatter(col_sums,kshot_heldout)
    # ax.scatter(col_sums,co_bps)
    ax.set_xlabel(r'Column sum of $1-R^2$')
    ax.set_ylabel(r'co_bps on held-out neurons')
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'co_bps_colsums.png'), dpi=300)
    plt.close()



    fig,ax = plt.subplots()
    ax.scatter(col_sums,kshot_heldout)
    # ax.scatter(col_sums,co_bps)
    ax.set_xlabel(r'Column sum of $1-R^2$')
    ax.set_ylabel(r'$100$-shot on held-out neurons')
    
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'kshot-heldout_colsums.png'), dpi=300)
    plt.close()


    fig,ax = plt.subplots()
    ax.scatter(col_sums,co_bps_heldin)
    # ax.scatter(col_sums,co_bps)
    ax.set_xlabel(r'Column sum of $1-R^2$')
    ax.set_ylabel(r'co_bps on held-in neurons')
    
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'cobps_heldin_colsums.png'), dpi=300)
    plt.close()



    fig,ax = plt.subplots()
    ax.scatter(k100shot_reallyheldout,co_bps_heldin)
    # ax.scatter(col_sums,co_bps)
    ax.set_xlabel(r'$k=100$-shot co_bps to really held out')
    ax.set_ylabel(r'co_bps on held-in neurons')
    
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'cobps_heldin_1000shot.png'), dpi=300)
    plt.close()

    # fig,ax = plt.subplots()
    # ax.scatter(k1000shot_reallyheldout,co_bps_heldin)
    # # ax.scatter(col_sums,co_bps)
    # ax.set_xlabel(r'$k=1000$-shot co_bps to really held out')
    # ax.set_ylabel(r'co_bps on held-in neurons')
    
    # fig.tight_layout()
    # fig.savefig(os.path.join(path_to_models, 'cobps_heldin_1000shot.png'), dpi=300)
    # plt.close()

    fig,ax = plt.subplots()
    ax.plot(Krange,Kshot_vs_Krange)
    # ax.scatter(col_sums,kshot_heldout)
    # ax.scatter(col_sums,co_bps)
    ax.set_xlabel(r'k')
    ax.set_ylabel(r'$k$-shot co_bps')
    ax.set_xscale('log')
    
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_models, 'kshot_vs_krange.png'), dpi=300)
    plt.close()


    plot_configs = [
        {
            'x': 'col_sums',
            'y': 'valid/co_bps',  
            'save_path': os.path.join(path_to_models, 'scatterplots','cobps_colsums.png'),
            'data': select_latents_dataframe,
            'print_corrcoef':True,
        },
        {
            'x': 'col_sums',
            'y': f'valid/{100}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps',  
            'save_path': os.path.join(path_to_models, 'scatterplots',f'k{100}shot_colsums.png'),
            'data': select_latents_dataframe,
            'xlabel':fr'Column sum $1-R^2$',
            'ylabel':r'$k={100}$-shot',
            'print_corrcoef':True,
        },
        {
            'x': 'col_sums',
            'y': f'valid/{500}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps',  
            'save_path': os.path.join(path_to_models, 'scatterplots',f'k{500}shot_colsums.png'),
            'data': select_latents_dataframe,
            'xlabel':fr'Column sum $1-R^2$',
            'ylabel':r'$k={500}$-shot',
            'print_corrcoef':True,
        },
        {
            'x': 'col_sums',
            'y': f'post_run/{100}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps',  
            'save_path': os.path.join(path_to_models, 'scatterplots',f'k{100}shot_heldout_colsums.png'),
            'data': select_latents_dataframe,
            'xlabel':fr'Column sum $1-R^2$',
            'ylabel':r'$k={100}$-shot held-out',
            'print_corrcoef':True,
        },
        {
            'x': 'col_sums',
            'y': f'post_run/{500}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps',  
            'save_path': os.path.join(path_to_models, 'scatterplots',f'k{500}shot_heldout_colsums.png'),
            'data': select_latents_dataframe,
            'xlabel':fr'Column sum $1-R^2$',
            'ylabel':r'$k={500}$-shot held-out',
            'print_corrcoef':True,
        },
        {
            'x': 'valid/co_bps',
            'y': f'valid/{100}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps',  
            'save_path': os.path.join(path_to_models, 'scatterplots',f'cobps_k{100}shot.png'),
            'data': select_latents_dataframe,
            'ylabel':r'$k={100}$-shot',
            'print_corrcoef':True,
            
        },
                {
            'x': 'valid/co_bps',
            'y': f'valid/{500}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps',  
            'save_path': os.path.join(path_to_models, 'scatterplots',f'cobps_k{500}shot.png'),
            'data': select_latents_dataframe,
            'ylabel':r'$k={500}$-shot',
            'print_corrcoef':True,
        },
        {
            'x' : 'valid/co_bps',
            'y' :  f'post_run/{100}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps',
            'save_path': os.path.join(path_to_models, 'scatterplots',f'k{100}shot_heldout_cobps.png'),
            'data': select_latents_dataframe,
            'ylabel':r'$k={100}$-shot held-out',
            'print_corrcoef':True,
        },        
        {
            'x' : 'valid/co_bps',
            'y' :  f'post_run/{500}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_co_bps',
            'save_path': os.path.join(path_to_models, 'scatterplots',f'k{500}shot_heldout_cobps.png'),
            'data': select_latents_dataframe,
            'ylabel':r'$k={500}$-shot held-out',
            'print_corrcoef':True,
        }
    ]
    for conf in plot_configs:
        plot_scatter_with_lines(**conf)
    

def check_extremes():
    saveloc = os.path.join(path_to_models, 'concat_model_data.csv')
    latents_dataframe = pd.read_csv(saveloc)
    latents_dataframe = load_latents(latents_dataframe)

    latents_dataframe.index = latents_dataframe['model_id'].str.split('_').str[-1]

    saveloc = os.path.join(path_to_models, 'cross_decoding_scores.csv')
    score_dataframe = pd.read_csv(saveloc)
    square_score_dataframe = score_dataframe.drop(['from_to_index','from','to'],axis=1).pivot(
        index = 'from_id',
        columns = 'to_id',
        values = 'score'
    )
    colmean_sorted = (square_score_dataframe.apply(lambda x:1-x)).mean(axis=0).sort_values(ascending=True)
    min_colmean = colmean_sorted.head(1) # what we think should be the good model
    max_colmean = colmean_sorted.tail(1) # what we think should be the bad model
    print(
        min_colmean.index,
        max_colmean.index,
        colmean_sorted
    )
    print(
        'From min to max col-mean model 1-R2',1-square_score_dataframe.loc[min_colmean.index,max_colmean.index].values,'\n'
        'From max to min col-mean model 1-R2',1-square_score_dataframe.loc[max_colmean.index,min_colmean.index].values
    )

    n_components = 3
    num_trials_to_plot = 5
    for name,model_id in zip(['min_colmean','max_colmean'],[min_colmean.index,max_colmean.index]):
        id = model_id.values[0].split('_')[-1]
        # print([thing.shape for thing in latents_dataframe.loc[id][['train_latents','test_latents']].values])
        latents = np.concatenate(latents_dataframe.loc[id][['train_latents','test_latents']].values,axis=0)
        P = PCA(n_components=n_components)
        latents_pcproj = P.fit_transform(latents.reshape(-1,latents.shape[-1])).reshape(*latents.shape[:2],n_components)

        fig = plt.figure()
        ax = fig.add_subplot()#projection='3d')
        print([t.shape for t in latents_pcproj[...,:3].swapaxes(0,-1).swapaxes(1,2)])
        for i in range(num_trials_to_plot):
            ax.plot(latents_pcproj[i,...,0].T,latents_pcproj[i,...,1].T,lw=1.4,alpha=0.7)#,latents_pcproj[...,2])
            ax.scatter(latents_pcproj[i,0,0],latents_pcproj[i,0,1],c='green',s=10)
            ax.scatter(latents_pcproj[i,-1,0],latents_pcproj[i,-1,1],c='red',s=10)
        ax.axis('off')
        saveloc = os.path.join(path_to_models, name+'_PCA.png')
        fig.tight_layout()
        fig.savefig(saveloc,dpi=300)
        plt.close()


    data = []
    for model_id in colmean_sorted.index[:]:
        id = model_id.split('_')[-1]
        latents = np.concatenate(latents_dataframe.loc[id][['train_latents','test_latents']].values,axis=0)
        p = PCA()
        p.fit(latents.reshape(-1,latents.shape[-1]))
        data.append(pd.DataFrame({
            'variance explained':p.explained_variance_ratio_,
            'model_id':model_id,
            'components':np.arange(p.n_components_),
            'decoding_colmean':[colmean_sorted.loc[model_id]]*p.n_components_,
        }))

    data = pd.concat(data,axis=0)
    
    fig,ax = plt.subplots()
    sns.lineplot(x='components',y='variance explained',units='model_id',hue='decoding_colmean',data=data,ax=ax)
    ax.set_yscale('log')
    # ax.set_xlim(0,20)
    # ax.set_ylim(1e-2,2e-1)
    fig.tight_layout()
    saveloc = os.path.join(path_to_models, 'variance_explained.png')
    fig.savefig(saveloc,dpi=300)
    
def convert_cross_decoding_scores_to_matlab():
    saveloc = os.path.join(path_to_models, 'concat_model_data.csv')
    latents_dataframe = pd.read_csv(saveloc)
    latents_dataframe = load_latents(latents_dataframe)
    latents_dataframe.index = latents_dataframe['model_id'].str.split('_').str[-1]


    saveloc = os.path.join(path_to_models, 'cross_decoding_scores_parallel.npy')
    scores = np.load(saveloc)
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores.csv')
    score_dataframe = pd.read_csv(saveloc)
    square_score_dataframe = score_dataframe.drop(['from_to_index','from','to'],axis=1).pivot(
        index = 'from_id',
        columns = 'to_id',
        values = 'score'
    )
    print(square_score_dataframe)
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores_and_kshot.mat')
    id = square_score_dataframe.index.str.split('_').str[-1]
    dict_to_save = {
        'scores' : square_score_dataframe.values,
        'id'  : id,
        '1000shot_reallyheldout' : latents_dataframe.loc[id][f'valid/{1000}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps'],
        '500shot_reallyheldout' : latents_dataframe.loc[id][f'valid/{500}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps'],
        '100shot_reallyheldout' : latents_dataframe.loc[id][f'valid/{100}shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps'],
    }
    print(dict_to_save['100shot_reallyheldout'])
    scipy.io.savemat(saveloc, dict_to_save)


if __name__ == '__main__':
    # load_models_and_store_latents()
    # plotting()
    # cross_decoding()
    # plot_cross_decoding_scores()
    # convert_cross_decoding_scores_to_matlab()
    # load_and_filternan_models_with_csvs()
    
    # load_model_datas()
    # plotting_histogram()
    # cross_decoding()
    # plot_cross_decoding_scores()
    # plot_kshot_and_crossdecoding()
    check_extremes()
    # convert_cross_decoding_scores_to_matlab()