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
# from hydra.utils import instantiate
from hydra.utils import instantiate
import seaborn as sns

from copy import deepcopy

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

CONFIG_PATH = "configs"
# CONFIG_NAME = "config"
# CONFIG_NAME = "config_cohmm_mc_maze"
# path_to_models = '/home/kabird/ray_results/all_models_validated_v2/teacher_state4_poisson_partial_eps0.01_length35/combined_traintrials1600'

CONFIG_NAME = "comparative_config"
path_to_models = '/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240316_144215_MultiFewshot'
dataframe_file_name = 'latents_dataframe.pkl'
threshold = 2e-3

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

    model_files = [f for f in all_files if (f[-4] != '.' and f[-3] != '.')]
    
    models = []
    model_datas = []
    for f in model_files:
        full_path = os.path.join(path_to_models, f,'csv_logs','metrics.csv')
        model_data = pd.read_csv(full_path) if os.path.exists(full_path) else None
        model_datas.append(model_data)
    model_datas = pd.concat(model_datas, axis=0).reset_index()
    csv_path = os.path.join(path_to_models, 'concat_model_data.csv')
    model_datas.to_csv(csv_path)



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
    omegaconf_resolvers()

    saveloc = os.path.join(path_to_models, dataframe_file_name)
    with open(saveloc, 'rb') as f:
        latents_dataframe = pkl.load(f)

    best_latents_dataframe = latents_dataframe[latents_dataframe.co_bps > (latents_dataframe.co_bps.max() - 1e-2)]

    # best_latents_dataframe = best_latents_dataframe.head(2)

    n_models = len(best_latents_dataframe)

    print('n_models:', n_models)
    # scores = np.zeros((n_models,n_models))

    # def regression_from_to(i,j):
    #     X = best_latents_dataframe.iloc[i]['train_latents']
    #     y = best_latents_dataframe.iloc[j]['train_latents']
    #     X,y = [thing.reshape(-1,thing.shape[-1]) for thing in [X,y]]
    #     print(cfg.decoding)
    #     if hasattr(cfg.decoding, 'preprocess_target'):
    #         y = instantiate(cfg.decoding.preprocess_target)(y)
    #     # X = np.log(X)
    #     # from sklearn.linear_model import LinearRegression
    #     model = instantiate(cfg.decoding.regression_model)
    #     model.fit(
    #         X,
    #         y
    #     )

    #     X = best_latents_dataframe.iloc[i]['test_latents']
    #     y = best_latents_dataframe.iloc[j]['test_latents']
    #     X, y = [thing.reshape(-1, thing.shape[-1]) for thing in [X, y]]
    #     X = np.log(X)
    #     pred_y = getattr(model, cfg.decoding.predict_method)(X)

    #     metric = instantiate(cfg.decoding.metric)
    #     score = np.stack([metric(
    #         y[sample_id],
    #         pred_y[sample_id]
    #     ) for sample_id in range(pred_y.shape[0])]).mean()
    #     # scores[i,j] = score
    #     return score

    train_latents = best_latents_dataframe['train_latents'].values
    train_latents_r = [thing.reshape(-1, thing.shape[-1]) for thing in train_latents]
    if hasattr(cfg.decoding, 'preprocess_target'):
        preprocess = instantiate(cfg.decoding.preprocess_target)
        train_latents_sampled = [preprocess(thing) for thing in train_latents_r]
    train_datasets = []
    for i, j in tqdm(list(product(range(n_models), range(n_models)))):
        train_datasets.append(
            (train_latents_r[i], train_latents_sampled[j])
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

    fig, ax = plt.subplots()
    sns.scatterplot(
        x='valid/co_bps',
        y='valid/1000shot_lfads_torch.post_run.fewshot_analysis.LinearLightning_reallyheldout_bps',
        data=latents_dataframe,
        ax=ax
    )
    ax.set_ylabel(r'$k=1000$-shot co_bps to really held out')
    ax.set_xlabel('co_bps')
    fig.savefig(os.path.join(path_to_models, 'co_bps_vs_1000shot.png'), dpi=250)

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
    im = ax.imshow(scores)
    ax.set_ylabel('input model')
    ax.set_xlabel('target model')
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(path_to_models, 'cross_decoding.png'), dpi=300)

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.log10(scores.flatten()), bins=30)
    fig.savefig(os.path.join(path_to_models, 'scores_histogram.png'))


def convert_cross_decoding_scores_to_matlab():
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores_parallel.npy')
    scores = np.load(saveloc)
    print(scores)
    saveloc = os.path.join(path_to_models, 'cross_decoding_scores.mat')
    scipy.io.savemat(saveloc, {'scores': scores})


if __name__ == '__main__':
    # load_models_and_store_latents()
    # plotting()
    # cross_decoding()
    # plot_cross_decoding_scores()
    # convert_cross_decoding_scores_to_matlab()
    # load_and_filternan_models_with_csvs()
    # load_model_datas()
    plotting_histogram()