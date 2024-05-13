import logging
import os
import warnings
from glob import glob
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf, open_dict
from ray import tune
from functools import partial
from typing import Optional
import pickle as pkl
from lfads_torch.post_run.nlb_fewshot import run_nlb_fewshot, run_model_on_numpy
from nlb_tools.load_and_save_latents import run_nlb_evaluation_protocol, run_fewshot_given_latents
from nlb_tools import sklearn_glm 
import numpy as np
import pandas as pd

from .utils import flatten

OmegaConf.register_new_resolver("relpath", lambda p: Path(__file__).parent / ".." / p)
OmegaConf.register_new_resolver("eval", eval)


glms_funcs = {
    # 'sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=500)'  : partial(sklearn_glm.fit_poisson_parallel,alpha=0.0,max_iter=500),
    # 'sklearn_glm.fit_poisson_parallel(alpha=0.0,max_iter=1000)' : partial(sklearn_glm.fit_poisson_parallel,alpha=0.0,max_iter=1000),
    # 'sklearn_glm.fit_poisson_parallel(alpha=0.01,max_iter=500)'  : partial(sklearn_glm.fit_poisson_parallel,alpha=0.1,max_iter=500),
    'sklearn_glm.fit_poisson_parallel(alpha=0.01,max_iter=500)'  : partial(sklearn_glm.fit_poisson_parallel,alpha=0.01,max_iter=500),
    'sklearn_glm.fit_poisson_parallel(alpha=0.1,max_iter=500)'  : partial(sklearn_glm.fit_poisson_parallel,alpha=0.1,max_iter=500),
    # 'sklearn_glm.fit_poisson_parallel(alpha=0.5,max_iter=500)'  : partial(sklearn_glm.fit_poisson_parallel,alpha=0.5,max_iter=500),
}

def load_model(model,config_path,run_dir,trial_ids=None,load_best=True):
    if "single" not in str(config_path) and run_dir:
        tune_trial_name = tune.get_trial_name()
        tune_trial_name_number = str(tune_trial_name).split('_')[-1]

        # runs = os.listdir( run_dir )
        # runs = [r for r in runs if 'run_model' in r]
        print(trial_ids)
        run_name = 'run_model_' + [r for r in trial_ids if tune_trial_name_number in r][0]
        checkpoint_dir = Path(run_dir) / run_name / 'lightning_checkpoints'
    elif run_dir is None:
        checkpoint_dir = Path(tune.get_trial_dir()) / 'lightning_checkpoints'

    print('checkpoint_dir',checkpoint_dir)

    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
        if load_best:
            ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
            print(ckpt_pattern)
            ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    return model,checkpoint_dir

def wrap_run_model_on_numpy(model,spikes_heldin):
    spikes_heldin_reallyheldoutremoved = spikes_heldin[:,:,22:]
    pred_full , latents = run_model_on_numpy(model,spikes_heldin_reallyheldoutremoved)
    print(pred_full.shape,latents.shape)
    pred_full = np.concatenate([np.zeros_like(pred_full)[:,:,:22],pred_full],axis=-1)
    return pred_full, latents


def run_model(
    overrides: dict = {},
    checkpoint_dir: str = None,
    config_path: str = "../configs/single.yaml",
    do_train: bool = True,
    do_posterior_sample: bool = True,
    do_fewshot_protocol: bool = True,
    do_post_run_analysis: bool = True,
    do_nlb_fewshot: bool = False,
    do_nlb_fewshot2: bool = False,
    variant : str = 'mc_maze_20',
    run_dir: Optional[os.PathLike] = None,
    trial_ids: Optional[list[str]] = None,
    load_best: bool = True,
):
    """Adds overrides to the default config, instantiates all PyTorch Lightning
    objects from config, and runs the training pipeline.
    """

    # Compose the train config with properly formatted overrides
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    with hydra.initialize(
        config_path=config_path.parent,
        job_name="run_model",
        version_base="1.1",
    ):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)

    print(OmegaConf.to_yaml(config))

    # Avoid flooding the console with output during multi-model runs
    if config.ignore_warnings:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Instantiate `LightningDataModule` and `LightningModule`
    datamodule = instantiate(config.datamodule, _convert_="all")
    model = instantiate(config.model)

    # If `checkpoint_dir` is passed, find the most recent checkpoint in the directory
    if checkpoint_dir:
        ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)

    if do_train:
        # If both ray.tune and wandb are being used, ensure that loggers use same name
        if "single" not in str(config_path) and "wandb_logger" in config.logger:
            with open_dict(config):
                config.logger.wandb_logger.name = tune.get_trial_name()
                config.logger.wandb_logger.id = tune.get_trial_name()
                print('tune trial name',tune.get_trial_name())
        callbacks = [instantiate(c) for c in config.callbacks.values()]
        print(callbacks)
        # Instantiate the pytorch_lightning `Trainer` and its callbacks and loggers
        trainer = instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
        )
        # Temporary workaround for PTL step-resuming bug
        if checkpoint_dir:
            ckpt = torch.load(ckpt_path)
            trainer.fit_loop.epoch_loop._batches_that_stepped = ckpt["global_step"]
        # Train the model
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if checkpoint_dir else None,
        )
        # Restore the best checkpoint if necessary - otherwise, use last checkpoint
        if config.posterior_sampling.use_best_ckpt:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    else:
        if checkpoint_dir:
            # If not training, restore model from the checkpoint
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])

    # Run the posterior sampling function
    if do_posterior_sample:
        if torch.cuda.is_available():
            model = model.to("cuda")
        call(config.posterior_sampling.fn, model=model, datamodule=datamodule)

    # ### Run few shot
    # if do_fewshot_protocol:
    #     if "single" not in str(config_path) and run_dir:
    #         tune_trial_name = tune.get_trial_name()
    #         tune_trial_name_number = str(tune_trial_name).split('_')[-1]

    #         # runs = os.listdir( run_dir )
    #         # runs = [r for r in runs if 'run_model' in r]
    #         print(trial_ids)
    #         run_name = 'run_model_' + [r for r in trial_ids if tune_trial_name_number in r][0]
    #         checkpoint_dir = Path(run_dir) / run_name / 'lightning_checkpoints'

    #     print('checkpoint_dir',checkpoint_dir)

    #     if checkpoint_dir:
    #         ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    #         if load_best:
    #             ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
    #             ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)
    #         model.load_state_dict(torch.load(ckpt_path)["state_dict"])

    
    #     trainer = instantiate(
    #         config.trainer,
    #         callbacks=[instantiate(c) for c in config.callbacks.values()],
    #         logger=[instantiate(lg) for lg in config.logger.values()],
    #         gpus=int(torch.cuda.is_available()),
    #     )
    
    #     fewshot_head_model = instantiate(
    #         config.fewshot_head_model_lightning
    #     )
    #     fewshot_trainer = instantiate(
    #         config.fewshot_trainer
    #     )
    #     # print(type(fewshot_head_model(1,1)))
    #     call(
    #         config.fewshot_protocol.fn,
    #         model=model,
    #         fewshot_head_model=fewshot_head_model,
    #         datamodule=datamodule,
    #         trainer=trainer,
    #         fewshot_trainer=fewshot_trainer,
    #         K = [1000]
    #     )

    if do_post_run_analysis:
        if "single" not in str(config_path) and run_dir:
            tune_trial_name = tune.get_trial_name()
            tune_trial_name_number = str(tune_trial_name).split('_')[-1]

            # runs = os.listdir( run_dir )
            # runs = [r for r in runs if 'run_model' in r]
            print(trial_ids)
            run_name = 'run_model_' + [r for r in trial_ids if tune_trial_name_number in r][0]
            checkpoint_dir = Path(run_dir) / run_name / 'lightning_checkpoints'
        elif run_dir is None:
            checkpoint_dir = Path(tune.get_trial_dir()) / 'lightning_checkpoints'

        print('checkpoint_dir',checkpoint_dir)

        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
            if load_best:
                ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
                print(ckpt_pattern)
                ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])

        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.post_run_analysis_callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
        )
        trainer.fit_loop.max_epochs = 1
        datamodule = instantiate(config.datamodule, _convert_="all")

        def training_step(self, batch, batch_idx):
            opt = self.optimizers()
            opt.zero_grad()
            loss = self._shared_step(batch, batch_idx, "train")
            # self.manual_backward(loss)
            opt.step()
            return None

        model.training_step = partial(training_step,model)

        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

        # print('model_outputs_valid',model.model_outputs_valid)
        # print('model_outputs_valid.factors.shape',torch.concat([d.factors for d in model.model_outputs_valid]).shape)
        outputs = model.model_outputs_valid
        factors = torch.concat([o[0].factors for o in outputs])
        print(factors.shape)
        torch.save(
            factors,
            Path(tune.get_trial_dir()) / 'model_outputs_valid'
        )

    if do_nlb_fewshot:
        model,checkpoint_dir = load_model(model,config_path,run_dir,trial_ids=trial_ids,load_best=load_best)
        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.post_run_analysis_callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
        )
        trainer.fit_loop.max_epochs = 1
        df = run_nlb_fewshot(model,variant=variant)
        df['path'] = checkpoint_dir
        df.to_csv(Path(tune.get_trial_dir()) / 'results_old.csv')

    if do_nlb_fewshot2:
        model,checkpoint_dir = load_model(model,config_path,run_dir,trial_ids=trial_ids,load_best=load_best)
        results_df,results_dict,latents_dict, output_dict = run_nlb_evaluation_protocol(
            model=model,
            run_model_on_numpy_pre=wrap_run_model_on_numpy, #run_model_on_numpy,
            variant=variant,
            do_fewshot=True,
            do_evaluation=False
        )
        DFs = []
        for fit_poisson_func_name, fit_poisson_func in glms_funcs.items():
            results_df,results_dict = run_fewshot_given_latents(
                latents_dict=latents_dict,
                variant=variant,
                output_dict=output_dict,
                fit_poisson_func=fit_poisson_func)
            # results_df['fewshot_code'] = fit_poisson_func_name
            results_df = results_df.rename(columns=lambda x: ':'.join([x,fit_poisson_func_name]) if 'shot' in x else x)
            DFs.append(results_df)
        result_dfs = pd.concat(DFs,axis=1).reset_index()
        result_dfs = result_dfs.loc[:, ~result_dfs.columns.duplicated()]
        savepath = Path(tune.get_trial_dir()) / 'results_new.csv'
        # savepath = ckpt_path.replace('.pth','_results6.csv')
        result_dfs.to_csv(savepath)
        