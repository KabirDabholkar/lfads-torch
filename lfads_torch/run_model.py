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

from .utils import flatten

OmegaConf.register_new_resolver("relpath", lambda p: Path(__file__).parent / ".." / p)
OmegaConf.register_new_resolver("eval", eval)


def run_model(
    overrides: dict = {},
    checkpoint_dir: str = None,
    config_path: str = "../configs/single.yaml",
    do_train: bool = True,
    do_posterior_sample: bool = True,
    do_fewshot_protocol: bool = True,
    post_run_analysis  : bool = True,
    run_dir: Optional[os.PathLike] = None,
    trial_ids: Optional[list[str]] = None,
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

    # Run few shot
    # if do_fewshot_protocol:
    #     # Temporary workaround for PTL step-resuming bug
    #     if checkpoint_dir:
    #         ckpt = torch.load(ckpt_path)
    #         trainer.fit_loop.epoch_loop._batches_that_stepped = ckpt["global_step"]
    #
    #     trainer = instantiate(
    #         config.trainer,
    #         callbacks=[instantiate(c) for c in config.callbacks.values()],
    #         logger=[instantiate(lg) for lg in config.logger.values()],
    #         gpus=int(torch.cuda.is_available()),
    #     )
    #
    #     fewshot_head_model = instantiate(
    #         config.fewshot_head_model
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

    if post_run_analysis:
        if "single" not in str(config_path) and run_dir:
            tune_trial_name = tune.get_trial_name()
            tune_trial_name_number = str(tune_trial_name).split('_')[-1]

            # runs = os.listdir( run_dir )
            # runs = [r for r in runs if 'run_model' in r]
            run_name = 'run_model_' + [r for r in trial_ids if tune_trial_name_number in r][0]
            checkpoint_dir = Path(run_dir) / run_name / 'lightning_checkpoints'

        print('checkpoint_dir',checkpoint_dir)

        if checkpoint_dir:
            ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
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
