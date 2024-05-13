import warnings

import pytorch_lightning as pl
import torch
from nlb_tools.evaluation import (
    bits_per_spike,
    eval_psth,
    speed_tp_correlation,
    velocity_decoding,
)
from scipy.linalg import LinAlgWarning

from lfads_torch.metrics import ExpSmoothedMetric
from lfads_torch.utils import send_batch_to_device

import matplotlib.pyplot as plt



class NLBEvaluation(pl.Callback):
    """Computes and logs all evaluation metrics for the Neural Latents
    Benchmark to tensorboard. These include `co_bps`, `fp_bps`,
    `behavior_r2`, `psth_r2`, and `tp_corr`.

    To enable this functionality, install nlb_tools
    (https://github.com/neurallatents/nlb_tools).
    """

    def __init__(self, log_every_n_epochs=20, decoding_cv_sweep=False):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        decoding_cv_sweep : bool, optional
            Whether to run a cross-validated hyperparameter sweep to
            find optimal regularization values, by default False
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.decoding_cv_sweep = decoding_cv_sweep
        self.smth_metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get the dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        s = 0
        val_dataloader = pred_dls[s]["valid"]
        train_dataloader = pred_dls[s]["train"]
        # Create object to store evaluation metrics
        metrics = {}
        # Get entire validation dataset from datamodule
        # (input_data, recon_data, *_), (behavior,) = trainer.datamodule.valid_data[s]
        sessbatch, (behavior,*_) = trainer.datamodule.valid_data[s]
        input_data = sessbatch.encod_data
        recon_data = sessbatch.recon_data
        recon_data = recon_data.detach().cpu().numpy()
        behavior = behavior.detach().cpu().numpy()
        # Pass the data through the model
        # TODO: Replace this with Trainer.predict? Hesitation is that switching to
        # Trainer.predict for posterior sampling is inefficient because we can't
        # tell it how many forward passes to use.
        rates = []
        # for batch in val_dataloader:
        #     batch = send_batch_to_device({s: batch}, pl_module.device)
        #     output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
        #     rates.append(output.output_params)
        model = pl_module
        batches = val_dataloader
        # print('batches[0] in nlb',list(batches)[0])
        outputs = [model.predict_step(
            send_batch_to_device({0: batch}, pl_module.device),0,sample_posteriors=False)
            for batch in batches]
        rates = [o[0].output_params.detach() for o in outputs]
        rates = torch.cat(rates).detach().cpu().numpy()
        factors = [o[0].factors.detach() for o in outputs]
        factors = torch.cat(factors).detach().cpu().numpy()
        # from sklearn.decomposition import PCA
        # n_components = 2
        # P = PCA(n_components=n_components)
        # factors_proj =P.fit_transform(factors.reshape(-1,factors.shape[-1])).reshape(*factors.shape[0:2],n_components)
        # fig,ax = plt.subplots()
        # for i in range(10):
        #     ax.plot(factors_proj[i,:,0],factors_proj[i,:,1])
        # fig.savefig('/home/kabird/lfads-torch-fewshot-benchmark/plots/PCA_nlb.png',dpi=300)


        # Compute co-smoothing bits per spike
        _, n_obs, n_heldin = input_data.shape
        heldout = recon_data[:, :n_obs, n_heldin:]
        rates_heldout = rates[:, :n_obs, n_heldin:]
        # import numpy as np
        # heldin = np.array(input_data)
        # fig,axs= plt.subplots(1,3)
        # ax = axs[0]
        # ax.imshow(heldin[0])
        # ax = axs[1]
        # ax.imshow(rates_heldout[0])
        # ax = axs[2]
        # ax.imshow(heldout[0])
        # fig.savefig('/home/kabird/lfads-torch-fewshot-benchmark/plots/test_nlb_heldout.png')
        
        co_bps = bits_per_spike(rates_heldout, heldout)
        metrics["nlb/co_bps"] = max(co_bps, -1.0)
        # print("nlb/co_bps",metrics["nlb/co_bps"],'trials:',heldout.shape[0])
        # Compute forward prediction bits per spike
        forward = recon_data[:, n_obs:]
        rates_forward = rates[:, n_obs:]
        fp_bps = bits_per_spike(rates_forward, forward)
        metrics["nlb/fp_bps"] = max(fp_bps, -1.0)
        # Get relevant training dataset from datamodule
        _, (train_behavior,*_) = trainer.datamodule.train_data[s]
        train_behavior = train_behavior.detach().cpu().numpy()
        # Get model predictions for the training dataset
        train_rates = []
        for batch in train_dataloader:
            batch = send_batch_to_device({s: batch}, pl_module.device)
            output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
            train_rates.append(output.output_params)
        train_rates = torch.cat(train_rates).detach().cpu().numpy()
        # Get firing rates for observed time points
        rates_obs = rates[:, :n_obs]
        train_rates_obs = train_rates[:, :n_obs]
        # Compute behavioral decoding performance
        if behavior.ndim < 3:
            tp_corr = speed_tp_correlation(heldout, rates_obs, behavior)
            metrics["nlb/tp_corr"] = tp_corr
        else:
            with warnings.catch_warnings():
                # Ignore LinAlgWarning from early in training
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                behavior_r2 = velocity_decoding(
                    train_rates_obs,
                    train_behavior,
                    trainer.datamodule.train_decode_mask,
                    rates_obs,
                    behavior,
                    trainer.datamodule.valid_decode_mask,
                    self.decoding_cv_sweep,
                )
            metrics["nlb/behavior_r2"] = max(behavior_r2, -1.0)
        # # Compute PSTH reconstruction performance
        # if hasattr(trainer.datamodule, "psth"):
        #     psth = trainer.datamodule.psth
        #     cond_idxs = trainer.datamodule.valid_cond_idx
        #     jitter = getattr(trainer.datamodule, "valid_jitter", None)
        #     psth_r2 = eval_psth(psth, rates_obs, cond_idxs, jitter)
        #     metrics["nlb/psth_r2"] = max(psth_r2, -1.0)
        # Compute smoothed metrics
        for k, v in metrics.items():
            if k not in self.smth_metrics:
                self.smth_metrics[k] = ExpSmoothedMetric(coef=0.7)
            self.smth_metrics[k].update(v, 1)
        # Log actual and smoothed metrics
        pl_module.log_dict(
            {
                **metrics,
                **{k + "_smth": m.compute() for k, m in self.smth_metrics.items()},
            }
        )
        # Reset the smoothed metrics (per-step aggregation not necessary)
        [m.reset() for m in self.smth_metrics.values()]


import hydra
from pathlib import Path
from utils import flatten
from omegaconf import OmegaConf
from functools import partial
from hydra.utils import call, instantiate
def test_nlb_callback(
    nlb_callback,
    overrides: dict = {},
    config_path: str = "../../configs/multi_few_shot_original_heldout_mc_maze.yaml",
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

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Instantiate `LightningDataModule` and `LightningModule`
    datamodule = instantiate(config.datamodule, _convert_="all")
    # trainer = instantiate(config.trainer)
    model = instantiate(config.model)
    trainer = instantiate(
            config.trainer,
            callbacks=[callback],
            # logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
        )
    trainer.fit_loop.max_epochs = 1
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


    # callback.on_validation_epoch_end(trainer,model)

if __name__=="__main__":
    OmegaConf.register_new_resolver("relpath", lambda p: Path(__file__).parent / "../../" / p)
    OmegaConf.register_new_resolver("eval", eval)
    callback = NLBEvaluation()
    
    test_nlb_callback(callback)

    

