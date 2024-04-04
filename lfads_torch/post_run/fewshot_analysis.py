import logging
import shutil
from pathlib import Path

import h5py
import torch
import numpy as np
from functools import partial

from tqdm import tqdm

from ..datamodules import reshuffle_train_valid
from ..utils import send_batch_to_device, transpose_lists

from ..tuples import SessionBatch, SessionOutput
from ..utils import transpose_lists
from ..metrics import bits_per_spike, regional_bits_per_spike

from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import warnings

import pytorch_lightning as pl

from torch import nn
from lfads_torch.modules.readin_readout import FanInLinear
from lfads_torch.model import LFADS

logger = logging.getLogger(__name__)


class LinearLightning(pl.LightningModule):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            lr_init: float,
            lr_adam_beta1: float,
            lr_adam_beta2: float,
            lr_adam_epsilon: float,
            weight_decay: float,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(in_features,out_features)

    def forward(self,inputs):
        logrates = self.model(inputs)
        return logrates  #torch.exp(inp)

    def configure_optimizers(self):
        hps = self.hparams
        # Create an optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hps.lr_init,
            betas=(hps.lr_adam_beta1, hps.lr_adam_beta2),
            eps=hps.lr_adam_epsilon,
            weight_decay=hps.weight_decay,
        )
        return optimizer

    def _general_step(self, batch, batch_idx):
        latents,target_spike_counts = batch
        pred_logrates = self(latents)
        return -bits_per_spike(pred_logrates,target_spike_counts)

    def training_step(self, batch, batch_idx):
        loss = self._general_step(batch, batch_idx)
        self.log("fewshot/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._general_step(batch,batch_idx)
        self.log("fewshot/validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

class FewshotLFADS(LFADS):
    def forward(
        self,
        batch: dict[SessionBatch],
        sample_posteriors: bool = False,
        output_means: bool = True,
    ) -> dict[SessionOutput]:
        with torch.no_grad():
            outputs = super().forward(batch,sample_posteriors=sample_posteriors,output_means=output_means)

        # Allow SessionBatch input
        if type(batch) == SessionBatch and len(self.readin) == 1:
            batch = {0: batch}
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Keep track of batch sizes so we can split back up
        batch_sizes = [len(batch[s].encod_data) for s in sessions]

        factors = [outputs[s].factors for s in sessions]
        # factors = torch.split(factors, batch_sizes)
        output_params = [self.readout[s](f) for s, f in zip(sessions, factors)]
        return output_params




def initialise_partial(partial_model,input_shape,output_shape):
    return partial_model(
        input_shape,
        output_shape
    )
def initialise_partial_many(partial_model_single,input_shape,output_shape):
    return [partial_model_single() for _ in range(output_shape)]

class FewshotTrainTest(pl.Callback):
    """
    Train and test (on validation set) the Fewshot head model.
    """

    def __init__(
            self,
            fewshot_head_model,
            fewshot_trainer,
            K: int,
            ratio: float = 0.3,
            seed: int = 0,
            log_every_n_epochs=20,
            fewshot_trainer_epochs: int = 50,
            use_recon_as_targets: bool = False,
            eval_type: str = 'valid',
            #decoding_cv_sweep=False,
        ):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        # self.decoding_cv_sweep = decoding_cv_sweep
        self.smth_metrics = {}

        self.fewshot_head_model_partial = fewshot_head_model
        self.fewshot_trainer = fewshot_trainer(gpus=int(torch.cuda.is_available()))
        self.K = K
        self.ratio = ratio
        self.seed = seed
        self.fewshot_dataloaders = None
        self.fewshot_trainer_epochs = fewshot_trainer_epochs
        self.use_recon_as_targets = use_recon_as_targets
        self.eval_type = eval_type

    def my_setup(self, trainer, pl_module, initialise_head: bool = True):
        datamodule = trainer.datamodule
        model = pl_module

        (input_data_sample, recon_data_sample, *_), _ = datamodule.valid_data[0]

        batches_train = model.batches_train
        batches_valid = model.batches_valid
        outputs_train = model.model_outputs_train
        outputs_valid = model.model_outputs_valid

        _, self.n_obs, self.n_heldin = input_data_sample.shape

        factors_train = torch.concat([t[0].factors for t in outputs_train])[:, :self.n_obs, :].detach()
        factors_valid = torch.concat([t[0].factors for t in outputs_valid])[:, :self.n_obs, :].detach()

        

        if self.use_recon_as_targets:
            recon_data = torch.concat([l[0][0].recon_data for l in list(batches_train)])
            fewshot_neurons_train = recon_data[..., :self.n_obs, :].detach()
            recon_data = torch.concat([l[0][0].recon_data for l in list(batches_valid)])
            fewshot_neurons_valid = recon_data[..., :self.n_obs, :].detach()
            self.target_name = "recon"
        else:
            fewshot_neurons_train = torch.concat([l[0][1][1] for l in list(batches_train)])[:, :self.n_obs, :].detach()
            fewshot_neurons_valid = torch.concat([l[0][1][1] for l in list(batches_valid)])[:, :self.n_obs, :].detach()
            self.target_name = "reallyheldout"
        self.true_validation_set = (factors_valid,fewshot_neurons_valid)

        train_samples = factors_train.shape[0]

        assert self.K<=train_samples

        X = factors_train[:self.K]
        Y = fewshot_neurons_train[:self.K]

        valid_size = int(self.ratio * X.shape[0])
        arrays = train_test_split(*[X, Y], test_size=valid_size, random_state=self.seed)
        self.X_train, self.Y_train = [a for i, a in enumerate(arrays) if (i - 1) % 2]
        self.X_val, self.Y_val = [a for i, a in enumerate(arrays) if i % 2]
        print('shapes', self.X_train.shape, self.Y_train.shape, self.X_val.shape, self.Y_val.shape)
        self.X_train, self.Y_train, self.X_val, self.Y_val = [
            tensor_.to(pl_module.device) for tensor_ in [self.X_train, self.Y_train, self.X_val, self.Y_val]
        ]

        if initialise_head:
            self.fewshot_head_model = self.fewshot_head_model_partial
            if isinstance(self.fewshot_head_model, partial):
                self.fewshot_head_model = self.fewshot_head_model(
                    factors_train.shape[-1],
                    fewshot_neurons_train.shape[-1]
                )

        if isinstance(self.fewshot_head_model,pl.LightningModule):
            fewshot_dataloader_train = DataLoader(
                TensorDataset(self.X_train, self.Y_train),
                batch_size=100
            )
            fewshot_dataloader_val = DataLoader(
                TensorDataset(self.X_val, self.Y_val),
                batch_size=100
            )
            self.fewshot_dataloaders = (fewshot_dataloader_train, fewshot_dataloader_val)
        else:
            self.X_train, self.Y_train, self.X_val, self.Y_val = [
                tensor_.to(pl_module.device) for tensor_ in [self.X_train, self.Y_train, self.X_val, self.Y_val]
            ]
            # self.X_train, self.Y_train, self.X_val, self.Y_val = [
            #     tensor_.to(
            #         pl_module.device
            #     ).reshape(
            #         -1,tensor_.shape[-1]
            #     ).detach().cpu().numpy() for tensor_ in [self.X_train, self.Y_train, self.X_val, self.Y_val]
            # ]
            self.fewshot_dataloaders = ((self.X_train,self.Y_train),(self.X_val,self.Y_val))

        if self.target_name == 'recon':
            y_pred = pl_module.readout[0](self.X_train)
            co_bps_train = bits_per_spike(y_pred,self.Y_train)

            pl_module.log_dict(
                {f'debugging/train_{self.K}shot_co_bps_{self.target_name}_truereadout': co_bps_train}
            )

            y_pred = pl_module.readout[0](self.X_val)
            co_bps_val = bits_per_spike(y_pred,self.Y_val)

            pl_module.log_dict(
                {f'debugging/val_{self.K}shot_co_bps_{self.target_name}_truereadout': co_bps_val}
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs best score k shot score at the end of the validation epoch.

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

        if not hasattr(pl_module,'model_outputs_train'):
            warnings.warn('Module has no attribute "model_latents_train".',RuntimeWarning)
            return

        if len(pl_module.model_outputs_train)==0:
            warnings.warn('"model_latents_train" is empty.',RuntimeWarning)
            return

        # if self.fewshot_dataloaders is None:
        #     self.my_setup(trainer, pl_module, initialise_head=True)
        # else:
        self.my_setup(trainer, pl_module, initialise_head=True)

        fewshot_train, fewshot_val = self.fewshot_dataloaders
        fewshot_head_model = self.fewshot_head_model

        print('Training few shot head...')

        if isinstance(fewshot_head_model,pl.LightningModule):
            # fewshot_head_model.fit = lambda train,val: self.fewshot_trainer(
            #     model=fewshot_head_model,
            #     train_dataloaders=fewshot_train,
            #     val_dataloaders=fewshot_val,
            # )
            self.fewshot_trainer.fit(
                model=fewshot_head_model,
                train_dataloaders = fewshot_train,
                val_dataloaders = fewshot_val,
            )
            self.fewshot_trainer.fit_loop.max_epochs += self.fewshot_trainer_epochs
        else: # fewshot_head_model is sklearn model
            fewshot_train = (f.reshape(-1,f.shape[-1]).detach().cpu().numpy() for f in fewshot_train)
            fewshot_head_model.fit(*fewshot_train)

        print('Done.\nTesting few shot head...')

        X_trueval,Y_trueval = self.true_validation_set

        if isinstance(fewshot_head_model,pl.LightningModule):
            # fewshot_head_model = fewshot_head_model.to(self.X_val.device)
            # pred = fewshot_head_model(self.X_val)
            fewshot_head_model = fewshot_head_model.to(X_trueval.device)
            pred = fewshot_head_model(X_trueval)
        else:
            # X_val, Y_val = fewshot_val
            # pred = fewshot_head_model.predict(X_val.reshape(-1,X_val.shape[-1]).detach().cpu().numpy())
            # pred = torch.tensor(pred.reshape(Y_val.shape),device=Y_val.device)
            # pred = torch.log(pred)
            pred = fewshot_head_model.predict(X_trueval.reshape(-1, X_trueval.shape[-1]).detach().cpu().numpy())
            pred = torch.tensor(pred.reshape(Y_trueval.shape),device=Y_trueval.device)
            pred = torch.log(pred)

        ### bps evaluated all recon neurons
        # valid_kshot_smoothing = bits_per_spike(pred, self.Y_val)
        valid_kshot_smoothing = bits_per_spike(pred, Y_trueval)
        head_module_name = '.'.join([self.fewshot_head_model.__class__.__module__,self.fewshot_head_model.__class__.__name__])
        pl_module.log_dict({f'valid/{self.K}shot_{head_module_name}_{self.target_name}_bps':valid_kshot_smoothing})

        if self.target_name=='recon':
            ### bps evaluated on the heldout subset of recon neurons, only if this is the recon set
            # valid_kshot_smoothing = bits_per_spike(pred[:,:,self.n_heldin:], self.Y_val[:,:,self.n_heldin:])
            valid_kshot_smoothing = bits_per_spike(pred[:, :, self.n_heldin:], Y_trueval[:, :, self.n_heldin:])
            head_module_name = '.'.join([self.fewshot_head_model.__class__.__module__,self.fewshot_head_model.__class__.__name__])
            pl_module.log_dict({f'{self.eval_type}/{self.K}shot_{head_module_name}_co_bps':valid_kshot_smoothing})


def run_fewshot_analysis(
        model,
        fewshot_head_model,
        datamodule,
        trainer,
        fewshot_trainer,
        K: list[int],
        ratio: float = 0.2,
        seed: int = 0,
    ):
    """Runs the model repeatedly to generate outputs for different samples
    of the posteriors. Averages these outputs and saves them to an output file.

    Parameters
    ----------
    model : lfads_torch.model.LFADS
        A trained LFADS model.
    datamodule : pytorch_lightning.LightningDataModule
        The `LightningDataModule` to pass through the `model`.
    filename : str
        The filename to use for saving output
    num_samples : int, optional
        The number of forward passes to average, by default 50
    """
    # Convert filename to pathlib.Path for convenience
    # filename = Path(filename)
    # Set up the dataloaders
    datamodule.setup()
    train_dls = datamodule.train_dataloader()
    # pred_dls = datamodule.predict_dataloader()

    # Set the model to evaluation mode
    model.eval()

    train_output = trainer.predict(model=model, dataloaders=train_dls)
    # train_dls[0][0]
    num_recon_neurons = list(train_dls)[0][0][0].recon_data.shape[-1]

    train_factors = torch.concat([t[0].factors for t in train_output])[:,:35,:]
    train_fewshot_neurons = torch.tensor(datamodule.train_fewshot_data)[:,:35,:]
    recon_data = torch.concat([l[0][0].recon_data for l in list(train_dls)])
    train_fewshot_neurons = torch.concat([train_fewshot_neurons,recon_data[...,:35,:]],axis=-1) #-23

    train_samples = train_factors.shape[0]
    for k in K:
        samples = np.random.choice(train_samples, size=k, replace=False)
        X = train_factors[samples]#.reshape(-1,train_factors.shape[-1])
        Y = train_fewshot_neurons[samples]#.reshape(-1,train_fewshot_neurons.shape[-1])

        valid_size = int(ratio * X.shape[0])
        arrays = train_test_split(*[X,Y], test_size=valid_size, random_state=seed)
        X_train,Y_train = [a for i, a in enumerate(arrays) if (i - 1) % 2]
        X_val,Y_val = [a for i, a in enumerate(arrays) if i % 2]
        print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)
        fewshot_dataloader_train = DataLoader(
            TensorDataset(X_train,Y_train),
            batch_size=100
        )
        fewshot_dataloader_val = DataLoader(
            TensorDataset(X_val,Y_val),
            batch_size=100
        )

        fewshot_head_model = fewshot_head_model(
            train_factors.shape[-1],
            train_fewshot_neurons.shape[-1]
        )

        fewshot_trainer.fit(
            model=fewshot_head_model,
            train_dataloaders = fewshot_dataloader_train,
            val_dataloaders = fewshot_dataloader_val,
        )
        output = fewshot_head_model(train_factors[:30])
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,2)
        ax = axs[0]
        im = ax.imshow(output[2].detach().cpu().T)
        plt.colorbar(im, ax=ax)
        ax = axs[1]
        im = plt.imshow(train_fewshot_neurons[2].detach().cpu().T)
        plt.colorbar(im, ax=ax)
        plt.savefig('/Users/kabir/Documents/code/lfads-torch/test_output.png')

    bits_per_spike(fewshot_head_model(X_val),Y_val)
    # model = FewshotLFADS()
    # trainer.fit()

    # print(model.readout[0])
    # model.readout_old = model.readout
    # model.few_shot_readout = FanInLinear(model.readout[0].in_features,num_recon_neurons)
    # model.readout = nn.ModuleList(modules=[FanInLinear(model.readout[0].in_features, num_recon_neurons)])

    # def forward(self,input):
    #     output = super().forward(self,input)
    #     return self.few_shot_readout(output.factors)
    #
    # model.forward = forward



    #print(model(list(train_dls)[0][0][0]))

    # trainer.fit(model=model,datamodule=datamodule)

    #
    # def run_ps_batch(s, batch):
    #     # Move the batch to the model device
    #     batch = send_batch_to_device({s: batch}, model.device)
    #     # Repeatedly compute the model outputs for this batch
    #     for i in range(num_samples):
    #         # Perform the forward pass through the model
    #         output = model.predict_step(batch, None, sample_posteriors=True)[s]
    #         # Use running sum to save memory while averaging
    #         if i == 0:
    #             # Detach output from the graph to save memory on gradients
    #             sums = [o.detach() for o in output]
    #         else:
    #             sums = [s + o.detach() for s, o in zip(sums, output)]
    #     # Finish averaging by dividing by the total number of samples
    #     return [s / num_samples for s in sums]
    #
    # for s, dataloaders in train_dls.items():
    #     # # Give each session a unique file path
    #     # sess_fname = f"{filename.stem}_sess{s}{filename.suffix}"
    #     # # Copy data file for easy access to original data and indices
    #     # dhps = datamodule.hparams
    #     # if dhps.reshuffle_tv_seed is not None:
    #     #     # If the data was shuffled, shuffle it when copying
    #     #     with h5py.File(dhps.data_paths[s]) as h5file:
    #     #         data_dict = {k: v[()] for k, v in h5file.items()}
    #     #     data_dict = reshuffle_train_valid(
    #     #         data_dict, dhps.reshuffle_tv_seed, dhps.reshuffle_tv_ratio
    #     #     )
    #     #     with h5py.File(sess_fname, "w") as h5file:
    #     #         for k, v in data_dict.items():
    #     #             h5file.create_dataset(k, data=v)
    #     # else:
    #     #     shutil.copyfile(datamodule.hparams.data_paths[s], sess_fname)
    #     for split in dataloaders.keys():
    #         # Compute average model outputs for each session and then recombine batches
    #         logger.info(f"Running few-shot protocol on Session {s} {split} data.")
    #         with torch.no_grad():
    #             post_means = [
    #                 run_ps_batch(s, batch) for batch in tqdm(dataloaders[split])
    #             ]
    #         post_means = SessionOutput(
    #             *[torch.cat(o).cpu().numpy() for o in transpose_lists(post_means)]
    #         )
    #         # # Save the averages to the output file
    #         # with h5py.File(sess_fname, mode="a") as h5file:
    #         #     for name in SessionOutput._fields:
    #         #         h5file.create_dataset(
    #         #             f"{split}_{name}", data=getattr(post_means, name)
    #         #         )
    #
