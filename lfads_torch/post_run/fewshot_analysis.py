import logging
import shutil
from pathlib import Path

import h5py
import torch
import numpy as np

from tqdm import tqdm

from ..datamodules import reshuffle_train_valid
from ..utils import send_batch_to_device, transpose_lists

from ..tuples import SessionBatch, SessionOutput
from ..utils import transpose_lists
from ..metrics import bits_per_spike, regional_bits_per_spike

from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split

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

    def my_setup(self, trainer, pl_module, initialise_head: bool = True):
        datamodule = trainer.datamodule
        model = pl_module

        # datamodule.setup()
        train_dls = datamodule.train_dataloader()
        # pred_dls = datamodule.predict_dataloader()

        # Set the model to evaluation mode

        #train_output = [model.predict_step(batch,i) for i,batch in enumerate(train_dls)]
        train_output = model.model_latents_train

        #train_output = trainer.predict(model=model, dataloaders=train_dls)
        # train_dls[0][0]
        num_recon_neurons = list(train_dls)[0][0][0].recon_data.shape[-1]

        train_factors = torch.concat([t[0].factors for t in train_output])[:, :35, :]
        train_fewshot_neurons = torch.tensor(datamodule.train_fewshot_data)[:, :35, :]
        # recon_data = torch.concat([l[0][0].recon_data for l in list(train_dls)])
        # train_fewshot_neurons = torch.concat([train_fewshot_neurons, recon_data[..., :35, :]], axis=-1)  # -23

        train_samples = train_factors.shape[0]
        k = self.K
        samples = np.random.choice(train_samples, size=k, replace=False)
        X = train_factors[samples]  # .reshape(-1,train_factors.shape[-1])
        Y = train_fewshot_neurons[samples]  # .reshape(-1,train_fewshot_neurons.shape[-1])

        valid_size = int(self.ratio * X.shape[0])
        arrays = train_test_split(*[X, Y], test_size=valid_size, random_state=self.seed)
        self.X_train, self.Y_train = [a for i, a in enumerate(arrays) if (i - 1) % 2]
        self.X_val, self.Y_val = [a for i, a in enumerate(arrays) if i % 2]
        #print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
        self.X_train, self.Y_train, self.X_val, self.Y_val = [
            tensor_.to(pl_module.device) for tensor_ in [self.X_train, self.Y_train, self.X_val, self.Y_val]
        ]
        fewshot_dataloader_train = DataLoader(
            TensorDataset(self.X_train, self.Y_train),
            batch_size=100
        )
        fewshot_dataloader_val = DataLoader(
            TensorDataset(self.X_val, self.Y_val),
            batch_size=100
        )
        self.fewshot_dataloaders = (fewshot_dataloader_train, fewshot_dataloader_val)

        if initialise_head:
            self.fewshot_head_model = self.fewshot_head_model_partial(
                train_factors.shape[-1],
                train_fewshot_neurons.shape[-1]
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

        if not hasattr(pl_module,'model_latents_train'):
            print('Module has no attribute "model_latents_train".')
            return

        if self.fewshot_dataloaders is None:
            self.my_setup(trainer, pl_module, initialise_head=True)
        else:
            self.my_setup(trainer, pl_module, initialise_head=True)



        fewshot_train, fewshot_val = self.fewshot_dataloaders
        fewshot_head_model = self.fewshot_head_model



        print('Training few shot head...')
        self.fewshot_trainer.fit(
            model=fewshot_head_model,
            train_dataloaders = fewshot_train,
            val_dataloaders = fewshot_val,
        )

        self.fewshot_trainer.fit_loop.max_epochs += self.fewshot_trainer_epochs

        print('Done.\nTesting few shot head...')
        fewshot_head_model = fewshot_head_model.to(self.X_val.device)
        valid_kshot_smoothing = bits_per_spike(fewshot_head_model(self.X_val), self.Y_val)

        pl_module.log_dict({f'valid/{self.K}_shot_cosmoothing':valid_kshot_smoothing})


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
    print(train_fewshot_neurons.mean())
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
