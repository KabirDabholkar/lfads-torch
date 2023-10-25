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
from ..metrics import bits_per_spike

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
            weight_decay: float
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
        return self._general_step(batch,batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._general_step(batch,batch_idx)

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

    train_factors = torch.concat([t[0].factors for t in train_output])
    train_fewshot_neurons = torch.tensor(datamodule.train_fewshot_data)

    train_samples = train_factors.shape[0]
    for k in K:
        samples = np.random.choice(train_samples, size=k, replace=False)
        X = train_factors[samples].reshape(-1,train_factors.shape[-1])
        Y = train_fewshot_neurons[samples].reshape(-1,train_fewshot_neurons.shape[-1])

        valid_size = int(ratio * X.shape[0])
        arrays = train_test_split(*[X,Y], test_size=valid_size, random_state=seed)
        X_train,Y_train = [a for i, a in enumerate(arrays) if (i - 1) % 2]
        X_val,Y_val = [a for i, a in enumerate(arrays) if i % 2]

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
