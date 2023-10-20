import logging
import shutil
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

from ..datamodules import reshuffle_train_valid
from ..tuples import SessionOutput
from ..utils import send_batch_to_device, transpose_lists

logger = logging.getLogger(__name__)


def run_fewshot_analysis(model, datamodule, trainer, num_samples=1):
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
    pred_dls = datamodule.predict_dataloader()
    # Set the model to evaluation mode
    model.eval()

    train_output = trainer.predict(model=model, dataloaders=train_dls)
    train_output

    print("test")

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