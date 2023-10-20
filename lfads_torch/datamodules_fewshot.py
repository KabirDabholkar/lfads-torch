from typing import Union

import h5py
import numpy as np

from lfads_torch.datamodules import (
    BasicDataModule,
    attach_tensors,
    reshuffle_train_valid,
)

# import pytorch_lightning as pl
# import torch


def split_datadict(data_dict, split_keep_first, num_new_heldout_neurons=None):
    part_to_move = int(split_keep_first)
    part_to_keep = 1 - part_to_move

    num_encod_neurons = data_dict["train_encod_data"].shape[2]
    num_recon_neurons = data_dict["train_recon_data"].shape[2]
    num_heldout_neurons = num_recon_neurons - num_encod_neurons
    if num_new_heldout_neurons is None:
        num_new_heldout_neurons = num_heldout_neurons // 2

    updates_dict = {}
    for k, v in data_dict.items():
        if "recon_data" in k:
            recon_key = k
            fewshot_key = k.replace("recon", "few-shot")
            encod_neurons, heldout_neurons = np.split(v, [num_encod_neurons], axis=2)
            order = 1 if split_keep_first else -1
            new_heldout_neurons, fewshot_neurons = np.split(
                heldout_neurons[::order], [num_new_heldout_neurons], axis=2
            )
            new_heldout_neurons, fewshot_neurons = (
                new_heldout_neurons[::order],
                fewshot_neurons[::order],
            )
            updates_dict[recon_key] = np.concatenate(
                [encod_neurons, new_heldout_neurons], axis=2
            )
            updates_dict[fewshot_key] = fewshot_neurons
    data_dict.update(updates_dict)

    # updates_dict = {}
    # for k, v in data_dict.items():
    #     if "recon_data" in k:
    #         k_ = k.replace("recon", "few-shot")
    #         split_at = (
    #             (v.shape[2] // 2)
    #             if split_keep_first
    #             else (v.shape[2] - v.shape[2] // 2)
    #         )
    #         parts = np.split(v, [split_at], axis=2)
    #         updates_dict[k] = parts[part_to_keep]
    #         updates_dict[k_] = parts[part_to_move]
    data_dict.update(updates_dict)
    for k, v in data_dict.items():
        print(k, v.shape)
    return data_dict


class DoubleHeldoutDataModule(BasicDataModule):
    def __init__(
        self,
        data_paths: list[str],
        batch_keys: list[str] = [],
        attr_keys: list[str] = [],
        batch_size: int = 64,
        reshuffle_tv_seed: int = None,
        reshuffle_tv_ratio: float = None,
        sv_rate: float = 0.0,
        sv_seed: int = 0,
        dm_ic_enc_seq_len: int = 0,
        split_keep_first: bool = True,
        num_new_heldout_neurons: Union[int, None] = None,
    ):
        super().__init__(
            data_paths,
            batch_keys=batch_keys,
            attr_keys=attr_keys,
            batch_size=batch_size,
            reshuffle_tv_seed=reshuffle_tv_seed,
            reshuffle_tv_ratio=reshuffle_tv_ratio,
            sv_rate=sv_rate,
            sv_seed=sv_seed,
            dm_ic_enc_seq_len=dm_ic_enc_seq_len,
        )
        self.save_hyperparameters()

    def setup(self, stage=None):
        hps = self.hparams
        data_dicts = []
        for data_path in hps.data_paths:
            # Load data arrays from the file
            with h5py.File(data_path, "r") as h5file:
                data_dict = {k: v[()] for k, v in h5file.items()}
            # Reshuffle the training / validation split
            if hps.reshuffle_tv_seed is not None:
                data_dict = reshuffle_train_valid(
                    data_dict, hps.reshuffle_tv_seed, hps.reshuffle_tv_ratio
                )
            split_datadict(
                data_dict,
                hps.split_keep_first,
                num_new_heldout_neurons=hps.num_new_heldout_neurons,
            )

            data_dicts.append(data_dict)
        # Attach data to the datamodule
        attach_tensors(self, data_dicts, extra_keys=hps.batch_keys)
        for attr_key in hps.attr_keys:
            setattr(self, attr_key, data_dict[attr_key])
