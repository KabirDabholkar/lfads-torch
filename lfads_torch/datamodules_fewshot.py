from typing import Union

import h5py
import numpy as np

from lfads_torch.datamodules import (
    BasicDataModule,
    attach_tensors,
    reshuffle_train_valid,
    MANDATORY_KEYS,
    to_tensor,
    SessionBatch,
    SessionDataset
)

# import pytorch_lightning as pl
import torch

MANDATORY_KEYS = {
    "train": ["encod_data", "recon_data","fewshot_data"],
    "valid": ["encod_data", "recon_data","fewshot_data"],
    "test": ["encod_data"],
}

##### overwriting attach tensors to include fewshot neurons #####
def attach_tensors(datamodule, data_dicts: list[dict], extra_keys: list[str] = []):
    hps = datamodule.hparams
    sv_gen = torch.Generator().manual_seed(hps.sv_seed)
    all_train_data, all_valid_data, all_test_data = [], [], []
    for data_dict in data_dicts:

        def create_session_batch(prefix, extra_keys=[]):
            # Ensure that the data dict has all of the required keys
            assert all(f"{prefix}_{key}" in data_dict for key in MANDATORY_KEYS[prefix])
            # Load the encod_data
            encod_data = to_tensor(data_dict[f"{prefix}_encod_data"])
            n_samps, n_steps, _ = encod_data.shape
            # Load the recon_data
            if f"{prefix}_recon_data" in data_dict:
                recon_data = to_tensor(data_dict[f"{prefix}_recon_data"])
            else:
                recon_data = torch.zeros(n_samps, 0, 0)
            if f"{prefix}_fewshot_data" in data_dict:
                fewshot_data = to_tensor(data_dict[f"{prefix}_fewshot_data"])
            else:
                fewshot_data = torch.zeros(n_samps, 0, 0)
            if hps.sv_rate > 0:
                # Create sample validation mask # TODO: Sparse and use complement?
                bern_p = 1 - hps.sv_rate if prefix != "test" else 1.0
                sv_mask = (
                    torch.rand(encod_data.shape, generator=sv_gen) < bern_p
                ).float()
            else:
                # Create a placeholder tensor
                sv_mask = torch.ones(n_samps, 0, 0)
            # Load or simulate external inputs
            if f"{prefix}_ext_input" in data_dict:
                ext_input = to_tensor(data_dict[f"{prefix}_ext_input"])
            else:
                ext_input = torch.zeros(n_samps, n_steps, 0)
            if f"{prefix}_truth" in data_dict:
                # Load or simulate ground truth TODO: use None instead of NaN?
                cf = data_dict["conversion_factor"]
                truth = to_tensor(data_dict[f"{prefix}_truth"]) / cf
            else:
                truth = torch.full((n_samps, 0, 0), float("nan"))
            # Remove unnecessary data during IC encoder segment
            sv_mask = sv_mask[:, hps.dm_ic_enc_seq_len :]
            ext_input = ext_input[:, hps.dm_ic_enc_seq_len :]
            truth = truth[:, hps.dm_ic_enc_seq_len :, :]
            # Extract data for any extra keys
            other = [to_tensor(data_dict[f"{prefix}_{k}"]) for k in extra_keys]
            return (
                SessionBatch(
                    encod_data=encod_data,
                    recon_data=recon_data,
                    ext_input=ext_input,
                    truth=truth,
                    sv_mask=sv_mask,
                ),
                tuple(other),
            )
        print('extra_keys',extra_keys)
        # Store the data for each session
        all_train_data.append(create_session_batch("train", extra_keys))
        all_valid_data.append(create_session_batch("valid", extra_keys))
        if "test_encod_data" in data_dict:
            all_test_data.append(create_session_batch("test"))
    # Store the datasets on the datamodule
    datamodule.train_data = all_train_data
    datamodule.train_ds = [SessionDataset(*train_data) for train_data in all_train_data]
    datamodule.valid_data = all_valid_data
    datamodule.valid_ds = [SessionDataset(*valid_data) for valid_data in all_valid_data]
    if len(all_test_data) == len(all_train_data):
        datamodule.test_data = all_test_data
        datamodule.test_ds = [SessionDataset(*test_data) for test_data in all_test_data]



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
            fewshot_key = k.replace("recon", "fewshot")
            encod_neurons, heldout_neurons = np.split(v, [num_encod_neurons], axis=2)
            # heldout_neurons = heldout_neurons[...,np.random.permutation(heldout_neurons.shape[-1])]
            order = 1 if split_keep_first else -1
            new_heldout_neurons, fewshot_neurons = np.split(
                heldout_neurons[...,::order], [num_new_heldout_neurons], axis=2
            )
            new_heldout_neurons, fewshot_neurons = (
                new_heldout_neurons[...,::order],
                fewshot_neurons[...,::order],
            )
            updates_dict[recon_key] = np.concatenate(
                [encod_neurons, new_heldout_neurons], axis=2
            )
            updates_dict[fewshot_key] = fewshot_neurons
    if 'psth' in data_dict.keys():
        num_new_recon_neurons = updates_dict['train_recon_data'].shape[-1] #num_recon_neurons - num_new_heldout_neurons
        updates_dict['psth'] = data_dict['psth'][...,:num_new_recon_neurons]
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
        print('hps:',hps)
        # Attach data to the datamodule
        attach_tensors(self, data_dicts, extra_keys=hps.batch_keys)
        for attr_key in hps.attr_keys:
            setattr(self, attr_key, data_dict[attr_key])
