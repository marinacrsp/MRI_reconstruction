import os
import random
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from data_utils import *
from fastmri.data.subsample import EquiSpacedMaskFunc, RandomMaskFunc
from fastmri.data.transforms import tensor_to_complex_np, to_tensor
from torch.utils.data import Dataset


class KCoordDataset(Dataset):

    def __init__(
        self,
        path_to_data: Union[str, Path, os.PathLike],
        n_volumes: int,
        n_slices: int = 3,
        acceleration: int = 4,
        center_frac: float = 0.15,
        mode: str='train',
        epsilon: float=1.e-10,
    ):
        self.mode = mode
        self.metadata = {}
        self.dict_volumes = []
        self.volume_masks = []
        self.center_idx = []
        self.epsilon = epsilon
        path_to_data = Path(path_to_data)
        if path_to_data.is_dir():
            files = sorted(
                [
                    file
                    for file in path_to_data.iterdir()
                    if file.suffix == ".h5" and "AXT1POST_205" in file.name
                    # if file.suffix == ".h5" and "AXT2_205" in file.name # T2 sequence
                    
                ]
            )[:n_volumes]
        else:
            files = [path_to_data]

        # For each MRI volume in the dataset...
        for vol_id, file in enumerate(files):
            # Load MRI volume
            with h5py.File(file, "r") as hf:
                volume_kspace = to_tensor(preprocess_kspace(hf["kspace"][()]))[
                    :n_slices
                ]

            if n_slices == 1: # Remove redundant first coordinate
                volume_kspace = volume_kspace.squeeze(0)
            
        

            ##################################################
            # Mask creation
            ##################################################
            mask_func = EquiSpacedMaskFunc(
                center_fractions=[center_frac], accelerations=[acceleration]
            )

            shape = (2, volume_kspace.shape[-2], volume_kspace.shape[-3])
            
            mask, _ = mask_func(
                shape, None, vol_id
            )  # use the volume index as random seed.

            _, left_idx, right_idx = remove_center(mask)


            ##################################################
            # Computing the indices
            ##################################################
            n_coils, Ny, Nx = volume_kspace.shape[:-1]
            ky_ids = torch.arange(Ny)
            
            if self.mode == 'test': 
                self.mask = mask.bool().expand(shape).permute(0,2,1)
                    
            elif self.mode == 'train': # NOTE: No undersampling mask
                self.mask = torch.ones((Ny, Nx))
                    
            else:
                raise f'Error with input {mode} commands'            
                
            ##################################################
            # Computing the targets
            ##################################################
            img = rss(inverse_fft2_shift(tensor_to_complex_np(volume_kspace))) # Compute the mean image out of all the coils
            kspace = fft2_shift(img)

            phase = np.angle(kspace)
            modulus_log = np.log(np.abs(kspace) + self.epsilon) # NORMALIZE MODULUS
            quant_log = np.abs(np.quantile(modulus_log, 0.998))
            modulus_log = modulus_log / quant_log

            train_ks = torch.tensor(modulus_log * np.exp(1j*phase))           
            kspace_tensor = torch.stack([torch.real(train_ks), torch.imag(train_ks)], dim = -1)

            ks_toappend = kspace_tensor.float().permute(2,0,1) # nchannels, ny, nx
            self.dict_volumes.append(
                ks_toappend)
                        
            self.metadata[vol_id] = {
                "file": file,
                "mask": mask,
                "shape": (n_coils, Ny, Nx),
                "plot_cste": quant_log,
                "center": {
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                },
                "target": ks_toappend,
            }
                        
        self.dict_volumes = torch.stack(self.dict_volumes, dim=0) # n_volumes, ch_dim, ny, nx
        
    def __getitem__(self, idx):
        return self.dict_volumes[idx] * self.mask
    
    def get_mask(self):
        return self.mask

    def __len__(self):
        return len(self.dict_volumes)


def seed_worker(worker_id):
    """
    Controlling randomness in multi-process data loading. The RNGs are used by
    the RandomSampler to generate random indices for data shuffling.
    """
    # Use `torch.initial_seed` to access the PyTorch seed set for each worker.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
