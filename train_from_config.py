# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Hook for training non-GP model with config file"""
"""
python train_from_config.py -cd configs/lm_gp_configs/ -cn train_exact_gp_pca_14H.yaml
"""
import hydra
from random import randint
from src import train
import torch

config_dir = '/mnt/jimmyxu/nfs_share2/Documents/biotransfer/configs/lm_gp_configs'
hydra.initialize_config_dir(config_dir)
cfg = hydra.compose("train_exact_gp_pca_14H.yaml")


@hydra.main()
def train_from_config(cfg):
    return train(**cfg)

if __name__ == "__main__":
    torch.cuda.set_device(1)
    train_from_config(cfg)
    #train_from_config()



