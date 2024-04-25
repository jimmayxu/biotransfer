# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Hook for training GP with config file"""
"""
python train_GP_from_config.py -cd configs/lm_gp_configs/ -cn train_exact_gp_pca_14H.yaml
python train_GP_from_config.py -cd configs/lm_gp_configs/ -cn train_exact_gp_pca_14L.yaml
python train_GP_from_config.py -cd configs/lm_gp_configs/ -cn train_exact_gp_pca_91H.yaml
python train_GP_from_config.py -cd configs/lm_gp_configs/ -cn train_exact_gp_pca_95L.yaml
"""
import hydra
from random import randint
from src import train_gp
import os 
import torch


config_dir = '/ssd/users/zx243/Documents/biotransfer/configs/lm_gp_configs'
hydra.initialize_config_dir(config_dir)
cfg = hydra.compose("train_exact_gp_pca_14H.yaml")


@hydra.main()
def train_from_config(cfg):
    return train_gp(**cfg)

if __name__ == "__main__":
    torch.cuda.set_device(1)
    train_from_config(cfg)
    #train_from_config()

