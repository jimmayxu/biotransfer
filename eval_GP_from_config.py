# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Hook for evaluating GP with config file"""

import hydra
from random import randint
from src import eval_gp
import torch



config_dir = '/mnt/jimmyxu/nfs_share2/Documents/biotransfer/configs/lm_gp_configs'
hydra.initialize_config_dir(config_dir)
cfg = hydra.compose("eval_exact_gp_pca_14H.yaml")



@hydra.main()
def eval_from_config(cfg):
    return eval_gp(**cfg)

if __name__ == "__main__":
    torch.cuda.set_device(1)
    eval_from_config(cfg)
    #eval_from_config()
