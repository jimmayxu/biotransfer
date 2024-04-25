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
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

config_dir = '/ssd/users/zx243/Documents/biotransfer/configs/lm_gp_configs'
hydra.initialize_config_dir(config_dir)
cfg = hydra.compose("train_BCR.yaml")

data_directory = '/ssd/users/zx243/Documents/biotransfer/example_data/bcr'


@hydra.main()
def train_from_config(cfg):
    return train_gp(**cfg)

dataDir = '/ssd/users/zx243/CR_HRA001149'
sampleID = 'HRS267727'
def data_prepare(dataDir):
    sampleIDs = os.listdir(dataDir)
    IGH_concat = pd.DataFrame()
    for sampleID in tqdm(sampleIDs):
        Table_barcode = pd.read_csv(os.path.join(dataDir, sampleID, 'data_prep', 'Table_barcode.csv'), index_col=0)
        Table_barcode_ = Table_barcode.loc[:, ["IGH", "clone_freq"]][~Table_barcode.clone_freq.isna()]

        full_chain_regions = pd.read_csv(os.path.join(dataDir, sampleID, 'data_prep', 'full_chain_regions.csv'), index_col=[0, 1])
        full_chain_regions_ = full_chain_regions[full_chain_regions.index.get_level_values('chain') == 'IGH'].reset_index('chain', drop=True)
        full_chain_regions_ = full_chain_regions_.loc[Table_barcode_.index]

        IGH_info = pd.concat([Table_barcode_, full_chain_regions_], axis=1)
        IGH_info.index.name = 'barcode'
        IGH_info.reset_index(inplace=True)
        IGH_info.loc[:, 'sampleID'] = sampleID
        IGH_concat = pd.concat([IGH_concat, IGH_info], ignore_index=True)
    IGH_concat.rename(columns={"IGH": "aa_seq", "clone_freq": "pred_aff"}, inplace=True)

    IGH_train, IGH_test = train_test_split(IGH_concat, test_size=0.25)
    IGH_train.to_csv("example_data/bcr/IGH/IGH_train.csv", index=False)
    IGH_test.to_csv("example_data/bcr/IGH/IGH_test.csv", index=False)


if __name__ == "__main__":


    torch.cuda.set_device(1)
    train_from_config(cfg)
    #train_from_config()

