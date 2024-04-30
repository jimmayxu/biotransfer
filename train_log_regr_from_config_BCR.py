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
from src import train_gp, train_log_regr
import os 
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from hilearn.plot import ROC_plot, PR_curve
from sklearn.metrics import precision_score
import numpy as np
import umap

config_dir = '/ssd/users/zx243/Documents/biotransfer/configs/lm_gp_configs'
hydra.initialize_config_dir(config_dir)
# cfg = hydra.compose("train_BCR_IGH.yaml")
cfg = hydra.compose("train_BCR_IGH_small.yaml")

data_directory = '/ssd/users/zx243/Documents/biotransfer/example_data/bcr'


@hydra.main()
def train_from_config(cfg):
    return train_log_regr(**cfg)


# sampleID = 'HRS267727'
def data_prepare(dataDir, is_small=True):
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

    if is_small:
        IGH_concat = IGH_concat.sample(25000)
        name = 'IGH_small'
    else:
        name = 'IGH'

    ## Make binarise the clonal size such that one clone = 0 and greater than one clone = 1
    #IGH_concat_small.pred_aff = torch.where(torch.tensor(IGH_concat_small.pred_aff.values) == 1, 1., 0.).cpu()
    IGH_concat.pred_aff = torch.where(torch.tensor(IGH_concat.pred_aff.values) == 1, 0., 1.).cpu()

    IGH_train, IGH_test = train_test_split(IGH_concat, test_size=0.25)

    if not os.path.exists('example_data/bcr/%s' % name):
        os.mkdir('example_data/bcr/%s' % name)
    IGH_train.to_csv("example_data/bcr/%s/%s_train.csv" % (name, name), index=False)
    IGH_test.to_csv("example_data/bcr/%s/%s_test.csv" % (name, name), index=False)

    # preliminary visualisation
    plt.pie(IGH_concat.pred_aff.value_counts(), autopct='%1.0f%%', labels=["clone freq = 1", "clone freq > 1"])
    plt.show()

    IGH_concat.aa_seq.map(len).hist()
    plt.xlabel('length of chain')
    plt.show()

def UMAP_demo(output):

    data = pd.read_csv('/ssd/users/zx243/Documents/biotransfer/example_data/bcr/IGH_small/IGH_small_train.csv')
    label = data.pred_aff

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(output.cpu().numpy())

    color_dict = {0: 'blue', 1: 'yellow'}

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[color_dict[i] for i in label.iloc[:]])
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()

if __name__ == "__main__":

    dataDir = '/ssd/users/zx243/CR_HRA001149'
    # data_prepare(dataDir, is_small=True)
    torch.cuda.set_device(1)
    cfg = hydra.compose("train_BCR_IGH_small.yaml")

    variable_regions = True
    cfg.train_set_cfg.variable_regions = variable_regions
    cfg.eval_set_cfg.variable_regions = variable_regions
    pred_prob_vr, targets_vr, output_pool = train_from_config(cfg)
    #train_from_config()

    variable_regions = False
    cfg.train_set_cfg.variable_regions = variable_regions
    cfg.eval_set_cfg.variable_regions = variable_regions
    pred_prob_pool, targets_pool, output_vr = train_from_config(cfg)




    res = ROC_plot(1-targets_vr, 1-pred_prob_vr, threshold=0.5, color='red', legend_label='vr_mean', base_line=False)
    res = ROC_plot(1-targets_pool, 1-pred_prob_pool, threshold=0.5, color='blue', legend_label='pool')
    plt.show()

    res = PR_curve(1-targets_vr, 1-pred_prob_vr, threshold=0.5, color='red', legend_label='vr_mean')
    res = PR_curve(1-targets_pool, 1-pred_prob_pool, threshold=0.5, color='blue', legend_label='pool')
    plt.show()

    #UMAP_demo(output_pool)

    """
    targets = targets_vr
    pred_prob = pred_prob_vr
    cr = classification_report(targets, pred_prob > 0.5, target_names=["one clone only", "more than one clone"])
    print(cr)
    auroc = roc_auc_score(targets, pred_prob)
    aupr = average_precision_score(targets, pred_prob)
    print('evaluation accuracy: %.2f, evaluation auroc: %.2f, evaluation aupr: %.2f' % (accuracy, auroc, aupr))

    pred_benchmark = np.ones(len(pred_prob))
    cr = classification_report(targets, pred_benchmark, target_names=["more than one clone", "one clone only"])
    print(cr)
    auroc = roc_auc_score(targets, pred_benchmark)
    aupr = average_precision_score(targets, pred_benchmark)
    print('evaluation accuracy: %.2f, evaluation auroc: %.2f, evaluation aupr: %.2f' % (accuracy, auroc, aupr))
    """


