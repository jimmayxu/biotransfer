# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Training classes and functions"""
import os
from importlib import import_module
import hydra
import pandas as pd
from torch.utils.data import DataLoader
import torch
import tqdm
import gpytorch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import joblib
import time
import numpy as np
#from cuml import PCA as cumlPCA
from sklearn.metrics import classification_report

#from dask_cuda import LocalCUDACluster
#from dask.distributed import Client


def validation(model, val_dataloader, feat_model, variable_regions=None, pca_model=None):
    """Evaluate model on validation set

    Args:
        model: Model to perform validation
        val_dataloader: Validation dataloader
        feat_model: Model to extract features
        mll: Marginal log likelihood loss function
        variable regions: ???
        pca_model: Model to perform pca

    Returns
        Average validation loss
    """
    correct = 0
    targets = torch.tensor([0.])
    pred = torch.tensor([0.])
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader):
            # extract features
            if torch.cuda.is_available():
                data, mask, target = batch["input_ids"].cuda(), batch["input_masks"].cuda(), batch["targets"].cuda()
            else:
                data, mask, target = batch["input_ids"], batch["input_masks"], batch["targets"]
            val_X = feat_model(data, mask, variable_regions=variable_regions)
            if pca_model is not None:
                val_X = val_X.cpu()
                val_X = torch.as_tensor(pca_model.transform(val_X.detach()))
                val_X = val_X.cuda()
            output = model(val_X)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target.flatten()).sum()
            pred = torch.cat([pred, predicted.squeeze(-1).cpu()])
            targets = torch.cat([targets, target.squeeze(-1).cpu()])

    accuracy = 100 * (correct.item()) / len(val_dataloader.dataset.data)
    pred = pred.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return pred, targets, accuracy


def train_log_regr(train_set_cfg, train_dataloader_cfg, feat_cfg, model_cfg=None,
             val_set_cfg=None, val_dataloader_cfg=None, logger_cfgs=None,
             eval_set_cfg=None, eval_dataloader_cfg=None, callback_cfgs=None, checkpoint_callback_cfg=None, seed=0, reload_checkpoint_path=None, reload_state_dict_path=None,
             strict_reload=True,
             experiment_name=None,
             nodes=None):
    """
    Train using given configurations.

    Args:
        train_set_cfg: Training dataset configuration
        train_dataloader_cfg: Training dataloader configuration
        feat_cfg: Feature extractor model configuration
        model_cfg: Model configuration
        val_set_cfg: Validation dataset configuration
        logger_cfgs: List of logger configurations
        callback_cfgs: List of callback configurations
        checkpoint_callback_cfg: Checkpoing callback configuration
        seed: Seed for initialization and training.
        reload_checkpoint_path: Path for reloading model from a
          pytorch-lightning checkpoint
        reload_state_dict_path: Path for reloading model from a
          pytorch state dictionary
        strict_reload: Whether or not the current model must
          exactly match the reloaded weight architecture
        experiment_name: Name of this experiment
        nodes: Number of nodes
    """
    os.makedirs(os.path.join('results', experiment_name), exist_ok=True)

    # Load training data handlers
    train_set = hydra.utils.instantiate(train_set_cfg)
    print("TRAIN SET SIZE: {}".format(len(train_set)))
    if hasattr(train_set, "collate_fn"):
        train_dataloader = DataLoader(dataset=train_set, collate_fn=train_set.collate_fn, **train_dataloader_cfg)
    else:
        train_dataloader = DataLoader(dataset=train_set, **train_dataloader_cfg)

    # Load feature extractor (pretrained language model)
    target_args = feat_cfg._target_.split(".")
    module_path = ".".join(target_args[:-1])
    module_name = target_args[-1]
    module = getattr(import_module(module_path), module_name)
    if torch.cuda.is_available():
        feat_model = module.load_from_checkpoint(feat_cfg.feat_path, model_config_file=feat_cfg.model_config_file,
                                             strict=feat_cfg.strict_reload, map_location=torch.device('cuda', torch.cuda.current_device()))
    else:
        feat_model = module.load_from_checkpoint(feat_cfg.feat_path, model_config_file=feat_cfg.model_config_file,
                                                 strict=feat_cfg.strict_reload,
                                                 map_location=torch.device('cpu'))


    # pca dimensionality reduction
    pca_dim = feat_cfg.pca_dim
    assert isinstance(pca_dim, int) or (pca_dim is None), 'pca_dim is either None or an positive integer'
    # extract variable_regions indices if variable_regions=True
    if not feat_cfg.variable_regions:
        variable_regions = None
    else:
        variable_regions = train_set.get_variable_regions()
        #print('Variable regions:', variable_regions)


    variable_regions = None
    pca_dim = None


    # prepare train data
    feat_model.eval()
    train_X = torch.tensor([])
    train_y = torch.tensor([])
    if torch.cuda.is_available():
        feat_model.cuda()
        train_X = train_X.cuda()
        train_y = train_y.cuda()
    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
            if torch.cuda.is_available():
                data, mask, target = batch["input_ids"].cuda(), batch["input_masks"].cuda(), batch["targets"].cuda()
            else:
                data, mask, target = batch["input_ids"], batch["input_masks"], batch["targets"]
            train_y = torch.cat((train_y, target.squeeze(-1)), 0)
            output = feat_model(data, mask, variable_regions=variable_regions)
            train_X = torch.cat((train_X, output), 0)


    print(train_X.size())
    print(train_y.size())

    torch.cuda.empty_cache()
    if pca_dim is not None: # perform pca
        if torch.cuda.is_available():
            train_X = train_X.cpu()
        pca_model = KernelPCA(pca_dim, kernel='linear', copy_X=False).fit(train_X)
        train_X = torch.from_numpy(pca_model.transform(train_X))
        if torch.cuda.is_available():
            train_X = train_X.cuda()
        print(train_X.size())
    else:
        pca_model = None

    """
    torch.cuda.empty_cache()
    start_time = time.time()
    if pca_dim is not None: # perform pca
        if torch.cuda.is_available():
            pca_model = cumlPCA(n_components=pca_dim, svd_solver='auto').fit(train_X)
            train_X = pca_model.transform(train_X)
            train_X = torch.as_tensor(train_X).cuda()
        else:
            pca_model = KernelPCA(pca_dim, kernel='linear', copy_X=False).fit(train_X)
            train_X = torch.as_tensor(pca_model.transform(train_X))
        
    else:
        pca_model = None
    
    print("PCA running \n--- %.3f minutes ---" % ((time.time() - start_time) / 60))

    """

    # np.savetxt('data_pca/train_X_pool.txt', train_X.cpu().numpy())
    # np.savetxt('data_pca/train_y_pool.txt', train_y.cpu().numpy())
    # train_X = np.loadtxt('data_pca/train_X_pool.txt')
    # train_y = np.loadtxt('data_pca/train_y_pool.txt')
    # train_X = torch.from_numpy(train_X).float().to(1)
    # train_y = torch.from_numpy(train_y).float().to(1)


    # load GP model
    target_args = model_cfg._target_.split(".")
    module_path = ".".join(target_args[:-1])
    module_name = target_args[-1]
    module = getattr(import_module(module_path), module_name)
    n_inputs = train_X.shape[1]  # makes a 1D vector of 784
    n_outputs = 2
    log_regr = module(n_inputs, n_outputs)
    if torch.cuda.is_available():
        log_regr.cuda()

    # -------TRAIN------
    torch.cuda.empty_cache()
    log_regr.train()

    # defining the optimizer
    optimizer = torch.optim.SGD(log_regr.parameters(), lr=model_cfg.lr)
    # defining Cross-Entropy loss
    criterion = torch.nn.CrossEntropyLoss()

    train_y = train_y.type(torch.LongTensor).cuda()

    epoch_iter = tqdm.tqdm(range(model_cfg.num_epochs))
    for i in epoch_iter:
        optimizer.zero_grad()
        outputs = log_regr(train_X)
        loss = criterion(outputs, train_y)
        # Loss.append(loss.item())
        loss.backward()
        optimizer.step()
        epoch_iter.set_postfix(loss=loss.item())


    # ------evaluation------
    torch.cuda.empty_cache()
    # Load evaluation data handlers
    eval_set = hydra.utils.instantiate(eval_set_cfg)
    if hasattr(eval_set, "collate_fn"):
        eval_dataloader = DataLoader(dataset=eval_set, collate_fn=eval_set.collate_fn, **eval_dataloader_cfg)
    else:
        eval_dataloader = DataLoader(dataset=eval_set, **eval_dataloader_cfg)
    log_regr.eval()
    pred, targets, accuracy = validation(log_regr, eval_dataloader, feat_model,
                                               variable_regions=variable_regions, pca_model=pca_model)
    cr = classification_report(targets, pred, target_names=["more than one clone", "one clone only"])
    print(cr)
    print('evaluation accuracy: %.2f%%' %accuracy)
    save_results = pd.DataFrame.from_dict({"targets": targets, "pred": pred})

    save_results.to_csv(os.path.join('results', experiment_name, 'pred_results.csv'))
