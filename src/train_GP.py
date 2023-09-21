# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Training classes and functions"""
import os
from importlib import import_module
import hydra
from torch.utils.data import DataLoader
import torch
import tqdm
import gpytorch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import joblib
import time
import numpy as np
from cuml import PCA as cumlPCA

#from dask_cuda import LocalCUDACluster
#from dask.distributed import Client


def validation(model, val_dataloader, feat_model, mll, variable_regions=None, pca_model=None, save_name=None):
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
    means = torch.tensor([0.])
    gt = torch.tensor([0.])
    Val_X = torch.tensor([])
    val_loss = []
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader):
            # extract features
            if torch.cuda.is_available():
                data, mask, target = batch["input_ids"].cuda(), batch["input_masks"].cuda(), batch["targets"].cuda()
            else:
                data, mask, target = batch["input_ids"], batch["input_masks"], batch["targets"]
            val_X_ = feat_model(data, mask, variable_regions=variable_regions)
            if pca_model is not None:
                val_X = torch.as_tensor(pca_model.transform(val_X_.detach()))
                val_X = val_X.cuda()
            output = model(val_X)  
            means = torch.cat([means, output.mean.cpu()])
            gt = torch.cat([gt, target.squeeze(-1).cpu()])
            val_loss.append(-mll(output, target.squeeze(-1)))

            Val_X = torch.cat((Val_X, val_X.cpu()), 0)

    gt = gt[1:]
    means = means[1:]
    if save_name is not None:
        np.savetxt('data_pca/%s_X.txt' % save_name, Val_X)
        np.savetxt('data_pca/%s_y.txt' % save_name, gt)
    
    MAE = torch.mean(torch.abs(means - gt))
    print('Test MAE: {}'.format(MAE))
    gt = gt.detach().cpu().numpy()
    means = means.detach().cpu().numpy()
    PCC = pearsonr(gt, means)[0]
    print('Test Pearson: {}'.format(PCC))
    mean_loss = sum(val_loss)/len(val_loss)
    return mean_loss.cpu().numpy(), MAE.numpy(), PCC


def train_gp(train_set_cfg, train_dataloader_cfg, feat_cfg, model_cfg=None,
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
    os.makedirs('data_pca/%s/' % train_set_cfg.chain, exist_ok=True)
    os.makedirs('results/%s/' % train_set_cfg.chain, exist_ok=True)

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
        print('Variable regions:', variable_regions)



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
    """
    start_time = time.time()
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

    np.savetxt('data_pca/%s/train_X.txt' %train_set_cfg.chain, train_X.cpu().numpy())
    np.savetxt('data_pca/%s/train_y.txt' % train_set_cfg.chain, train_y.cpu().numpy())
    """
    train_X_ = np.loadtxt('results/train_X_pca.txt')
    train_X = torch.as_tensor(train_X_).cuda()
    train_y_ = np.loadtxt('results/train_y.txt')
    train_y = torch.as_tensor(train_y_).cuda()
    """


    # load GP model
    target_args = model_cfg._target_.split(".")
    module_path = ".".join(target_args[:-1])
    module_name = target_args[-1]
    module = getattr(import_module(module_path), module_name)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if model_cfg.inducing_points:
        inducing_points=min([model_cfg.inducing_points, len(train_X)])
        gp_model = module(train_X, train_y, likelihood, inducing_points=inducing_points)
    else:
        gp_model = module(train_X, train_y, likelihood)
    if torch.cuda.is_available():
        gp_model.cuda()
        likelihood.cuda()
    print(gp_model)

    # -------TRAIN------
    torch.cuda.empty_cache()

    gp_model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(gp_model.parameters(), lr=model_cfg.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    epoch_iter = tqdm.tqdm(range(model_cfg.num_epochs))
    # 21k samples per iteration
    for i in epoch_iter:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = gp_model(train_X)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        
        loss.backward()
        #iterator.set_postfix(loss=loss.item())
        optimizer.step()
        epoch_iter.set_postfix(loss=loss.item())

    # ------Validation------

    # Load validation data handlers
    if val_set_cfg is not None:
        val_set = hydra.utils.instantiate(val_set_cfg)
        if hasattr(val_set, "collate_fn"):
            val_dataloader = DataLoader(dataset=val_set, collate_fn=val_set.collate_fn, **val_dataloader_cfg)
        else:
            val_dataloader = DataLoader(dataset=val_set, **val_dataloader_cfg)
    if val_set_cfg is not None:
        gp_model.eval()
        likelihood.eval()
        val_loss, val_MAE, val_PCC = validation(gp_model, val_dataloader, feat_model, mll, variable_regions=variable_regions, pca_model=pca_model, save_name='%s/valid' %val_set_cfg.chain)
        print('validation loss:', val_loss)
        print('validation MAE:', val_MAE)
        print('validation PCC:', val_PCC)
    
    # save model
    torch.save(gp_model.state_dict(), 'results/%s/GP_model_state_dict.pth' %train_set_cfg.chain)
    if pca_dim is not None: # pca is used and save the pca model
        joblib.dump(pca_model, 'results/%s/pca_model.sav')


    # ------evaluation------
    eval_set = hydra.utils.instantiate(eval_set_cfg)
    if hasattr(eval_set, "collate_fn"):
        eval_dataloader = DataLoader(dataset=eval_set, collate_fn=eval_set.collate_fn, **eval_dataloader_cfg)
    else:
        eval_dataloader = DataLoader(dataset=eval_set, **eval_dataloader_cfg)

    gp_model.eval()
    likelihood.eval()
    eval_loss, eval_MAE, eval_PCC = validation(gp_model, eval_dataloader, feat_model, mll, variable_regions=variable_regions, pca_model=pca_model, save_name='%s/test' %eval_set_cfg.chain)
    print('evaluation loss:', eval_loss)
    print('evaluation MAE:', eval_MAE)
    print('evaluation PCC:', eval_PCC)


"""
 if pca_dim is not None: # perform pca
        if torch.cuda.is_available():
            cluster = LocalCUDACluster(protocol="ucx",
                           enable_tcp_over_ucx=True,
                           enable_nvlink=True,
                           enable_infiniband=False)
            client = Client(cluster)
            pca_cuml = cumlPCA(n_components=pca_dim, svd_solver='auto')
            client.submit(pca_cuml.fit_transform)
            data = client.scatter(train_X.cpu().numpy(), broadcast=True)
            client.run(pca_cuml.fit_transform(data)
            train_X = client.map(lambda data, model: pca_cuml.fit_transform(data), data, pure=False)
            train_X = client.gather(train_X)
            train_X = train_X.cuda()
        else:
            pca_model = KernelPCA(pca_dim, kernel='linear', copy_X=False).fit(train_X)
            train_X = torch.from_numpy(pca_model.transform(train_X))
        
    else:
        pca_model = None
    
"""