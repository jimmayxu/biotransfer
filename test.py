import torch
torch.cuda.set_device(3)
from cuml import PCA as IncrementalPCA

torch.cuda.is_available()
import numpy as np
train_y = np.loadtxt('results/train_y.txt')
train_X = np.loadtxt('results/train_X.txt')

train_X_small = train_X[:, :1000]
np.savetxt('results/train_X1000.txt', train_X_small)
pca_dim = 1000
pca_cuml = IncrementalPCA(n_components=pca_dim, svd_solver='auto')
train_X_pca = pca_cuml.fit_transform(train_X)
torch.cuda.empty_cache()

import torch
train_y = torch.from_numpy(train_y)

