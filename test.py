import os
import numpy as np
import umap
data_path = 'results'
data_folder = 'train_gp_91H'
import seaborn as sns

import matplotlib.pyplot as plt

train_y = np.loadtxt('%s/%s/data_pca/train_y.txt' %(data_path, data_folder))
train_X = np.loadtxt('%s/%s/data_pca/train_X.txt' %(data_path, data_folder))

train_X = np.loadtxt('data_pca/train_X.txt')
train_y = np.loadtxt('data_pca/train_y.txt')

sns.set(context='notebook', style='white', rc={'figure.figsize':(14,10)})

trans = umap.UMAP(n_neighbors=5, random_state=42).fit(train_X)
plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=train_y, cmap='Spectral')
plt.show()


from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(train_X)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s= 5, c=train_y, cmap='Spectral')
plt.show()