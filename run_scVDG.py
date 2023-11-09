import time
import numpy as np
import os
import matplotlib.pyplot as plt

from scVDG import data_preparation

folder_path = 'example_data/scVDG'


#expression_extraction(folder_path, 'bcr')
#expression_extraction(folder_path, 'tcr')
#indices, distances = nearest_neighbour(folder_path, 'bcr', 100)

ctype = 'bcr'
save_file = False

data_prep = data_preparation(folder_path, ctype)

mtx_ctype = data_prep.expression_extraction(save_file=save_file)

k = 100
indices, distances = data_prep.nearest_neighbour(k=k, mtx_ctype=mtx_ctype, save_file=save_file)




indices = np.loadtxt(os.path.join(folder_path, 'nearest_neighour/%s_indices_k%d.txt' %('bcr', k)))
distances = np.loadtxt(os.path.join(folder_path, 'nearest_neighour/%s_dists_k%d.txt' %('bcr', k)))


distances_ = distances[:, 1:]
indices_ = indices[:, 1:]
# distance metric to similarity metric
similarity = 1. / (distances_ / np.max(distances_))
similarity[:, 0]


similarity.max()
np.quantile(similarity, 0.9)
#similarity_prop = np.true_divide(similarity, similarity.max(axis=1, keepdims=True))
#similarity_prop = np.true_divide(similarity, similarity.mean(axis=1, keepdims=True))
#similarity_prop = np.true_divide(similarity, np.quantile(similarity, 0.9))
similarity_prop = np.true_divide(similarity, similarity.max())

xx = data_prep.cr_table
clone_freq = data_prep.cr_table['BCR_clone.freq']
clone_freq_indexed = clone_freq.values[indices_.flatten().astype(int)].reshape(indices_.shape)

rest_contribute = (clone_freq_indexed * similarity_prop).sum(axis=1)
clone_freq_corrected = clone_freq + rest_contribute

plt.hist(np.log(clone_freq))
plt.scatter(clone_freq, clone_freq_corrected)
plt.show()

freq[[0,100],[100,1]]