import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import fast_matrix_market as fmm
from pyflann import FLANN
from scipy import sparse
import time


class data_preparation():
    def __init__(self, folder_path='example_data/scVDG', ctype_name='bcr'):
        self.folder_path = folder_path
        self.ctype_name = ctype_name

        self.cr_table = pd.read_csv(os.path.join(self.folder_path,
                                        'GSE158055_covid19_BCR_TCR/GSE158055_covid19_%s_vdjnt_pclone.tsv' % self.ctype_name),
                           sep="\t")
    def expression_extraction(self, save_file=True):
        anno = pd.read_csv(os.path.join(self.folder_path, 'GSE158055_cell_annotation.csv'))
        mtx = fmm.mmread(os.path.join(self.folder_path, 'GSE158055_covid19_counts.mtx'), parallelism=80).T

        print('Start tocsr')
        t = time.time()
        mtx = mtx.tocsr()
        print('Time used for tocsr is %.2f min' %((time.time() - t)/60))

        # cell preparation
        cell_index = anno.index[anno.cellName.isin(self.cr_table['cellBarcode'])]
        t = time.time()
        mtx_ctype = mtx[cell_index]
        print('Time used for %s exp extraction is %.2f min' %(self.ctype_name, (time.time() - t)/60))

        if save_file:
            #save csr file
            sparse.save_npz(os.path.join(self.folder_path, "%s_expression.npz" %self.ctype_name), mtx_ctype)

        return mtx_ctype

    def nearest_neighbour(self, k=100, mtx_ctype=None, save_file=True):
        os.makedirs(os.path.join(self.folder_path, 'nearest_neighour'), exist_ok = True)
        if mtx_ctype is None:
            mtx_ctype = sparse.load_npz(os.path.join(self.folder_path, "%s_expression.npz" %self.ctype_name))

        mtx = mtx_ctype.log1p()
        mtx_array = mtx.toarray()

        flann = FLANN()
        t = time.time()
        indices, distances = flann.nn(mtx_array, mtx_array, k)
        print('Time used for flann is %.2f min' % ((time.time() - t) / 60))
        if save_file:
            np.savetxt(os.path.join(self.folder_path, 'nearest_neighour/%s_indices_k%d.txt' %(self.ctype_name, k)), indices)
            np.savetxt(os.path.join(self.folder_path, 'nearest_neighour/%s_dists_k%d.txt' %(self.ctype_name, k)), distances)

        return indices, distances


    def weighted_clone_freq(self, indices, distances):

        distances_ = distances[:, 1:]
        indices_ = indices[:, 1:]
        # distance metric to similarity metric
        similarity = 1. / (distances_ / np.max(distances_))

        similarity_prop = np.true_divide(similarity, similarity.sum(axis=1, keepdims=True))

        clone_freq = self.cr_table['BCR_clone.freq']
        clone_freq_indexed = clone_freq.values[indices_.flatten().astype(int)].reshape(indices_.shape)

        rest_contribute = (clone_freq_indexed * similarity_prop).sum(axis=1)
        clone_freq_weighted = clone_freq + rest_contribute

        return clone_freq_weighted



"""

your_matrix_back = sparse.load_npz(os.path.join(folder_path, "B_cell_expression.npz"))

code = tcr_table['TCRA_cdr3aa']
bcr_table['BCR_clone.freq'].hist()

(tcr_table['TCR_clone.freq'] - tcr_table['TCR_pclone.freq']).value_counts()
len(tcr_table)
plt.show()
tcr_table.sampleID.value_counts()
tcr_table.PatientID.value_counts()



dataset = np.array(
    [[1., 1, 1, 2, 3],
     [10, 10, 10, 3, 2],
     [100, 100, 2, 30, 1]
     ])
dataset = sparse.csr_matrix(dataset)
data.tocsr()[pd.Index([0,2])]

sparse.csc_matrix(data, ())

testset = np.array(
    [[1., 1, 1, 1, 1],
     [90, 90, 10, 10, 1]
     ])
testset = sparse.csr_matrix(testset)
flann = FLANN()
indices, distances = flann.nn(
    dataset, dataset, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)


flann.build_index(testset)
flann.nn_index(testset, 2)
result
dists
"""