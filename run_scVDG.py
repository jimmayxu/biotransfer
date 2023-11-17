"""
python run_scVDG.py -C 'bcr'
python run_scVDG.py -C 'tcr'

"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import fast_matrix_market as fmm
import argparse

from scVDG import data_preparation


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--folder_path', type=str, default='example_data/scVDG')
    parser.add_argument('-C', '--ctype_name', type=str, default='bcr')
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arg()
    folder_path = args.folder_path
    ctype_name = args.ctype_name
    """
    folder_path = 'example_data/scVDG'
    ctype_name = 'tcr'
    """

    cr_table = pd.read_csv(os.path.join(folder_path, 'GSE158055_covid19_BCR_TCR/GSE158055_covid19_%s_vdjnt_pclone.tsv' % ctype_name), sep="\t")
    anno = pd.read_csv(os.path.join(folder_path, 'GSE158055_cell_annotation.csv'))
    mtx = fmm.mmread(os.path.join(folder_path, 'GSE158055_covid19_counts.mtx'), parallelism=80).T



    data_prep = data_preparation(folder_path, ctype_name)

    cellBarcode = cr_table['cellBarcode']
    mtx_ctype = data_prep.expression_extraction(anno, mtx, cellBarcode)

    k = 100
    indices, distances = data_prep.nearest_neighbour(k=k, mtx_ctype=mtx_ctype)

    clone_freq = cr_table['%s_clone.freq' %ctype_name.upper()]
    clone_freq_corrected = data_prep.corrected_clone_freq(clone_freq, indices, distances)

    clone_freq_corrected.index = cellBarcode
    clone_freq_corrected.reset_index().to_csv(os.path.join(data_prep.save_folder, '%s-clone_freq_corrected.csv' %(ctype_name)))



    """
    
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
    """