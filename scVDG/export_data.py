import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

ctype_name = 'tcr'
folder_path = 'example_data/scVDG'
save_folder = 'example_data/covid/replicate_corrected'

table = pd.read_csv(os.path.join('example_data/scVDG/data_prep', '%s-clone_freq_corrected.csv' %(ctype_name)))

corrected_freq = np.log(table['%s_clone.freq' %ctype_name.upper()])

cr_table = pd.read_csv(os.path.join(folder_path, 'GSE158055_covid19_BCR_TCR/GSE158055_covid19_%s_vdjnt_pclone.tsv' % ctype_name), sep="\t")
cr_table.columns
for cdr_name in ['TCRA_cdr3aa', 'TCRB_cdr3aa']:
# for cdr_name in ['BCRH_cdr3aa', 'BCRL/K_cdr3aa']:
    export_table = cr_table.iloc[:, :3]
    export_table['aa_seq'] = cr_table[cdr_name]
    export_table['pred_aff'] = corrected_freq

    train, test = train_test_split(export_table)

    cdr_name_ = cdr_name.replace('/', '')
    os.makedirs(os.path.join(save_folder, cdr_name_), exist_ok=True)
    train.to_csv(os.path.join(save_folder, cdr_name_, '%s_train.csv' %cdr_name_), index=False)
    test.to_csv(os.path.join(save_folder, cdr_name_, '%s_test.csv' %cdr_name_), index=False)





