import pandas as pd
import os
folder_path = 'example_data/scVDG/GSE158055_covid19_BCR_TCR'
table = pd.read_csv(os.path.join(folder_path, 'GSE158055_covid19_bcr_vdjnt_pclone.tsv'), sep="\t")

table.sampleID.value_counts()
table.PatientID.value_counts()

data = table[['sampleID', 'BCR_clonal', 'BCRH_cdr3aa']]

import scipy.io
counts = scipy.io.mmread(os.path.join(folder_path, 'GSE158055_covid19_features.tsv.gz'))

table.columns

sum(table['BCR_clonal'])