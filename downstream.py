import numpy as np
import pandas as pd
import seaborn as sns
import umap
import matplotlib.pyplot as plt
folder_names = ['14H', '14L', '91H', '95L']


df = pd.DataFrame()
for folder_name in folder_names:
    performance = pd.read_csv('results/train_gp_%s/performance.csv' %folder_name, index_col=0)
    df = pd.concat([df, performance], axis=1)

df.columns = ['Ab-14-H', 'Ab-14-L', 'Ab-91-H', 'Ab-95-L']

subdf = df.loc['evaluation_PCC'].reset_index()
subdf.columns = ['Variants', 'Affinity Prediction Pearson Correlation']
sns.set_style("whitegrid", {'grid.linestyle': ':'})
sns.barplot(subdf, x='Variants', y='Affinity Prediction Pearson Correlation')
plt.savefig('figures/Pearson_corr.pdf', dpi=300)
plt.show()
plt.close()

subdf = df.loc['evaluation_SPEAR'].reset_index()
subdf.columns = ['Variants', 'Affinity Prediction Spearman Correlation']
sns.set_style("whitegrid", {'grid.linestyle': ':'})
sns.barplot(subdf, x='Variants', y='Affinity Prediction Spearman Correlation')
plt.savefig('figures/Spearman_corr.pdf', dpi=300)
plt.show()
plt.close()




data_path = 'results'
data_folder = 'train_gp_14H'

train_y = np.loadtxt('%s/%s/data_pca/train_y.txt' %(data_path, data_folder))
train_X = np.loadtxt('%s/%s/data_pca/train_X.txt' %(data_path, data_folder))


trans = umap.UMAP(n_neighbors=5, random_state=42).fit(train_X)
#embedding = pd.DataFrame(trans.embedding_, columns=['UMAP 1', 'UMAP 2'])
#ax = sns.scatterplot(embedding, x='UMAP 1', y='UMAP 2', hue=train_y, palette='Spectral')
#ax.get_legend().remove()



points = plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=train_y, cmap='Spectral')
ax = plt.gca()
plt.colorbar(points)
ax.set_xticks([])
ax.set_yticks([])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("Ab-14-H")
plt.savefig('figures/UMAP_train_gp_14H.pdf', dpi=300)
plt.show()
plt.close()
