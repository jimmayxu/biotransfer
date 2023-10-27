import os
import numpy as np
data_path = 'results'
data_folder = 'train_gp_14H'
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import xgboost as xg
import seaborn as sns

train_y = np.loadtxt('%s/%s/data_pca/train_y.txt' %(data_path, data_folder))
train_X = np.loadtxt('%s/%s/data_pca/train_X.txt' %(data_path, data_folder))
eval_X = np.loadtxt('%s/%s/data_pca/test_X.txt' %(data_path, data_folder))
eval_y = np.loadtxt('%s/%s/data_pca/test_y.txt' %(data_path, data_folder))

# random forest regressor

regressor = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=40)
regressor = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', random_state = 18, n_jobs=40)
regressor.fit(train_X, train_y)
pred_y = regressor.predict(eval_X)


# gradient boosting regressor
xgb_r = xg.XGBRegressor(n_estimators = 500, seed = 123, n_jobs=60, learning_rate=0.01, booster="gblinear", objective="reg:squaredlogerror")
xgb_r.fit(train_X, train_y)

pred_y = xgb_r.predict(eval_X)

# --------- evaluation -----------#
PCC = pearsonr(eval_y, pred_y)[0]
SPEAR = spearmanr(eval_y, pred_y)[0]
print(PCC)
print(SPEAR)


data_folders = ['train_gp_14H', 'train_gp_14L', 'train_gp_91H', 'train_gp_95L']
Table = pd.DataFrame()
for data_folder in tqdm(data_folders):
    train_y = np.loadtxt('%s/%s/data_pca/train_y.txt' % (data_path, data_folder))
    train_X = np.loadtxt('%s/%s/data_pca/train_X.txt' % (data_path, data_folder))
    eval_X = np.loadtxt('%s/%s/data_pca/test_X.txt' % (data_path, data_folder))
    eval_y = np.loadtxt('%s/%s/data_pca/test_y.txt' % (data_path, data_folder))

    xgb_r = xg.XGBRegressor(n_estimators=500, seed=123, n_jobs=60, learning_rate=0.01, booster="gblinear",
                            objective="reg:squaredlogerror")
    xgb_r.fit(train_X, train_y)

    pred_y = xgb_r.predict(eval_X)

    # --------- evaluation -----------#
    PCC = pearsonr(eval_y, pred_y)[0]
    SPEAR = spearmanr(eval_y, pred_y)[0]
    Table = pd.concat([Table, pd.Series((PCC, SPEAR), name=data_folder)], axis=1)

Table.index = ['evaluation_PCC', 'evaluation_SPEAR']

Table.columns = ['Ab-14-H', 'Ab-14-L', 'Ab-91-H', 'Ab-95-L']

all = Table.stack()
all = all.reset_index()
all.columns = ['metric', 'Variants', 'value']
all['Methods'] = 'linear'


per = df.loc[['evaluation_PCC', 'evaluation_SPEAR']]
per = per.stack().reset_index()
per.columns = ['metric', 'Variants', 'value']
per['Methods'] = 'GP'


all_ = pd.concat([all, per])

PCC = all_[all_.metric == 'evaluation_PCC']
PCC.loc[:, 'Affinity Prediction Pearson Correlation'] = PCC['value']
SPEAR = all_[all_.metric == 'evaluation_SPEAR']
SPEAR.loc[:, 'Affinity Prediction Spearman Correlation'] = SPEAR['value']


sns.barplot(SPEAR, x='Variants', y='Affinity Prediction Spearman Correlation', hue='Methods')
plt.savefig('figures/LinearRegressor_spear.pdf', dpi=300)
plt.show()

sns.barplot(PCC, x='Variants', y='Affinity Prediction Pearson Correlation', hue='Methods')
plt.savefig('figures/LinearRegressor_pcc.pdf', dpi=300)
plt.show()
"""
from sklearn.model_selection import RandomizedSearchCV
# Create the random grid
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 80)
# Fit the random search model
rf_random.fit(train_X, train_y)
"""