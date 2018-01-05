import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_path = "D:/LiangYiHuai/gao/";
print('Loading data...')
input_data = pd.read_csv(data_path + 'input_data.csv',
                    delimiter='\t',
                    dtype={'ID' : 'category',
                            'x1' : np.uint32,
                            'x2' : np.uint32,
                            'x3' : np.uint32,
                            'x4' : np.uint32,
                            'x5' : np.uint32,
                            'x6' : np.float,
                            'finaltime' : np.uint32})

input_data.pop('ID')

y_train = input_data.pop('finaltime')
x_train = input_data;

d_train = lgb.Dataset(x_train, y_train)

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.15 ,
        'verbose': 0,
        'num_leaves': 40,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'num_rounds': 50,
        'metric': 'auc',
    }

# lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=5)
lgbm_model = lgb.train(params, train_set=d_train, verbose_eval=5)
# Verbose_eval prints output after every 5 iterations

predictions = lgbm_model.predict(x_train)
print("prediction result: %s"%predictions);

