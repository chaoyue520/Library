# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb

# read data ----------------------------------------------------------------------------

data_set = pd.read_table('/home/gaofeixiang/cbo/data/data_set_g_v3_sample.txt',
                         sep='\t',
                         na_values=['', 'NA', 'NULL', 'na', 'null'])

# split train valid oot dataset --------------------------------------------------------
train_set = data_set[(data_set.data_set_seg == 1)]
valid_set = data_set[(data_set.data_set_seg == 2)]
oot_set = data_set[(data_set.data_set_seg == 3)]

# selected model vars ------------------------------------------------------------------
cols_x_df = pd.read_table('/home/gaofeixiang/cbo/model_v1_online/vars_model_selected_neibu.txt',
                          sep='\t',
                          header=None,
                          names=['var_name'],
                          na_values=['', 'NULL', 'NA', 'null', 'na'])
cols_x = list(cols_x_df.var_name)

# to DMatrix

train_data = train_set[cols_x].as_matrix()
train_label = train_set['y_general_v2'].as_matrix()
dtrain = xgb.DMatrix(train_data, label=train_label)

valid_data = valid_set[cols_x].as_matrix()
valid_label = valid_set['y_general_v2'].as_matrix()
dvalid = xgb.DMatrix(valid_data, label=valid_label)

oot_data = oot_set[cols_x].as_matrix()
oot_label = oot_set['y_general_v2'].as_matrix()
doot = xgb.DMatrix(oot_data, label=oot_label)

# parameter
param = {'max_depth': 2,
         'learning_rate': 0.01,
         'min_child_weight': 50,
         'silent': 1,
         'objective': 'binary:logistic',
         'eval_metric': 'auc'}

evallist = [(dtrain, 'train'), (dvalid, 'valid'), (doot, 'oot')]

bst = xgb.train(param,
                dtrain,
                num_boost_round=1800,
                evals=evallist)

# 打分 ------------------------------------------------------------------------------
dafen_data_path = '/home/gaofeixiang/xjd_ymd_m12_01/'
data_x_file = 'xjd_ymd_x.txt'
pred_file = 'xjd_ymd_fraud_score'

dafen_set = pd.read_table(dafen_data_path + data_x_file,
                          sep='\t',
                          na_values=['NA', 'na', 'NaN', '\\N', 'NULL', 'null', ''])
dafen_data = dafen_set[cols_x].as_matrix()
ddafen = xgb.DMatrix(dafen_data)
pred_df = dafen_set[['apply_mon', 'prod_code_flg', 'passid']]
pred_df['score'] = bst.predict(ddafen)

# 模型分修正 ------------------------------------------------------------------------

pred_df = pred_df.assign(xbeta_old=np.log(pred_df.score / (1 - pred_df.score)))
pred_df.loc[pred_df.xbeta_old >= 10, 'xbeta_old'] = 10
pred_df.loc[pred_df.xbeta_old <= -10, 'xbeta_old'] = -10
pred_df = pred_df.assign(xbeta_new=pred_df.xbeta_old * 1.48 - 1.36)
pred_df = pred_df.assign(score_new=1 / (1 + np.exp(-1 * pred_df.xbeta_new)))

pred_df.score_new.describe()
pred_df.score.describe()

pred_df.to_csv(dafen_data_path + pred_file,
               sep='\t',
               header=True,
               index=False)

# save model -------------------------------------------------------------------------

bst.save_model('/home/gaofeixiang/cbo/data/model/general_v2_union.model')

# dump model with feature map
feat_map_df = pd.DataFrame({'id': [i for i in range(len(cols_x))]})
feat_map_df = feat_map_df.assign(feat_name=cols_x)
feat_map_df = feat_map_df.assign(type=['q' for i in range(len(cols_x))])
feat_map_df.to_csv('/home/gaofeixiang/cbo/data/model/feat_map_general_v2_union.txt',
                   sep='\t',
                   header=False,
                   index=False)

bst.dump_model('/home/gaofeixiang/cbo/data/model/general_v2_union.dump.raw.txt',
               '/home/gaofeixiang/cbo/data/model/feat_map_general_v2_union.txt')

# feature importance
fscore_dict = bst.get_fscore(fmap='/home/gaofeixiang/cbo/data/model/feat_map_general_v2_union.txt')
features = []
scores = []
for key in fscore_dict:
    features.append(key)
    scores.append(fscore_dict[key])

fscore_df = pd.DataFrame({'features': features, 'scores': scores})
fscore_df.to_csv('/home/gaofeixiang/cbo/data/model/feature_importance_general_v2_union.txt',
                 sep='\t',
                 header=True,
                 index=False,
                 encoding='utf-8')