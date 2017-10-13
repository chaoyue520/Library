#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import gc
import re

reload(sys)
sys.setdefaultencoding('utf8')

#记录程序运行时间
import time 
start_time = time.time()


# read data 

na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']

data_set = pd.read_table('/home/fsg/jiwenchao/ac_class/data/data_set_0705.txt',sep = '\t',na_values = na_values)


# split train valid oot dataset 
train_set = data_set[(data_set.data_set_tag == 1)]
valid_set = data_set[(data_set.data_set_tag == 2)]
oot_set = data_set[(data_set.data_set_tag == 3)]


#概览数据
data_set['risk_tag_union'].value_counts()
train_set.groupby(['risk_tag_union'])['passid'].count()
valid_set.groupby(['risk_tag_union'])['passid'].count()
oot_set.groupby(['risk_tag_union'])['passid'].count()


# selected model vars 
col_x_old = list(data_set.columns)

#临时删除关键字段
remove_vars=['passid','sessionid','data_set_tag'
,'sessionid.1','risk_tag_union','paytype_credit_pay','uid_base_payamt']


#只保留最终进入模型的变量
col_x = [x for x in col_x_old if x not in remove_vars]

#逻辑判断
len(col_x) == len(col_x_old)-len(remove_vars)


#释放内存
del data_set
print gc.collect


# to DMatrix
train_data = train_set[col_x].as_matrix()
train_label = train_set['risk_tag_union'].as_matrix()
dtrain = xgb.DMatrix(train_data, label = train_label)

valid_data = valid_set[col_x].as_matrix()
valid_label = valid_set['risk_tag_union'].as_matrix()
dvalid = xgb.DMatrix(valid_data, label = valid_label)

oot_data = oot_set[col_x].as_matrix()
oot_label = oot_set['risk_tag_union'].as_matrix()
doot = xgb.DMatrix(oot_data, label = oot_label)


#模型数据分配
evallist = [(dtrain, 'train'), (dvalid, 'valid'), (doot, 'oot')]


# parameters
param = { 'objective': 'binary:logistic',    #定义学习任务及相应的学习目标
          'eval_metric': 'auc',      #校验数据所需要的评价指标
          'max_depth': 3,            #避免过拟合，过大则会学到更多的局部特征，易导致过拟合
          'learning_rate': 0.01,     #可用来防止过拟合，eta，学习速率，更新过程中用到的收缩步长
          'min_child_weight': 50,   #子节点中最小的样本权重和，调高可以避免过拟合，越大算法越conservative
          'silent': 1,               #取1时表示以缄默方式运行，不打印运行时信息，取0时表示打印出运行时信息。
          #'lambda':1,               #L2正则化权重，减少过拟合
          #'alpha': 1,               #L1正则化权重，减少过拟合 
          'gamma': 0.8,             #值越大，算法越保守
          'max_delta_step': 1,       #限制每棵树权重改变的最大步长
          #'subsample': 0.95,         #避免过拟合，但是如果过小，则易导致欠拟合，子样本占整个样本集合的比例，可防止过拟合
          #'colsample_bytree': 0.95,  #避免过拟合，但是如果过小，则易导致欠拟合，在建立树时对特征采样的比例，可防止过拟合
          #'scale_pos_weight': 1,    #类别十分不平衡
          'seed': 1                  #随机数的种子。缺省值为0
          }


# 迭代次数 the max number of iterations
num_rounds=950



# 释放内存
del data_set
del train_data
del valid_data
del oot_data

gc.set_debug(gc.DEBUG_STATS|gc.DEBUG_LEAK)
print gc.collect()


#tunning the model
bst = xgb.train(param,dtrain,num_boost_round = num_rounds , evals = evallist)


print "best best_ntree_limit:",bst.best_ntree_limit 

#输出运行时间
endtime = time.time()
cost_time = endtime - start_time
print "xgboost success!",'\n',"cost time:",cost_time,"(s)"

############################################################################ load model ##############################################################################
######## 思考：save_model和dump_model的区别
#1、Both functions save_model and dump_model save the model, the difference is that in dump_model you can save feature name and save tree in text format.
#2、The load_model will work with model from save_model.


#1、save model 
xpath='/home/fsg/jiwenchao/ac_class/data'
model_name='/01_ac_class_fraud.model'

bst.save_model(xpath+model_name)


############################################################################ 输出重要变量 ##############################################################################
######## 1、dump model with feature map
######## 2、利用dump_model函数，生成tree map

#1、dump model with feature map
feat_map_df = pd.DataFrame({'id': [i for i in range(len(col_x))]})
feat_map_df = feat_map_df.assign(feat_name = col_x)
feat_map_df = feat_map_df.assign(type = ['q' for i in range(len(col_x))])
feat_map_df.to_csv('/home/fsg/jiwenchao/ac_class/data/feat_map_ac_class_fraud.txt',sep = '\t',header = False,index = False,encoding='utf-8')

#2、dump_model函数源码：https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/core.py
#报错1：ascii code can't encode characters in position 131-136  ordinal not in range(128)
#上述错误解决方案：reload(sys)   sys.setdefaultencoding('utf8')
bst.dump_model('/home/fsg/jiwenchao/ac_class/data/feat_map_ac_class_fraud.dump.raw.txt', '/home/fsg/jiwenchao/ac_class/data/feat_map_ac_class_fraud.txt')


############################################################################ 输出重要变量 ##############################################################################
######## 1、feature importance：特征评分可以看成是被用来分离决策树的次数
######## 2、get_fscore函数源码解释：https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/core.py
######## 3、组合成数据框的形式，并按照score排序
######## 4、输出到本地txt文件


#1、变量名称以及重要性得分字典表
fscore_dict = bst.get_fscore(fmap = '/home/fsg/jiwenchao/ac_class/data/feat_map_ac_class_fraud.txt')

#2、字典表转化为数据框的形式，包括key，value和value的标准化%
features = []
scores = []
for key in fscore_dict:
     features.append(key)
     scores.append(fscore_dict[key])

#重要性归一化，保留三位有效数字
ratio=[]
for key in fscore_dict:
     ratio.append(round(fscore_dict[key]*1.0/sum(scores)*100.0,3))

#3、组合成数据框的形式，并按照score排序
fscore_df = pd.DataFrame({'features': features, 'scores': scores, 'ratio(%)':ratio})
fscore_df=fscore_df.sort_values(by=['scores'],ascending=False,inplace=False)

#4、输出到本地txt文件
fscore_df.to_csv('/home/fsg/jiwenchao/ac_class/data/feature_importance_ac_class_fraud.txt',sep = '\t', header = True, index = False,encoding = 'utf-8')



###############################################################################################################################################################
######################################################################### 模型分修正：data_set_score #####################################################################
######## 1、

xpath='/home/fsg/jiwenchao/ac_class/data'
model_name='/01_ac_class_fraud.model'

# read data 

na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']

data_set = pd.read_table('/home/fsg/jiwenchao/ac_class/data/data_set_0705.txt',sep = '\t',na_values = na_values)

# selected model vars 
col_x_old = list(data_set.columns)

#临时删除关联字段
remove_vars=['passid','sessionid','data_set_tag','sessionid.1','risk_tag_union','paytype_credit_pay','uid_base_payamt']

col_x = [x for x in col_x_old if x not in remove_vars]


data_set_all = data_set[col_x].as_matrix()
data_prob_value = xgb.DMatrix(data_set_all)


# load model 
bst = xgb.Booster()
bst.load_model(xpath+model_name)



#保留三列对应值
pred_df = data_set[['sessionid','passid','risk_tag_union']]
pred_df['score'] = bst.predict(data_prob_value)  #添加一列score，注意ddafen数据的类型，DMatrix


#保留数据，用于模型分修正
pred_df.to_csv('/home/fsg/jiwenchao/ac_class/data/data_set_score.txt',sep = '\t', header = True, index = False,encoding = 'utf-8')


import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression

# compute xbeta_old
xbeta_old = []
for i in range(pred_df.shape[0]):
     score_i = pred_df.score[i]
     if score_i == 1:
          xbeta_old.append([10])
     elif score_i == 0:
          xbeta_old.append([-10]) 
     else:
          xbeta_old.append([math.log(score_i/(1-score_i))])

tag = np.array(pred_df.risk_tag_union)

# use weight to restore sample
xbeta_old_w = []
tag_w = []
weight=219
for i in range(len(tag)):
     if tag[i] == 0:
          xbeta_old_w.extend([xbeta_old[i]] * weight)
          tag_w.extend([tag[i]] * weight)
     else:
          xbeta_old_w.append(xbeta_old[i])
          tag_w.append(tag[i])

# compute coef
lr = LogisticRegression()
lr.fit(xbeta_old_w, tag_w)
a = lr.coef_[0][0]
b = lr.intercept_[0]

a,b

############################################################################### get修正参数a,b END#########################################################################


# 模型分修正参数

a= 1.1181731214010886
b= -5.178439674780841

pred_df = pred_df.assign(xbeta_old = np.log(pred_df.score/(1-pred_df.score)))
pred_df.loc[pred_df.xbeta_old >= 10, 'xbeta_old'] = 10 
pred_df.loc[pred_df.xbeta_old <= -10, 'xbeta_old'] = -10
pred_df = pred_df.assign(xbeta_new = pred_df.xbeta_old*a + b)
pred_df = pred_df.assign(score_new = 1/(1+np.exp(-1 * pred_df.xbeta_new)))



pred_df.to_csv(xpath + '/ac_class_score_new.txt',sep = '\t',header = True,index = False)



##################在最后验证均值的时候，注意扩展抽样数据

data=pred_df
#正负样本分离
data_risk_tag_union_0=data[data.risk_tag_union==0]
data_risk_tag_union_1=data[data.risk_tag_union==1]

#正样本扩展，weight=219
data_risk_tag_union_0_all=pd.concat([data_risk_tag_union_0]*219)

#上下拼接扩展后的正样本和全部负样本
data=pd.concat([data_risk_tag_union_0_all,data_risk_tag_union_1])


#验证是否修正后的均值是否一致
data.score_new.describe()


data.shape
#导出数据用于阈值选定
data[['sessionid', 'passid','risk_tag_union','score_new']].to_csv(xpath + '/ac_class_score_new_last.txt',sep = '\t',header = True,index = False)

###############################################################################  END  ############################################################################


############################################################################# 生成pmml文件 #######################################################################
#备注：生成pmml文件模板   #1，#2，#3
#4、通过jave生成pmml文件
#参考：https://github.com/jpmml/jpmml-xgboost

#1、生成pmml文件模板
java -jar target/converter-executable-1.2-SNAPSHOT.jar --model-input xgboost.model --fmap-input xgboost.fmap --target-name mpg --pmml-output xgboost.pmml

#2、input
/home/fsg/jiwenchao/ac_class/data/01_ac_class_fraud.model
/home/fsg/jiwenchao/ac_class/data/feat_map_ac_class_fraud.txt

#3、output
01_ac_class_fraud.pmml

#4、生成pmml文件
java -jar ~/u_model02/edu/online/target/converter-executable-1.0-SNAPSHOT.jar --model-input /home/fsg/jiwenchao/ac_class/data/01_ac_class_fraud.model --fmap-input /home/fsg/jiwenchao/ac_class/data/feat_map_ac_class_fraud.txt --target-name mpg --pmml-output 01_ac_class_fraud.pmml



#筛选重要变量并作图
col='is_payamt_tail_99'

m=data_set.shape[0]
a=data_set.groupby(col)['risk_tag_union'].count()
b=data_set.groupby(col)['risk_tag_union'].sum()

#交易量占比
a_0=a[0]*1.0/m
b_0=b[0]*1.0/a[0]

#欺诈量占比
a_1=a[1]*1.0/m
b_1=b[1]*1.0/a[1]

#condation1、condation2 占比和欺诈数占比
str(round(a_0*100,2))+'%',str(round(b_0*100,2))+'%'
str(round(a_1*100,2))+'%',str(round(b_1*100,2))+'%'



#predict error
preds = bst.predict(doot)
labels = doot.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

