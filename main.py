# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#%%
from IPython import get_ipython
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, minmax_scale
from bayes_opt import BayesianOptimization
from sklearn.metrics import cohen_kappa_score

%matplotlib inline

%load_ext autoreload
%autoreload 2
from DSB_func_ishikawa import *

import warnings
warnings.filterwarnings('ignore')

#カテゴリカルの指定とか
# %%
path = './input/'
print('Loading...')
train = pd.read_csv(f'{path}train.csv')
test = pd.read_csv(f'{path}test.csv')

#%%
# train_df[(train_df['event_code']==4100) & (train_df['type'] == 'Assessment')]
dropping_in_data=[
    'installation_id','game_session', 'event_id', 'squeeze_target', \
    'type', 'event_code', 'event_data', 'timestamp', 'date', \
    'game_time', 'event_count', 'game_session|installation_id', 'world',\
    'title|installation_id']
#correctとincorrectを予測したほうがいいかも
dropping_in_labels=['game_session','installation_id','title','num_correct','num_incorrect', 'accuracy']
print(train.dtypes)
print(dropping_in_data)
print(dropping_in_labels)

#%%
train = transform(train)
title_encoder = LabelEncoder().fit(train['title'])
train['title'] = title_encoder.transform(train['title'])

#%%
train = data_squeeze(train,mode='train')

#%%
labels = pd.read_csv('./input/train_labels.csv')
labels['game_session|installation_id'] = labels['game_session'] +'|'+ labels['installation_id']
labels = labels.drop(columns=dropping_in_labels, axis=1)

#%%
train = pd.merge(train, labels, on='game_session|installation_id')
train = train.drop(columns=dropping_in_data, axis=1)
del labels, dropping_in_labels
gc.collect()

#%%
test = transform(test)
test['title'] = title_encoder.transform(test['title'])
test = data_squeeze(test, mode='test')
test = test.drop(columns=dropping_in_data, axis=1)

#%%
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

#%%
categoricals = ['title', 'hour', 'weekday', 'exp_assess', 'installation_id->hour.mean']
def bo_run_lgb(n_estimators, subsample, learning_rate, feature_fraction, train=train, test=test):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    target = train['accuracy_group']
    train = train.drop(['accuracy_group'], axis=1)
    oof_pred = np.zeros((len(train)))
    y_pred = np.zeros((len(test)))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train, target)):
        #print('Fold {}'.format(fold + 1))
        x_train, x_val = train.loc[tr_ind], train.loc[val_ind]
        y_train, y_val = target.loc[tr_ind], target.loc[val_ind]
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

        params = {
            'n_estimators':int(n_estimators),
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': subsample,
            'subsample_freq': 1,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'early_stopping_rounds': 100,
            'verbose': -1
            }

        model = lgb.train(params, train_set, early_stopping_rounds = 50, valid_sets=[train_set, val_set], verbose_eval=False)
        oof_pred[val_ind] = model.predict(x_val)
        #print('Partial score of fold {} is: {}'.format(fold, eval_qwk_lgb_regr(y_val, oof_pred[val_ind])[1]))
        y_pred += model.predict(test) / 5
    loss_score = cohen_kappa_score(target, np.round(oof_pred), weights='quadratic')#trainのロス
    result = pd.Series(np.argmax(oof_pred))
    #print('Our oof cohen kappa score is: ', loss_score)
    #print(result.value_counts(normalize = True))
    del kf, target, train, oof_pred, y_pred, x_train, x_val, y_train, y_val, train_set, val_set, model, result
    gc.collect()
    return -(loss_score)#model, y_pred

#%%
pbounds = {
    'n_estimators':(5000, 50000),
    'subsample': (0.5, 0.9),
    'learning_rate': (0.001, 0.01),
    'feature_fraction': (0.5, 0.9),
    }
optimizer = BayesianOptimization(f=bo_run_lgb, pbounds=pbounds)
optimizer.maximize(init_points=1, n_iter=1)
max_param = optimizer.max['params']
del optimizer
gc.collect()
#%%
def final_run_lgb(train=train, test=test, max_param=optimizer.max['params']):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    target = train['accuracy_group']
    train = train.drop(['accuracy_group'], axis=1)
    oof_pred = np.zeros((len(train)))
    y_pred = np.zeros((len(test)))
    print(max_param)
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train, target)):
        print('Fold {}'.format(fold + 1))
        x_train, x_val = train.loc[tr_ind], train.loc[val_ind]
        y_train, y_val = target.loc[tr_ind], target.loc[val_ind]
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

        params = {
            'n_estimators':int(max_param['n_estimators']),
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': max_param['subsample'],
            'subsample_freq': 1,
            'learning_rate': max_param['learning_rate'],
            'feature_fraction': max_param['feature_fraction'],
            'lambda_l1': 1,
            'lambda_l2': 1,
            'early_stopping_rounds': 100,
            'verbose': -1
            }

        model = lgb.train(params, train_set, early_stopping_rounds = 50, valid_sets=[train_set, val_set], verbose_eval=1000)
        oof_pred[val_ind] = model.predict(x_val)
        print('Partial score of fold {} is: {}'.format(fold, cohen_kappa_score(y_val, np.round(oof_pred[val_ind]))))
        y_pred += model.predict(test) / 5
    loss_score = cohen_kappa_score(target, np.round(oof_pred), weights='quadratic')#trainのロス
    result = pd.Series(np.argmax(oof_pred))
    print('Our oof cohen kappa score is: ', loss_score)
    print(result.value_counts(normalize = True))
    return model, y_pred
#%%
model, prediction = final_run_lgb(train, test)
importance = pd.DataFrame(model.feature_importance(), index=test.columns, columns=['importance']).sort_values('importance', ascending=False)
display(importance)
del train, test, model
#%%
#クラスタリングでもいいかも
sub = pd.read_csv(f'{path}sample_submission.csv')
sub['accuracy_group'] = np.round(prediction)
sub.to_csv('submission.csv', index = False)
display(sub['accuracy_group'].value_counts())
sub['accuracy_group'].hist(log=True)
print(sub.dtypes)
print('done!')
