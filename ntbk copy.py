#%%
from IPython import get_ipython
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from sklearn.preprocessing import LabelEncoder, minmax_scale
import copy
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

init_point01_23 = 100
n_iter01_23 = 100
s_init_point = 100
s_n_iter = 100
path = './input/'#'../input/data-science-bowl-2019/'

#カテゴリカルの指定とか他の特徴の追加（特にclip）
#外れ値

#%%
########################################################################################################
#Functions
########################################################################################################
def accumulated_features(df):
    df.loc[df['event_data'].str.contains('true'), 'T'] = 1
    df.loc[df['event_data'].str.contains('false'), 'F'] = 1
    df.loc[df['T']==1, 'true_count'] = df[df['T']==1].groupby(['T']).cumcount()
    df.loc[df['F']==1, 'false_count'] = df[df['F']==1].groupby(['F']).cumcount()
    current_t = 0
    current_f = 0
    for n, (t, f) in enumerate(zip(df['true_count'], df['false_count'])):
        if current_t < t:
            current_t = t
        df['true_count'].iloc[n] = current_t
        if current_f < f:
            current_f = f
        df['false_count'].iloc[n] = current_f
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def type_check(all_df):
    print('type_check')
    print('installation_id')
    for x in all_df['installation_id'].unique():
        if not type(x) == str:
            print(str(type(x)))
    print('game_session')
    for x in all_df['game_session'].unique():
        if not type(x) == str:
            print(str(type(x)))
    print('title')
    for x in all_df['title'].unique():
        if not type(x) == str:
            print(str(type(x)))
    return

def groupon(df, group_col, calc_col, method):
    dic=df.groupby(group_col)[calc_col].agg([method])[method].to_dict()
    return df[group_col].map(dic)

def game_session_feature(all_df):
    #時系列に整列していないとバグる
    print('game_session_feature:')
    all_df['exp_assess'] = 0
    all_df['session_title'] = ''
    all_df['session_type'] = ''
    all_df['final_game_time'] = 0
    df = all_df[all_df['type']=='Assessment']#Assessmentのみに絞る
    df = df[df['event_code']==2000]#gamestartに絞る
    df['installation_id|title'] = df['installation_id'].str.cat(df['title'], sep='|')
    unique_ea1 = df['installation_id|title'].unique()#index順のarray
    for u in tqdm(unique_ea1):
        #installation_idとtitleの組み合わせでイテレーション
        df_it = df[df['installation_id|title']==u]
        #game_sessionから、それが現在のイテレーションの中で何番目に固有か(何度目か)
        itg_unique_func = lambda g: int(np.where(df_it['game_session'].unique() == g)[0])#tupleからintへ
        session_title_func = lambda g: df_it.loc[df_it['game_session'] == g, 'title'].iloc[0]
        session_type_func = lambda g: df_it.loc[df_it['game_session'] == g, 'type'].iloc[0]
        #installation_idとtitleの組み合わせで固有のgame_sessionを抽出し、mapでそれが何番目のgame_sessionかを返す
        df.loc[df['installation_id|title']==u, 'exp_assess'] = df_it['game_session'].map(itg_unique_func)
        df.loc[df['installation_id|title']==u, 'session_title'] = df_it['game_session'].map(session_title_func)
        df.loc[df['installation_id|title']==u, 'session_type'] = df_it['game_session'].map(session_type_func)
    all_df.loc[(all_df['type']=='Assessment') & (all_df['event_code']==2000), 'exp_assess'] = df['exp_assess'].values
    all_df.loc[(all_df['type']=='Assessment') & (all_df['event_code']==2000), 'session_title'] = df['session_title'].values
    all_df.loc[(all_df['type']=='Assessment') & (all_df['event_code']==2000), 'session_type'] = df['session_type'].values
    del df, df_it
    gc.collect()
    
    return all_df

def transform(all_df):
    #installation_idに対するtitleのnuniqueと
    #その予測対象までのinstallation_idに対するcollect:falseの数
    #その予測対象までのinstallation_idに対するcollect:trueの数
    #その予測対象までのinstallation_idに対する上記二つからのaccuracy_group

    #all_df = all_df[all_df['type'].isin(['Assessment', 'Game'])]
    
    title_dic = {'Mushroom Sorter (Assessment)':'Mushroom Sorter', 'Bird Measurer (Assessment)':'Bird Measurer', 'Cauldron Filler (Assessment)':'Cauldron Filler', 'Cart Balancer (Assessment)':'Cart Balancer', 'Chest Sorter (Assessment)':'Chest Sorter'}
    all_df.loc[all_df['type']=='Assessment', 'title'] = all_df.loc[all_df['type']=='Assessment', 'title'].map(title_dic)
    #Assessmentに対して前のGameのかかった時間を出す(Gameは0)
    #event_countを使わないと進んでないのに高速で間違いまくるやつのスピードが早くなる
    all_df['timestamp']=pd.to_datetime(all_df['timestamp'])
    all_df['date'] = pd.to_datetime(all_df.timestamp)
    all_df['hour'] = all_df['date'].dt.hour#何時か
    all_df['weekday'] = all_df['date'].dt.weekday

    #↓ラベルをくっつける用+特徴量生成用
    all_df['game_session|installation_id'] = all_df['game_session'].str.cat(all_df['installation_id'], sep='|')
    #そのgame_sessionの合計トランザクション数->startから未来の数がわかってしまうためNG
    all_df = game_session_feature(all_df)

    #それぞれのgame_sessionの最終的にかかった時間（type=GameなのでStart=2000まで）
    all_df.loc[all_df['type']=='Assessment', 'game_time'] = 0#高速化用、最終的にgame_time自体は落とすため問題ない
    all_df['final_game_time'] = groupon(all_df, 'game_session|installation_id', 'game_time', 'max')
    fgt_mean = all_df.loc[~(all_df['final_game_time']==0), 'final_game_time'].median()
    all_df['final_game_time'] = all_df['final_game_time'].fillna(fgt_mean)
    all_df.loc[all_df['final_game_time'] == 0, 'final_game_time'] = fgt_mean
    #そのsessionの経過スピード（type=GameなのでStart=2000まで
    #全てのユーザがそれぞれのゲームにかかる時間の平均
    #特徴量生成用
    all_df['title|installation_id'] = all_df['title'].str.cat(all_df['installation_id'], sep='|')
    #installation_idごとのユニークなタイトルの数
    all_df['installation_id|title.nunique'] = all_df.groupby(['installation_id'])['title'].transform('nunique')
    #平均何時にプレイしているか(installation_idに対してhourが1つだとvarは0)
    #all_df['installation_id->hour.mean'] = groupon(all_df, 'installation_id', 'hour', 'mean').astype(int)
    #all_df['installation_id->hour.var'] = groupon(all_df, 'installation_id', 'hour', 'var')
    #game_session一回につき平均何時間プレイしているか(installation_idに対してgame_sessionが1つだとvarは0)

    #all_df['installation_id->final_game_time.mean'] = groupon(all_df, 'installation_id', 'final_game_time', 'mean').astype(int)
    #all_df['installation_id->final_game_time.var'] = groupon(all_df, 'installation_id', 'final_game_time', 'var')
    
    #それぞれのinstallation_idのトランザクションに対して連番を振る
    all_df['installation_id|transaction_count'] = all_df.groupby(['installation_id']).cumcount()
    #all_df = all_df.drop(['final_game_time'], axis=1)

    print('accumulated_feature:')
    all_df['T'] = 0
    all_df['F'] = 0
    all_df['true_count'] = 0
    all_df['false_count'] = 0
    assess_df=all_df[all_df['type']=='Assessment']
    for iteration_id, df_it in tqdm(assess_df.groupby('installation_id', sort = False)):
        df_it = accumulated_features(df_it)
        assess_df.loc[(assess_df['installation_id']==iteration_id), 'true_count'] = df_it['true_count'].values
        assess_df.loc[(assess_df['installation_id']==iteration_id), 'false_count'] = df_it['false_count'].values
    all_df.loc[all_df['type']=='Assessment', 'true_count']=assess_df['true_count'].values
    all_df.loc[all_df['type']=='Assessment', 'false_count']=assess_df['false_count'].values
    del assess_df
    gc.collect()
    all_df['accum_accuracy'] = all_df['true_count'] / (all_df['false_count']+1)
    display(all_df['accum_accuracy'].value_counts())
    ##############最後のgame_sessionに関する特徴量
    #今回は最終的な予測対象の1行がtrueかどうかを予測するモデルにする
    
    return all_df

def data_squeeze(input_df, mode):
    print('data_squeeze:')
    input_df['squeeze_target'] = 0
    input_df['temp_index']=input_df.index.values
    df = input_df[input_df['type']=='Assessment']#アセスメントだけに処理を絞る
    df = df[df['event_code']==2000]#gamestartに絞る
    
    del input_df
    gc.collect()

    if mode=='test':
        print('test_squeeze:')
        df['installation_id|game_session'] = 0#使わないがあとでdropするため
        id_list = df['installation_id'].unique()
        for u in tqdm(id_list):
            df_it = df[df['installation_id']==u]
            df.loc[df['temp_index']==df_it.iloc[-1]['temp_index'], 'squeeze_target'] = 1
    elif mode=='train':
        print('train_squeeze:')
        df['installation_id|game_session'] = df['installation_id'] +'|'+ df['game_session']
        unique_list = df['installation_id|game_session'].unique()
        for u in tqdm(unique_list):
            df_it = df[df['installation_id|game_session']==u]
            df.loc[df['temp_index']==df_it.iloc[-1]['temp_index'], 'squeeze_target'] = 1
    
    sq_df = df[df['squeeze_target'] == 1].drop(['temp_index', 'installation_id|game_session'], axis=1)
    
    del df, df_it
    gc.collect()

    return sq_df

def processing_data(path):
    print('Loading...')
    train = pd.read_csv(f'{path}train.csv')
    test = pd.read_csv(f'{path}test.csv')
    dropping_in_data=['game_session', 'event_id', 'squeeze_target', \
        'type', 'event_code', 'event_data', 'timestamp', 'date', \
        'game_time', 'event_count', 'game_session|installation_id', 'world',\
        'title|installation_id', 'T', 'F']#,'final_game_time']
    dropping_in_labels=['game_session','installation_id','title','num_correct','num_incorrect', 'accuracy']
    print(dropping_in_data)
    print(dropping_in_labels)

    train = transform(train)
    display(train['final_game_time'].value_counts())
    display(train['session_title'].value_counts())
    display(train['session_type'].value_counts())
    title_encoder = LabelEncoder().fit(train['title'])
    train['title'] = title_encoder.transform(train['title'])

    train = data_squeeze(train,mode='train')

    labels = pd.read_csv(f'{path}train_labels.csv')
    labels['game_session|installation_id'] = labels['game_session'] +'|'+ labels['installation_id']
    labels = labels.drop(columns=dropping_in_labels, axis=1)

    train = pd.merge(train, labels, on='game_session|installation_id')
    train = train.drop(columns=dropping_in_data, axis=1)
    del labels, dropping_in_labels
    gc.collect()

    test = transform(test)
    test['title'] = title_encoder.transform(test['title'])
    test = data_squeeze(test, mode='test')
    test = test.drop(columns=dropping_in_data, axis=1)

    test_one_shot_child = test.loc[test['final_game_time'].isnull(), 'final_game_time'].values
    train_one_shot_child = train.loc[train['final_game_time'].isnull(), 'final_game_time'].values
    
    unique_ids = np.concatenate([train['installation_id'].unique(), test['installation_id'].unique()])
    id_encoder = LabelEncoder().fit(unique_ids)
    train['installation_id'] = id_encoder.transform(train['installation_id'])
    test['installation_id'] =  id_encoder.transform(test['installation_id'])

    unique_session_title = np.concatenate([train['session_title'].unique(), test['session_title'].unique()])
    ust_encoder = LabelEncoder().fit(unique_session_title)
    train['session_title'] = ust_encoder.transform(train['session_title'])
    test['session_title'] =  ust_encoder.transform(test['session_title'])

    unique_session_type = np.concatenate([train['session_type'].unique(), test['session_type'].unique()])
    usp_encoder = LabelEncoder().fit(unique_session_type)
    train['session_type'] = usp_encoder.transform(train['session_type'])
    test['session_type'] =  usp_encoder.transform(test['session_type'])

    label01_23 = train['accuracy_group'].copy()
    label01_23.loc[label01_23.isin([0, 1])] = 0
    label01_23.loc[label01_23.isin([2, 3])] = 1
    return train, test, test_one_shot_child, train_one_shot_child, label01_23

#%%
train, test, test_one_shot_child, train_one_shot_child, label01_23 = processing_data(path)
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
label01_23 = label01_23.to_frame(name='splitted_label')
print(train.columns.values)
print(test.columns.values)
train.to_csv(f'preprocessed_train.csv', index = False)
test.to_csv(f'preprocessed_test.csv', index = False)
label01_23.to_csv(f'preprocessed_label01_23.csv', index = False)
#%%
print(f'ONESHOT:\ntest:{test_one_shot_child}\ntrain:{train_one_shot_child}')
display(train[train['installation_id'].isin(train_one_shot_child)])
del train, test, label01_23
gc.collect()
#%%
########################################################################################################
#LightGBM
########################################################################################################
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from bayes_opt import BayesianOptimization
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
train = pd.read_csv(f'preprocessed_train.csv')
test = pd.read_csv(f'preprocessed_test.csv')
label01_23 = pd.read_csv(f'preprocessed_label01_23.csv')
display(label01_23.head())
categoricals = ['title', 'hour', 'weekday', 'exp_assess', 'session_title', 'session_type']
#%%
def opt_lgb(num_leaves, max_depth, bagging_fraction, bagging_freq, colsample_bytree, learning_rate, train=train, test=test, target=label01_23['splitted_label']):
    kf = GroupKFold(n_splits=5)
    inst_id = train['installation_id']
    train = train.drop(['accuracy_group', 'installation_id'], axis=1)
    oof_pred = np.zeros((len(train)))
    y_pred = np.zeros((len(test)))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train, target, groups=inst_id)):
        x_train, x_val = train.loc[tr_ind], train.loc[val_ind]
        y_train, y_val = target.loc[tr_ind], target.loc[val_ind]
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

        params = {
            'boosting_type': 'gbdt',
            'metric': 'auc',
            'objective': 'binary',
            'n_jobs': -1,
            'seed': 42,
            'num_leaves': int(num_leaves),
            'learning_rate': learning_rate,
            'max_depth': int(max_depth),
            'lambda_l1': 1,
            'lambda_l2': 1,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': int(bagging_freq),
            'colsample_bytree': colsample_bytree,
            'verbose': 0
            }

        model = lgb.train(params, train_set, early_stopping_rounds = 50, valid_sets=[train_set, val_set], verbose_eval=False)
        oof_pred[val_ind] = model.predict(x_val)
    loss_score = cohen_kappa_score(target, np.round(oof_pred), weights='quadratic')#trainのロス
    del kf, target, train, oof_pred, y_pred, x_train, x_val, y_train, y_val, train_set, val_set, model
    gc.collect()
    return loss_score

#%%
pbounds = {
    'num_leaves': (20, 300),
    'max_depth': (3, 40),
    'bagging_fraction': (0.4, 0.9),
    'bagging_freq': (1, 10),
    'colsample_bytree': (0.4, 1),
    'learning_rate': (0.0025, 0.01)
    }
optimizer = BayesianOptimization(f=opt_lgb, pbounds=pbounds)
optimizer.maximize(init_points=init_point01_23, n_iter=n_iter01_23, acq = 'ucb', xi = 0.0, alpha = 1e-6)
param01_23 = optimizer.max['params']
del optimizer
gc.collect()
#|  167      |  0.4118   |  0.9      |  1.0      |  1.0      |  0.01     |  3.0      |  31.41    
#%%
####################################
#####DISTINGUISH 01, 23
####################################
def run_lgb(train, test, in_param, target):
    kf = GroupKFold(n_splits=5)
    inst_id = train['installation_id']
    train = train.drop(['accuracy_group', 'installation_id'], axis=1)
    oof_pred = np.zeros((len(train)))
    test_pred = np.zeros((len(test)))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train, target, groups=inst_id)):
        print(set(train.columns.values)-set(test.columns.values))
        print('Fold {}'.format(fold + 1))
        x_train, x_val = train.loc[tr_ind], train.loc[val_ind]
        y_train, y_val = target.loc[tr_ind], target.loc[val_ind]
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

        params = {
            'boosting_type': 'gbdt',
            'metric': 'auc',
            'objective': 'binary',
            'n_jobs': -1,
            'seed': 42,
            'num_leaves': int(in_param['num_leaves']),
            'learning_rate': in_param['learning_rate'],
            'max_depth': int(in_param['max_depth']),
            'lambda_l1': 1,
            'lambda_l2': 1,
            'bagging_fraction': in_param['bagging_fraction'],
            'bagging_freq': int(in_param['bagging_freq']),
            'colsample_bytree': in_param['colsample_bytree'],
            'verbose': 0
            }
        model = lgb.train(params, train_set, early_stopping_rounds = 50, valid_sets=[train_set, val_set], verbose_eval=1000)
        oof_pred[val_ind] = model.predict(x_val)
        print('Partial score of fold {} is: {}'.format(fold, cohen_kappa_score(y_val, np.round(oof_pred[val_ind]))))
        test_pred += model.predict(test, axis=1) / 5
    loss_score = cohen_kappa_score(target, np.round(oof_pred), weights='quadratic')#trainのロス
    train_pred = pd.Series(oof_pred)
    print('Our oof cohen kappa score is: ', loss_score)
    return model, test_pred, train_pred

#%%
#01_23を予測
test.head()
print('DISTINGUISH: 01|23')
model01_23, test_pred01_23, train_pred01_23 = run_lgb(train, test.drop(['installation_id'], axis=1), param01_23, label01_23)
importance = pd.DataFrame(model01_23.feature_importance(), index=test.drop(['installation_id'], axis=1).columns, columns=['importance']).sort_values('importance', ascending=False)
display(importance)
del model01_23
gc.collect()

#%%
#追加
train['predicted01_23'] = train_pred01_23
test['predicted01_23'] = test_pred01_23
del train_pred01_23, test_pred01_23
gc.collect()
categoricals = ['title', 'hour', 'weekday', 'exp_assess', 'session_title', 'session_type', 'predicted01_23']
#%%
####################################################スタッキング
def stack_opt_lgb(num_leaves, max_depth, bagging_fraction, bagging_freq, colsample_bytree, learning_rate, train=train, test=test, target=train['accuracy_group']):
    kf = GroupKFold(n_splits=5)
    inst_id = train['installation_id']
    train = train.drop(['accuracy_group', 'installation_id'], axis=1)
    oof_pred = np.zeros((len(train)))
    y_pred = np.zeros((len(test)))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train, target, groups=inst_id)):
        x_train, x_val = train.loc[tr_ind], train.loc[val_ind]
        y_train, y_val = target.loc[tr_ind], target.loc[val_ind]
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

        params = {
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'objective': 'regression',
            'n_jobs': -1,
            'seed': 42,
            'num_leaves': int(num_leaves),
            'learning_rate': learning_rate,
            'max_depth': int(max_depth),
            'lambda_l1': 1,
            'lambda_l2': 1,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': int(bagging_freq),
            'colsample_bytree': colsample_bytree,
            'verbose': 0
            }

        model = lgb.train(params, train_set, early_stopping_rounds = 50, valid_sets=[train_set, val_set], verbose_eval=False)
        oof_pred[val_ind] = model.predict(x_val)
    loss_score = cohen_kappa_score(target, np.round(oof_pred), weights='quadratic')#trainのロス
    del kf, target, train, oof_pred, y_pred, x_train, x_val, y_train, y_val, train_set, val_set, model
    gc.collect()
    return loss_score

#%%
stack_pbounds = {
    'num_leaves': (20, 300),
    'max_depth': (3, 40),
    'bagging_fraction': (0.4, 0.9),
    'bagging_freq': (1, 10),
    'colsample_bytree': (0.4, 1),
    'learning_rate': (0.0025, 0.01)
    }
stack_optimizer = BayesianOptimization(f=stack_opt_lgb, pbounds=stack_pbounds)
stack_optimizer.maximize(init_points=s_init_point, n_iter=s_n_iter, acq = 'ucb', xi = 0.0, alpha = 1e-6)
stack_param = stack_optimizer.max['params']
del stack_optimizer
gc.collect()

#%%
def stack_run_lgb(train, test, in_param, target):
    kf = GroupKFold(n_splits=5)
    inst_id = train['installation_id']
    train = train.drop(['accuracy_group', 'installation_id'], axis=1)
    oof_pred = np.zeros((len(train)))
    test_pred = np.zeros((len(test)))
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train, target, groups=inst_id)):
        print('Fold {}'.format(fold + 1))
        x_train, x_val = train.loc[tr_ind], train.loc[val_ind]
        y_train, y_val = target.loc[tr_ind], target.loc[val_ind]
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)

        params = {
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'objective': 'regression',
            'n_jobs': -1,
            'seed': 42,
            'num_leaves': int(in_param['num_leaves']),
            'learning_rate': in_param['learning_rate'],
            'max_depth': int(in_param['max_depth']),
            'lambda_l1': 1,
            'lambda_l2': 1,
            'bagging_fraction': in_param['bagging_fraction'],
            'bagging_freq': int(in_param['bagging_freq']),
            'colsample_bytree': in_param['colsample_bytree'],
            'verbose': 0
            }

        model = lgb.train(params, train_set, early_stopping_rounds = 50, valid_sets=[train_set, val_set], verbose_eval=1000)
        oof_pred[val_ind] = model.predict(x_val)
        print('Partial score of fold {} is: {}'.format(fold, cohen_kappa_score(y_val, np.round(oof_pred[val_ind]))))
        test_pred += model.predict(test) / 5
    loss_score = cohen_kappa_score(target, np.round(oof_pred), weights='quadratic')#trainのロス
    train_pred = pd.Series(oof_pred)
    print('Our oof cohen kappa score is: ', loss_score)
    return model, test_pred, train_pred

#%%
print('stacking:')
stack_model, stack_test_pred, _ = stack_run_lgb(train, test.drop(['installation_id'], axis=1), stack_param, train['accuracy_group'])
importance = pd.DataFrame(stack_model.feature_importance(), index=test.drop(['installation_id'], axis=1).columns, columns=['importance']).sort_values('importance', ascending=False)
display(importance)
del stack_model
gc.collect()

#%%
sub = pd.read_csv(f'{path}sample_submission.csv')
sub['accuracy_group'] = np.round(stack_test_pred).astype(int)
print(f'one_shot_child:{len(sub[sub["installation_id"].isin(test_one_shot_child)])}')
sub.loc[sub['installation_id'].isin(test_one_shot_child), 'accuracy_group'] = 0
sub.to_csv('LGBM.csv', index = False)
display(sub['accuracy_group'].value_counts())
sub['accuracy_group'].hist(log=True)
print(sub.dtypes)
print('done!')
print(f'01_23:{param01_23}')
print(f'stack:{stack_param}')

#%%
print(train.dtypes)
print(train['title'].unique())

#%%
del train, test, sub
gc.collect()

# %%
#################################################################################################################
#Neural Network
#################################################################################################################
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import category_encoders as ce

#%%
#read from csv
train = pd.read_csv(f'preprocessed_train.csv')
test = pd.read_csv(f'preprocessed_test.csv')
target = train['accuracy_group']
train = train.drop(['accuracy_group', 'installation_id'], axis=1)
test = test.drop(['installation_id'], axis=1)

#%%
#one-hot-encoder
encoding_columns = categoricals = ['title', 'hour', 'weekday', 'exp_assess', 'session_title', 'session_type']
z_score_columns = set(train.columns.values) - (set(encoding_columns))
onehot_enc = ce.OneHotEncoder(cols=encoding_columns, handle_unknown='impute')
train = onehot_enc.fit_transform(train)
test = onehot_enc.fit_transform(test)

#%%
#z_score
def z_score(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore
for x in z_score_columns:
    train[x] = z_score(train[x].values)
display(train.head())
for x in z_score_columns:
    test[x] = z_score(test[x].values)
display(test.head())

#%%
num_epochs = 10
early_stopping = 50
column_num = len(train.columns.values)#targetは除く長さ
row_num = len(train)
nn_lr=0.001
wd=0.0001
print(column_num)
print(row_num)

#%%
A = column_num
B = column_num - int(column_num/4)
C = column_num - int(column_num/4)*3
D = 5
print(f'{A},{B},{C},{D},{E}')

class TwoStage(nn.Module):
    def __init__(self):
        super(TwoStage, self).__init__()
        self.first = nn.Sequential(
            nn.Linear(A, B),
            nn.ReLU(),
            nn.Linear(B, C),
            nn.ReLU(),
            nn.Linear(C, D))
        self.second = nn.Sequential(
            nn.Linear(A+E, B),
            nn.ReLU(),
            nn.Linear(B, C),
            nn.ReLU(),
            nn.Linear(C, 1),
            nn.Softmax())#4種の確率
    
    def forward(self, x):
        y = self.first(x)
        z = torch.cat((x, y), axis=1)
        print(z)
        print(z.shape)
        output = self.second(z)
        return output

if path == './input/':
    model = TwoStage().cuda()
else:
    model = TwoStage()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=nn_lr, weight_decay=wd)#weight_decay

in_train = torch.from_numpy(train.values)#np_srをテンソルにしたもの
in_test = torch.from_numpy(test.values)
in_target = torch.from_numpy(target.values)
tr_dataset = torch.utils.data.TensorDataset(in_train.float(), in_target.float())
print(f"in_tensor:{in_train.size()}")
del train, test, target
gc.collect()
# %%
all_loss=[]
best_loss=99999
es_count=0
model.train()
for epoch in range(1, num_epochs+1):
    for data, target in DataLoader(tr_dataset, batch_size=20, shuffle=True):
        batch = data
        if path == './input/':
            batch = Variable(batch).cuda()
        else:
            batch = Variable(batch)
        if path == './input/':
            target = Variable(target).cuda()
        else:
            target = Variable(target)
        # ===================forward=====================
        output = model(batch)
        loss = criterion(output, target)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if loss.data.item() < best_loss:
        print('[BEST] ', end='')
        torch.save(model.state_dict(), f'{epoch}.pth')
        best_loss = loss.data.item()
        es_count = 0
    es_count += 1
    print(f'epoch [{epoch}/{num_epochs}], loss:{loss.data.item()}')
    all_loss.append([epoch, loss.data.item()])
    if es_count == early_stopping:
        print('early stopping!')
        break#early_stopping

#%%
best_iteration = np.argmin([x[1] for x in all_loss])
print(f'best_iteration:{all_loss[best_iteration]}')

# %%
if path == './input/':
    model = best_model().cuda()
else:
    model = best_model()
best_model.load_state_dict(torch.load(f'{all_loss[best_iteration][0]}.pth'))
result=np.empty((0,1))
model.eval()
for n, data in enumerate(DataLoader(in_test, batch_size=1, shuffle=False)):#シャッフルしない
    print(f'TEST:{n}')
    batch = data
    batch = batch.reshape(batch.size(0)*3)
    batch = Variable(batch).cuda()
    # ===================forward=====================
    output = best_model(batch)
    if path == './input/':
        result = np.vstack([result, output.data.cpu().numpy()[0]])
    else:
        result = np.vstack([result, output.data.numpy()[0]])
del mode, best_model, 
#%%
nn_result = result.reshape(-1)
sub2 = pd.read_csv(f'{path}sample_submission.csv')
sub2['accuracy_group'] = np.round(nn_result).astype(int)
sub2.to_csv('NN.csv', index = False)
display(sub2['accuracy_group'].value_counts())
sub2['accuracy_group'].hist(log=True)