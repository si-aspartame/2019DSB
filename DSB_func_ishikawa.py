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
import warnings
warnings.filterwarnings('ignore')


def accumulated_features(df):
    df.loc[df['event_data'].str.contains('true'), 'true'] = 1
    df.loc[df['event_data'].str.contains('false'), 'false'] = 1
    current_t = 0
    current_f = 0
    for t, f in zip(df['true'], df['false']):
        if current_t < t:
            current_t = t
        t = current_t
        if current_f < f:
            current_f = f
        f = current_f
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



def convert(sub):
    dic={0:0, 1:3, 2:2}
    sub.loc[sub['(num_incorrect+1)*num_correct']==0, 'accuracy_group'] = 0 #(n+1)*0
    sub.loc[sub['(num_incorrect+1)*num_correct']==1, 'accuracy_group'] = 3 #(0+1)*1
    sub.loc[sub['(num_incorrect+1)*num_correct']==2, 'accuracy_group'] = 2 #(1+1)*1
    sub['accuracy_group'].fillna(1)
    return sub['accuracy_group']

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
        #installation_idとtitleの組み合わせで固有のgame_sessionを抽出し、mapでそれが何番目のgame_sessionかを返す
        df.loc[df['installation_id|title']==u, 'exp_assess'] = df_it['game_session'].map(itg_unique_func)
    all_df.loc[(all_df['type']=='Assessment') & (all_df['event_code']==2000), 'exp_assess'] = df['exp_assess'].values
    del df, df_it
    gc.collect()
    
    return all_df

def transform(train_df, test_df):
    #installation_idに対するtitleのnuniqueと
    #その予測対象までのinstallation_idに対するcollect:falseの数
    #その予測対象までのinstallation_idに対するcollect:trueの数
    #その予測対象までのinstallation_idに対する上記二つからのaccuracy_group
    train_df = train_df[train_df['type'].isin(['Assessment', 'Game'])]
    test_df = test_df[test_df['type'].isin(['Assessment', 'Game'])]
    train_length = len(train_df)
    all_df = pd.concat([train_df, test_df])

    del train_df, test_df
    gc.collect()
    
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
    session_count_dic = all_df.loc[all_df['type']=='Game', 'game_session|installation_id'].value_counts().to_dict()
    all_df['session_count'] = all_df['game_session|installation_id'].map(session_count_dic)
    count_mean = all_df.loc[~(all_df['session_count']==0), 'session_count'].mean()#0でないものの平均
    all_df.loc[all_df['session_count']==0, 'session_count'] = count_mean 
    all_df['session_count'] = all_df['session_count'].fillna(count_mean)
    all_df = game_session_feature(all_df)

    del session_count_dic
    gc.collect()
    
    #それぞれのgame_sessionの最終的にかかった時間（type=GameなのでStart=2000まで）
    all_df.loc[all_df['type']=='Assessment', 'game_time'] = 0#高速化用、最終的にgame_time自体は落とすため問題ない
    all_df['final_game_time'] = groupon(all_df, 'game_session', 'game_time', 'max')
    fgt_mean = all_df.loc[~(all_df['final_game_time']==0), 'final_game_time'].mean()
    all_df['final_game_time'] = all_df['final_game_time'].fillna(fgt_mean)
    all_df.loc[all_df['final_game_time'] == 0, 'final_game_time'] = fgt_mean
    #そのsessionの経過スピード（type=GameなのでStart=2000まで）
    all_df['session_speed'] = all_df['final_game_time'] / all_df['session_count']
    #全てのユーザがそれぞれのゲームにかかる時間の平均
    all_df['(A)title->session_speed.mean'] = groupon(all_df, 'title', 'session_speed', 'mean')
    #特徴量生成用
    all_df['title|installation_id'] = all_df['title'].str.cat(all_df['installation_id'], sep='|')


    #all_df['title|installation_id.nunique'] = all_df['title|installation_id'].nunique()
    
    
    #それぞれのinstallation_idのスピードの平均
    all_df['(B)title|installation_id->session_speed.mean'] = groupon(all_df, 'title|installation_id', 'session_speed', 'mean')
    #平均と比べてそのゲームを進めるのが早いか
    all_df['(A)-(B)'] = all_df['(A)title->session_speed.mean'] - all_df['(B)title|installation_id->session_speed.mean']
    #平均何時にプレイしているか(installation_idに対してhourが1つだとvarは0)
    all_df['installation_id->hour.mean'] = groupon(all_df, 'installation_id', 'hour', 'mean').astype(int)
    all_df['installation_id->hour.var'] = groupon(all_df, 'installation_id', 'hour', 'var')
    #game_session一回につき平均何時間プレイしているか(installation_idに対してgame_sessionが1つだとvarは0)
    all_df['installation_id->final_game_time.mean'] = groupon(all_df, 'installation_id', 'final_game_time', 'mean').astype(int)
    all_df['installation_id->final_game_time.var'] = groupon(all_df, 'installation_id', 'final_game_time', 'var')
    #それぞれのinstallation_idのトランザクションに対して連番を振る
    all_df['installation_id|transaction_count'] = all_df.groupby(['installation_id']).grouper.group_info[0]
    all_df['world'] = LabelEncoder().fit_transform(all_df['world'])
    all_df = all_df.drop(['session_speed', 'session_count', 'final_game_time'], axis=1)

    print('accumulated_feature:')
    all_df['true'] = 0
    all_df['false'] = 0
    assess_df=all_df[all_df['type']=='Assessment']
    for iteration_id, df_it in tqdm(assess_df.groupby('installation_id', sort = False)):
        df_it = accumulated_features(df_it)
        assess_df.loc[(assess_df['installation_id']==iteration_id), 'true'] = df_it['true'].values
        assess_df.loc[(assess_df['installation_id']==iteration_id), 'false'] = df_it['false'].values
    all_df.loc[all_df['type']=='Assessment', 'true']=assess_df['true'].values
    all_df.loc[all_df['type']=='Assessment', 'false']=assess_df['false'].values
    del assess_df
    gc.collect()
    display(all_df['true'].value_counts())
    all_df['accum_accuracy'] = all_df['true'] / (all_df['false']+1) 
    ##############最後のgame_sessionに関する特徴量
    #今回は最終的な予測対象の1行がtrueかどうかを予測するモデルにする
    transformed_train = all_df[:train_length]
    transformed_test = all_df[train_length:]

    del all_df
    gc.collect()
    
    return transformed_train, transformed_test

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
    sq_df['title'] = LabelEncoder().fit_transform(sq_df['title'])
    
    del df, df_it
    gc.collect()

    return sq_df