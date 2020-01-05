
import numpy as np
import pandas as pd
import itertools
import os, gc
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from IPython import get_ipython
import matplotlib.pyplot as plt
import lightgbm as lgb
from IPython.core.display import display
from tqdm import tqdm

def transform(train_df, test_df):
    all_df = pd.concat([train_df, test_df])
    all_df['date'] = pd.to_datetime(all_df.timestamp)
    all_df['hour'] = all_df['date'].dt.hour#何時か
    #tryは評価対象のAssessmentが1になる列
    
    all_df['try'] = 0
    print('making try(Bird)...')
    # Bird Measurer Assessmentは4100と4110の二回があり、4110がトライ
    bool_for_bird = (all_df['title'] == 'Bird Measurer (Assessment)') & (all_df['event_code'] == 4110)
    all_df.loc[bool_for_bird, 'try'] = 1
    print(all_df['try'].value_counts())

    print('making try(Other)...')
    # Bird Measurer以外は4100がトライ
    bool_for_other = (all_df['event_code'] == 4100) & (all_df['type'] == 'Assessment') & ~(all_df['title'] == 'Bird Measurer (Assessment)')
    all_df.loc[bool_for_other, 'try'] = 1
    print(all_df['try'].value_counts())
    #tryが1であるうちの、最後のgame_session群

    print('adding last_game_session...')
    all_df = adding_last_game_session(all_df)

    ##############最後のgame_sessionに関する特徴量
    #今回は最終的な予測対象の1行がtrueかどうかを予測するモデルにする
    transformed_train = all_df[:len(train_df)]
    transformed_test = all_df[len(train_df):]
    return transformed_train, transformed_test

    
def adding_last_game_session(df):
    all_try = df[df['try'] == 1]
    id_list = all_try['installation_id'].unique()#全データのinstallation_idのユニークな値
    print(id_list)
    #installation_idで最後のgame_sessionの値を返す辞書
    print('making dictionary...')
    final_try_dic =  {i : max(all_try[all_try['installation_id']==i]['game_session'].index.values) for i in id_list}
    last_game_session_dic = {i : all_try[all_try['installation_id']==i]['game_session'].iat[-1] for i in id_list}
    #try==1で、かつdicでその行のinstallation_idに対応するgame_sessionと等しい場合1
    #そうでなければ0
    print('mapping final_try & last_game_session...')
    #installation_idから、そのinstallation_idのトライのうち、最大のインデックス＝最後のトライを返す
    df['final_try'] = df['installation_id'].map(final_try_dic)
    #instllation_idから、そのinstallation_idのトライのうち、最後のトライを含むgame_sessionを返す
    df['last_game_session'] = df['installation_id'].map(last_game_session_dic)
    print('adding about sessions is done!')
    return df



