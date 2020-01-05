# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#%%
from IPython import get_ipython
import numpy as np
import pandas as pd
# pd.set_option('display.max_columns', 1000)
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
# from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, KFold
import gc
import json
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization

%matplotlib inline

%load_ext autoreload
%autoreload 2
from DSB_func_ishikawa import *

import warnings
warnings.filterwarnings('ignore')


#%%
#・event_idは、あるユーザのあるゲームに対して一つ割り当てられる。
#  一度閉じてもevent_idは変更されない、game_sessionは変更される。
#・game_sessionは、ゲームの種別ごとの、毎回のプレイ(開いてから閉じるまで)に与えられるIDであり
#  一度閉じると同じユーザであっても変更される。
#・timestampはtimestampだがstring型なので変換が必要
#・event_dataは謎のjson
#・installation_idでユーザを区別している。
#・event_countはevent_dataから抽出されるゲームセッション内の増加値で、1が引かれている。
#・event_codeはそれぞれのtitleで固有のイベントの種類だが、2000は常にゲームスタートを表す。
#・game_timeはゲーム開始からの秒数(ms)
#・titleはゲームタイトル
#・typeはゲームまたはビデオのタイプ
#  'Game', 'Assessment(評価)', 'Activity(活動？)', 'Clip(ビデオクリップ？)'
#・worldはゲームやビデオが属するセクション、 
#  'NONE' (at the app's start screen), TREETOPCITY' (Length/Height),
#  'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight)

# それぞれのinstallation_idの最後のAssesment群について、
# そのAssesmentが
# 3:最初のトライでtrue
# 2:二回目のトライでtrue
# 1:三回目以降のトライでtrue
# 0:正解できなかった
# を最終的に予測する
# そのためには最後のAssesmentの最終トライでcorrectがtrueかどうかを確認(num_correct)し
# その後、そのAssessment群でいくつ不正解したかを確認(num_incorrect)する。
# そこから最終的なAccuracy_groupを算出する

# Bird MeasurerのAssessmentは二段階で、4100のあとに4110が続く。この場合予測するのは4110の方。
# 他は4100一回のみ

#Assessmentとそれ以外で指標を分ける

#全部横に並べて入れるか、最終try以前のデータから作った特徴量を付加した1行で予測をするか
#installation_idごとの最後のAssessment群
#今までのAssessmentのTrue回数

#testデータのそれぞれのinstallation_idの最後のトランザクションは常に2000でAssessmentが始まる
#その結果を予測する

#その子供が最後の2000で始まるAssessmentで良い結果を残すにはどういう条件が必要か

#過去に同じAssessmentをプレイしている、またその結果
#今までのAssessmentのプレイ回数
#時間をかけてやっている、かけずにやっている
#clipはスキップ可能か？可能ならスキップしているか

# %%
print('Loading')
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
# train_labels_df = pd.read_csv('./input/train_labels.csv')
# sample_submission = pd.read_csv('./input/sample_submission.csv')


#%%


#%%
print(train_df.shape)
new_train_df, new_test_df = transform(train_df, test_df)
print(train_df.shape)

#%%
train_df.info()

#%%
if train_df['event_id'].all() == new_train_df['event_id'].all():
    print('OKです。')
#%%
display(train_df[train_df['event_data'].str.contains('"correct":true', na=False)].query("type=='Assessment' & event_code==4100"))

#%%
display(train_df[train_df['type'] == 'Assessment']['title'].value_counts().index)
train_df.info()

#%%
display(train_df['installation_id'].value_counts())
train_df['installation_id'].value_counts().hist(log=True)

# %%
