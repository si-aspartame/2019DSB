
#%%
from IPython import get_ipython
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, KFold
import gc
import json
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization


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
# %%
print('Loading')
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
train_labels = pd.read_csv('./input/train_labels.csv')
sample_submission = pd.read_csv('./input/sample_submission.csv')

#%%
train_labels.head()
#sample_submission.head()

#%%
train_df = pd.read_csv('./input/train.csv')

#%%
display(train_df[train_df['type'] == 'Assessment']['title'].unique())
display(train_df[train_df['type'] == 'Assessment']['title'].value_counts().index)
#%%
tr_ev4100=train_df[train_df['event_code']==4110]
tr_ev4100_ases=tr_ev4100[tr_ev4100['type']=='Assessment']
tr_ev4100_ases_inst6a6=tr_ev4100_ases[tr_ev4100_ases['installation_id']=='0006a69f']
display(tr_ev4100_ases_inst6a6)
print(len(tr_ev4100_ases_inst6a6))
display(train_df['event_code'].value_counts())
#%%
train['title_event_code'] = train['title'].astype(str) + '_' + train['event_code'].astype(str)
le_title=LabelEncoder()
le_title.fit_transform(train['title'])
le_world=LabelEncoder()
le_world.fit_transform(train['world'])

# %%
