
#%%
from IPython import get_ipython
import os
import gc
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder

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
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
train_labels = pd.read_csv('./input/train_labels.csv')
sample_submission = pd.read_csv('./input/sample_submission.csv')

#%%
#trainとtestのカラムは同じ
print(set(train_df.columns).difference(set(test_df.columns)))
#installationIDは全くかぶらない
print(set(train_df['installation_id'].unique()).intersection(set(test_df['installation_id'].unique())))

#train_labelsに含まれるのはtypeがAssessmentのものだけである
#なぜならば、評価についてのみ予測するから
#それぞれのinstallation_idについて、最後に行われたAssessmentの種類について、それが何回目でパスされたかを答えとする
#＝二種類目以降のAssessmentなのかどうかは重要、一種類だけやってすぐやめる可能性
# %%
print(train_df['installation_id'].unique())
print(test_df['installation_id'].unique())

# %%
