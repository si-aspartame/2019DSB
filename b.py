import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
submission_list = os.listdir('/kaggle/input')
print(submission_list)
csv_list=['convert-to-regression-bf75e3/submission', \
'convert-to-regression-with-tuning/submission', \
'data-science-bowl-data-minification/Groupk_test_X', \
'data-science-bowl-data-minification/Strat_test_X', \
'simple/submission' \
]
path = '/kaggle/input/'
print('reading...')
csv_files=[]
for c in csv_list:
    print(c)
    df=pd.read_csv(path+'/'+c+'.csv')
    csv_files.append(df)
submit=pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
print('blending...')
mrs=[]
for n, f in enumerate(csv_files):
    print(n)
    mrs.append(f['accuracy_group'])
name_and_mr_dic=dict(zip(csv_list, mrs))
all_meter_reading_df=pd.DataFrame(name_and_mr_dic)
submit['accuracy_group'] = all_meter_reading_df.median(axis=1)
print(submit['accuracy_group'][0], '|', mrs[0][0], '|',mrs[1][0], '|',mrs[2][0],'|',mrs[3][0], '|',mrs[4][0])
submit['accuracy_group'] = submit['accuracy_group'].astype(int)
print('writing...')
submit.to_csv("/kaggle/working/submission.csv", index=False)
print('done!')
