import pandas as pd


sub_rf_hpt = pd.read_csv(r'sub\submission_rf_hpt.csv')
sub_rf_hpt_sfm = pd.read_csv(r'sub\submission_rf_hpt_sfm.csv')
sub_lr = pd.read_csv(r'sub\submission_lr_7_17.csv')


sub = pd.read_csv(r'input\gender_submission.csv')
sub['Survived'] = sub_rf_hpt['Survived']+sub_rf_hpt_sfm['Survived']+sub_lr['Survived']

# print(sub['Survived'].value_counts())
sub['Survived'] = (sub['Survived'] >= 2).astype(int)
sub.to_csv(r'sub\submission_em.csv', index=False)
