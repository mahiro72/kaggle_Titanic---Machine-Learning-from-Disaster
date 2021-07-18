import pandas as pd


sub_1 = pd.read_csv(r'sub\submission_lr.csv')
sub_2 = pd.read_csv(r'sub\submission_rf_sfm_th01.csv')
sub_3 = pd.read_csv(r'sub\submission_xgb_opt1.csv')
sub_4 = pd.read_csv(r'sub\submission_xgb_opt3.csv')
sub_5 = pd.read_csv(r'sub\submission_xgb_opt4.csv')
# sub_6 = pd.read_csv(r'sub\submission_xgb_opt2.csv')


sub = pd.read_csv(r'input\gender_submission.csv')
sub['Survived'] = sub_1['Survived']+sub_2['Survived']+sub_3['Survived']+sub_4['Survived']+sub_5['Survived']

sub['Survived'] = (sub['Survived'] >= 4).astype(int)

print(sub.Survived.value_counts())
# sub.to_csv(r'sub\submission_en_4.csv', index=False)
