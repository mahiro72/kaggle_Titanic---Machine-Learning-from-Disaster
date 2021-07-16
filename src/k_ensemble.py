import pandas as pd

sub_lgb = pd.read_csv(r'C:\Users\tsubotanikei0921\Desktop\kaggle_Titanic---Machine-Learning-from-Disaster\sub\submission_lgb_sec.csv')
sub_rf = pd.read_csv(r'C:\Users\tsubotanikei0921\Desktop\kaggle_Titanic---Machine-Learning-from-Disaster\sub\submission_rf_sfm.csv')
sub_lr = pd.read_csv(r'C:\Users\tsubotanikei0921\Desktop\kaggle_Titanic---Machine-Learning-from-Disaster\sub\submission_lr.csv')

sub = pd.read_csv(r'C:\Users\tsubotanikei0921\Desktop\kaggle_Titanic---Machine-Learning-from-Disaster\input\gender_submission.csv')
sub['Survived'] = sub_lgb['Survived'] + sub_rf['Survived'] + sub_lr['Survived']

sub['Survived'] = (sub['Survived'] >= 2).astype(int)
sub.to_csv('sub\submission_ensemble.csv',index=False)
