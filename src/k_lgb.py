from numpy.lib.twodim_base import tri
import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
import optuna


test = pd.read_csv(r'input\test.csv')
train = pd.read_csv(r'input\train.csv')

drop_columns = ['Name','Ticket','Cabin']

df = pd.concat([train,test]).reset_index(drop=True)
df = df.drop(drop_columns,axis=1)

df['Age'] = df.Age.fillna(df['Age'].median())
df['Embarked'] = df.Embarked.fillna(df['Embarked'].mode())
df['Fare'] = df.Fare.fillna(df['Fare'].median())


df['Sex_Pclass'] = (
    df.Sex.astype(str)+"_"+df.Pclass.astype(str)
)


categorical_columns = [
    'Sex','Embarked','Sex_Pclass'
]

for col in categorical_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(df[col])
    df.loc[:,col] = lbl.transform(df[col])

train = df[:len(train)]
test = df[len(train):]

test = test.drop('Survived',axis=1)

X_train, X_val, y_train, y_val = train_test_split(
    train.drop('Survived',axis = 1),
    train['Survived'],
    test_size=0.3,
    random_state=123
)


lgb_train = lgb.Dataset(X_train,y_train,categorical_feature = categorical_columns)
lgb_eval = lgb.Dataset(X_val,y_val,reference=lgb_train, categorical_feature = categorical_columns)

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    score = (x-2)**2
    print('x: %1.3f, score: %1.3f' % (x, score))
    return score

study = optuna.create_study()
study.optimize(objective, n_trials=100)


params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'lambda_l1': optuna.trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    # 'lambda_l2': optuna.trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    # 'num_leaves': optuna.trial.suggest_int('num_leaves', 2, 256),
    # 'feature_fraction': optuna.trial.suggest_uniform('feature_fraction', 0.4, 1.0),
    # 'bagging_fraction': optuna.trial.suggest_uniform('bagging_fraction',0.4, 1.0),
    # 'bagging_freq': optuna.trial.suggest_int('bagging_freq', 1, 7),
    # 'min_child_samples': optuna.trial.suggest_int('min_child_samples',5 ,100),
}

model = lgb.train(params,lgb_train,valid_sets=lgb_eval,verbose_eval=10,num_boost_round=1000,early_stopping_rounds=10)

y_pre = model.predict(X_val)
y_pre = (y_pre >= 0.5).astype(int)
print(f'score={accuracy_score(y_val,y_pre)}')

y_pre_sub = model.predict(test)
y_pre_sub = (y_pre_sub > 0.5).astype(int)
sub = pd.read_csv(r'input\gender_submission.csv')
sub['Survived'] = y_pre_sub.astype(int)
sub.to_csv(r'sub\submission_'+str('lgb_sec')+'.csv',index=False)




