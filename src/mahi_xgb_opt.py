import pandas as pd
import numpy as np
import sys,os
from pandas.core.base import DataError
from sklearn import preprocessing
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("."))

from model.models import XGBoost


def set_data():
    test = pd.read_csv(r'input\test.csv')
    train = pd.read_csv(r'input\train.csv')

    drop_columns = ['Name','Ticket','Cabin','SibSp','Parch']

    df = pd.concat([train,test]).reset_index(drop=True)

    df['existCabin'] = df['Cabin'].copy()
    df['existCabin'] = df.existCabin.fillna('NONE')
    df.loc[df.existCabin!='NONE','existCabin'] = 'EXIST'

    df['Age'] = df.Age.fillna(df['Age'].median())
    df['Embarked'] = df.Embarked.fillna(df['Embarked'].mode())
    df['Fare'] = df.Fare.fillna(df['Fare'].median())


    df['Familysize'] = df['SibSp']+df['Parch']+1


    df = df.drop(drop_columns,axis=1)

    categorical_columns = [
        'Embarked','Sex','existCabin'
    ]

    df = pd.get_dummies(df,columns=categorical_columns)


    train = df[:len(train)]
    test = df[len(train):]
    # print(len(test))

    test = test.drop('Survived',axis=1)

    return train,test

train,test = set_data()



def objective(trial):

    X_train, X_val, y_train, y_val = train_test_split(
        train.drop('Survived',axis = 1),
        train['Survived'],
        test_size=0.3,
        random_state=123
    )

    params = {
        'objective': 'binary:logistic',
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
        'min_childe_weigh':trial.suggest_loguniform('min_childe_weigh', 1e-8, 1.0),
        'alpha':trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'gamma':trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'colsample_bytree':trial.suggest_loguniform('colsample_bytree', 1e-8, 1.0),
        'subsample':trial.suggest_loguniform('subsample', 1e-8, 1.0),
    }


    xgb = XGBoost(X_train,y_train,X_val,y_val,**params)

    y_pre = xgb.predict(X_val)
    y_pre = (y_pre>0.5).astype(int)

    return (1-xgb.accuracy_score(y_val,y_pre))
    


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective,n_trials=5000)

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
