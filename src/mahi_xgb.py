import pandas as pd
import numpy as np
import sys,os
from pandas.core.base import DataError
from sklearn import preprocessing
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna



sys.path.append(os.path.abspath("."))

from sklearn.model_selection import train_test_split
from model.models import LogisticRegression,RandomForest,XGBoost


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

    # df['Sex_Pclass'] = (
    #     df.Sex.astype(str)+"_"+df.Pclass.astype(str)
    # )

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



def run():

    X_train, X_val, y_train, y_val = train_test_split(
        train.drop('Survived',axis = 1),
        train['Survived'],
        test_size=0.3,
        random_state=123
    )

    params = {
        'max_depth': 13, 
        'n_estimators': 428, 
        'learning_rate': 0.310003873787448, 
        'min_childe_weigh': 0.00011255244212051311, 
        'alpha': 0.00983097229650888, 
        'gamma': 0.015693277326804787, 
        'colsample_bytree': 1.0290651991464946e-07, 
        'subsample': 0.6532162104904355
        }


    xgb = XGBoost(X_train,y_train,X_val,y_val,**params)

    y_pre = xgb.predict(X_val)
    y_pre = (y_pre>0.5).astype(int)

    print(f'score={xgb.accuracy_score(y_val,y_pre)}')

    y_pre_sub= xgb.predict(test)
    y_pre_sub = (y_pre_sub>0.5).astype(int)

    submit(y_pre_sub)

    

def submit(sub):
    """ submit """
    submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': sub})
    submission.to_csv(r'sub\submission_xgb_opt4.csv',index=False)


if __name__ == "__main__":
    run()
