import pandas as pd
import numpy as np
import sys,os
from pandas.core.base import DataError
from sklearn import preprocessing
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("."))

from sklearn.model_selection import train_test_split
from model.models import LogisticRegression,RandomForest,XGBoost


def run():
    test = pd.read_csv(r'input\test.csv')
    train = pd.read_csv(r'input\train.csv')

    drop_columns = ['Name','Ticket','Cabin','SibSp','Parch','Sex']

    df = pd.concat([train,test]).reset_index(drop=True)

    df['existCabin'] = df['Cabin'].copy()
    df['existCabin'] = df.existCabin.fillna(0)
    df.loc[df.existCabin!=0,'existCabin'] = 1

    df['Age'] = df.Age.fillna(df['Age'].median())
    df['Embarked'] = df.Embarked.fillna(df['Embarked'].mode())
    df['Fare'] = df.Fare.fillna(df['Fare'].median())

    df['Sex_Pclass'] = (
        df.Sex.astype(str)+"_"+df.Pclass.astype(str)
    )
    df['Familysize'] = df['SibSp']+df['Parch']+1


    df = df.drop(drop_columns,axis=1)

    categorical_columns = [
        'Embarked','Sex_Pclass'
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



    lr = LogisticRegression()

    model = lr.model




    model.fit(X_train,y_train)
    
    y_pre = model.predict(X_val)


    print(f'score={lr.accuracy_score(y_val,y_pre)}')


    """ submit """
    y_pre_sub = model.predict(test)
    lr.submit(
        y_predict=y_pre_sub,
        name='lr_7_17'
    )


if __name__ == "__main__":
    run()