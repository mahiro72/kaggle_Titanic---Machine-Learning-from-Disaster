import pandas as pd
import numpy as np
import sys,os
from sklearn import preprocessing
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("."))

from sklearn.model_selection import train_test_split
from model.models import LogisticRegression,RandomForest,XGBoost


def run():
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



    rf = RandomForest()

    model = rf.model

    sfm_columns = rf.sfm(X_train,y_train,test.columns)

    X_train = X_train[sfm_columns]
    X_val = X_val[sfm_columns]

    test = test[sfm_columns]


    model.fit(X_train,y_train)
    
    y_pre = model.predict(X_val)

    
    
    print(f'score={rf.accuracy_score(y_val,y_pre)}')


    """ submit """
    y_pre_sub = model.predict(test)
    rf.submit(
        y_predict=y_pre_sub,
        name='rf_sfm_sec'
    )



if __name__ == "__main__":
    run()