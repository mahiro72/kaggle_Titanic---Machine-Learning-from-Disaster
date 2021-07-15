import pandas as pd
import numpy as np
import sys,os
from sklearn import preprocessing

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
        test_size=0.4
        )


    lr = LogisticRegression()
    rf = RandomForest()
    xgb = XGBoost()

    model = lr.model

    model.fit(X_train,y_train)
    y_pre = model.predict(X_val)

    print(f'score={lr.score(y_val,y_pre)}')

    y_pre_sub = model.predict(test)



    """ submit """
    # sub = pd.read_csv(r'input\gender_submission.csv')
    # sub['Survived'] = y_pre_sub.astype(int)
    # sub.to_csv(r'sub\submission_rf.csv',index=False)

if __name__ == "__main__":
    run()