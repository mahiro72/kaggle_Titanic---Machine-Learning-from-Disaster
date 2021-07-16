import pandas as pd
import numpy as np
import sys,os
from sklearn import preprocessing
import matplotlib.pyplot as plt


from tqdm import tqdm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score




sys.path.append(os.path.abspath("."))

from sklearn.model_selection import train_test_split
from model.models import LogisticRegression,RandomForest,XGBoost


def run():
    test = pd.read_csv(r'input\test.csv')
    train = pd.read_csv(r'input\train.csv')

    drop_columns = ['Name','Ticket','Cabin','SibSp','Parch']

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
    
    #条件設定
    max_score = 0
    SearchMethod = 0
    RFC_grid = {RandomForestClassifier(): {"n_estimators": [i for i in range(1, 50)],
                                        "criterion": ["gini", "entropy"],
                                        "max_depth":[i for i in range(1, 20)]
                                        }}

    #ランダムフォレストの実行
    for model, param in tqdm(RFC_grid.items()):
        print(model)
        clf = GridSearchCV(model, param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        score = f1_score(y_val, y_pred, average="micro")

        if max_score < score:
            max_score = score
            best_param = clf.best_params_
            best_model = model.__class__.__name__

    print("ベストスコア:{}".format(max_score))
    print("モデル:{}".format(best_model))
    print("パラメーター:{}".format(best_param))







    """ submit """
    # y_pre_sub = model.predict(test)
    # rf.submit(
    #     y_predict=y_pre_sub,
    #     name='rf_third'
    # )


if __name__ == "__main__":
    run()



