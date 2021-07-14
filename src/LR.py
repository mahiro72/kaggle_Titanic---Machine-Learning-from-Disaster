import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

test = pd.read_csv(r'input\test.csv')
train = pd.read_csv(r'input\train.csv')

dropva = ['Name','Ticket','Cabin']
train = train.drop(dropva,axis=1)
x_train = train.drop('Survived',axis = 1)
y_train = train['Survived']

# print(x_train.head())

clf = LogisticRegression()
clf.fit(x_train,y_train)