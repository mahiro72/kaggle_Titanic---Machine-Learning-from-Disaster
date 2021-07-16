import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

test = pd.read_csv(r'input\test.csv')
train = pd.read_csv(r'input\train.csv')

drop_col = ['Name','Ticket','Cabin','Embarked']
train = train.drop(drop_col,axis=1)
test = test.drop(drop_col,axis=1)

train = train.fillna(0)
test = test.fillna(0)



x_train = train.drop('Survived',axis = 1)
y_train = train['Survived']

# print(train.columns)

# print(train.Pclass.unique())
# from sklearn import preprocessing

# lbl = preprocessing.LabelEncoder()

sex_map = {
    "male":0,
    'female':1
}
x_train['Sex'] = x_train.Sex.map(sex_map)
test['Sex'] = test.Sex.map(sex_map)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pre = model.predict(test)
print(y_pre)


sub = pd.read_csv(r'input\gender_submission.csv')
sub['Survived'] = y_pre
sub.to_csv(r'sub\submission2.csv',index=False)
