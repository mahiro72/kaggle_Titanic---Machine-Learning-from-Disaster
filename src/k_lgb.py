import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score

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

params = {
    'objective': 'binary'
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



# sub = pd.read_csv(r'input\gender_submission.csv')
# sub['Survived'] = y_pre
# sub.to_csv(r'sub\submission2.csv',index=False)
