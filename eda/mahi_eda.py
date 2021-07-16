#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

#%%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#%%
df = pd.concat([train,test]).reset_index(drop=True)
drop_columns = ['Name','Ticket','Cabin']

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


df = df.drop(drop_columns,axis=1)

categorical_columns = [
    'Sex','Embarked','Sex_Pclass'
]

for col in categorical_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(df[col])
    df.loc[:,col] = lbl.transform(df[col])
#%%
df['Familysize'] = df['SibSp']+df['Parch']+1
#%%
df['Familysize']
#%%
sns.countplot(x='Familysize',data=df,hue='Survived')

#%%
sns.countplot(x='Sex_Pclass',data=df,hue='Survived')
#%%

#%%
sns.countplot(x='existCabin',data=df,hue='Survived')


#%%
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#%%
sns.distplot(df.Fare)