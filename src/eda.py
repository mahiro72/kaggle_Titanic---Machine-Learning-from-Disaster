#%%
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

#%%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#%%
df = pd.concat([train.drop('Survived',axis=1),test]).reset_index(drop=0)

#%%
df
#%%
plt.bar([0,1],train.Survived.value_counts())