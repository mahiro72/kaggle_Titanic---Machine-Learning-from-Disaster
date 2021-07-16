from pandas.core.algorithms import mode
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import ensemble
import xgboost as xgb
import pandas as pd

from sklearn import feature_selection


class Model():
    def __init__(self):
        self.model = None

    def submit(self,y_predict,name):
        sub = pd.read_csv(r'input\gender_submission.csv')
        sub['Survived'] = y_predict.astype(int)
        sub.to_csv(r'sub\submission_'+str(name)+'.csv',index=False)

    def name(self):
        return str(self.model)
    
    def accuracy_score(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)

    


class LogisticRegression(Model):
    def __init__(self):
        self.model = linear_model.LogisticRegression(
            max_iter=500
        )


        

class RandomForest(Model):
    def __init__(self):
        self.model = ensemble.RandomForestClassifier(
            n_jobs=-1
            )
    

    def sfm(self,x_train,y_train,features):
        if self.model == None:
            raise Exception('Please specify the model first!!')

        sfm = feature_selection.SelectFromModel(
            estimator=self.model
        )
        sfm.fit(x_train,y_train)
        support = sfm.get_support()

        return [x for x,y in zip(features,support) if y ==True]



    
class XGBoost(Model):
    def __init__(self):
        self.model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimators=200,
        eval_metric='mlogloss'
    )




