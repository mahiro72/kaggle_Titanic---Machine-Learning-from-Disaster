from pandas.core.algorithms import mode
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import ensemble
import xgboost as xgb


class Model():
    def __init__(self):
        self.model = None

    def name(self):
        return str(self.model)


class LogisticRegression(Model):
    def __init__(self):
        self.model = linear_model.LogisticRegression(
            max_iter=500
        )

    def score(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)
        

class RandomForest(Model):
    def __init__(self):
        self.model = ensemble.RandomForestClassifier(
            n_jobs=-1
            )
    
    def score(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)



    
class XGBoost(Model):
    def __init__(self):
        self.model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimators=200,
        eval_metric='mlogloss'
    )
    
    def score(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)



