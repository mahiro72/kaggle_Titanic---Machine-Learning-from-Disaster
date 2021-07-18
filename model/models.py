from pandas.core.algorithms import mode
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import ensemble
import xgboost as xgb
import pandas as pd

from sklearn import feature_selection


class Model:
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
            n_jobs=-1,
            random_state=123,
            criterion='gini',
            max_depth=4,
            n_estimators=30
            )
    

    def sfm(self,x_train,y_train,features,threshold):
        if self.model == None:
            raise Exception('Please specify the model first!!')

        sfm = feature_selection.SelectFromModel(
            estimator=self.model,
            threshold=threshold
        )
        sfm.fit(x_train,y_train)
        support = sfm.get_support()

        return [x for x,y in zip(features,support) if y ==True]



    
class XGBoost(Model):
    def __init__(self,X_train,y_train,X_val,y_val,**params):

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
          
        params = params
        num_round = 500
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]#訓練データはdtrain、評価用のテストデータはdvalidと設定
        
        self.model = xgb.train(params,
                        dtrain,#訓練データ
                        num_round,#設定した学習回数
                        early_stopping_rounds=20,
                        evals=watchlist,
                        )
    
    def predict(self,data):
        d_data = xgb.DMatrix(data)
        y_pre = self.model.predict(d_data)
        return y_pre