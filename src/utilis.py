import numpy as np
import pandas as pd
import dill
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
  

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict, scoring_metric="accuracy"):
    try:
        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            gs = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring=scoring_metric, verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = {
                "Train Accuracy": train_score,
                "Test Accuracy": test_score
            }
            
            
        return report
    

    except Exception as e:
        raise CustomException(e,sys)


            
            