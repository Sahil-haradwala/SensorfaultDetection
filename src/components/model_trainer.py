import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Generator, List, Tuple

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Assuming these are correctly defined in your project structure
from src.constant import *
from src.excep.exception import CustomException
from src.logger import logging
from src.utils.mail_utils import MainUtils


@dataclass
class ModelTrainingConfig:
    # Fixed: Provide a string directory name instead of an undefined variable
    artifact_folder = "artifacts" 
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        self.utils = MainUtils()
        
        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier() # Fixed typo
        }
    
    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        try:
            report = {}
            
            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                
                model.fit(X_train, y_train)
                
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                train_model_accuracy = accuracy_score(y_train, y_train_pred)
                test_model_accuracy = accuracy_score(y_test, y_test_pred)
                
                # Fixed: Use model_name instead of calling .keys() on the model object
                report[model_name] = test_model_accuracy
                
            return report
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_best_model(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
        try:
            model_report: dict = self.evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=self.models
            )
            
            print(model_report)
            
            best_model_score = max(sorted(model_report.values()))
            
            # Fixed: Corrected list/index bracket placement
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Fixed: Use model name as key instead of model score
            best_model_object = self.models[best_model_name]
            
            return best_model_name, best_model_score, best_model_object
        
        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object: object, best_model_name, X_train, y_train) -> object:
        try:
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_model_paramgrid"]
            
            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1 
            )
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            print("Best params are:", best_params)
            
            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            logging.info("Extracting config file path")
            
            # Fixed: Replaced redundant inline code with a call to the get_best_model method
            best_model_name, best_model_score, best_model = self.get_best_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
            
            # Fine-tune the best model
            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=X_train,
                y_train=y_train
            )
            
            # Re-train and predict with fine-tuned model
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            best_model_score = accuracy_score(y_test, y_pred)
            
            print(f"Best model name: {best_model_name} with score: {best_model_score}")
            
            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold of 0.5")
                
            logging.info("Best found model on both training and testing dataset")
            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")
            
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            return self.model_trainer_config.trained_model_path
            
        except Exception as e:
            raise CustomException(e, sys)