import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.constant import *
from src.excep.exception import CustomException
from src.logger import logging 
from src.utils.mail_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join(artifact_folder)
    transform_train_file_path = os.path.join(artifact_dir, "train.npy")
    transform_test_file_path = os.path.join(artifact_dir, "test.npy")
    transform_object_file_path = os.path.join(artifact_dir, "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig() # ✅ FIXED: Added ()
        self.utils = MainUtils()
        
    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_objects(self):
        try:
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            # Note: To use the scaler, you can expand steps: [imputer_step, ('scaler', RobustScaler())]
            preprocessor = Pipeline(steps=[imputer_step])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def intiate_data_transformation(self):
        logging.info("Entered intiate data transformation method of a data data transformation class")
        try:
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)
            
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            preprocessor = self.get_data_transformer_objects()
            
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test) # ✅ FIXED: Changed to transform()
            
            preprocessor_path = self.data_transformation_config.transform_object_file_path
            
            # ✅ FIXED: Corrected parenthesis layout
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)
            
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]
            
            # Save transformed data as local files
            np.save(self.data_transformation_config.transform_train_file_path, train_arr)
            np.save(self.data_transformation_config.transform_test_file_path, test_arr)
            
            # ✅ FIXED: Now returns 3 parameters to match TrainingPipeline unpack rules
            return (train_arr, test_arr, preprocessor_path)
            
        except Exception as e:
            raise CustomException(e, sys)
