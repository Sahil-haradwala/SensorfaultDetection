import shutil
import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.excep.exception import CustomException
from flask import request
from src.constant import *
from src.utils.mail_utils import MainUtils
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname : str='predictions'
    prediction_file_name:str='prediction_file.csv'
    model_file_path: str=os.path.join(artifact_folder,'model.pkl')
    preprocessor_path: str=os.path.join(artifact_folder,'preprocessor.pkl')
    prediction_file_path:str=os.path.join(prediction_output_dirname,prediction_file_name)

class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def save_input_files(self) -> str:
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)
            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            input_csv_file.save(pred_file_path)
            return pred_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def Predict(self, features): 
        try:
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(file_path=self.prediction_pipeline_config.preprocessor_path)
            transform_x = preprocessor.transform(features)
            preds = model.predict(transform_x)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

    def get_predited_dataframe(self, input_dataframe_path: str): 
        try:
            prediction_column_name: str = TARGET_COLUMN
            
            # 1. Read the CSV using semicolon separator to prevent merged single columns
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path, sep=';')
            
            input_dataframe = input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe
            
            # 2. Pivot data from long format (Timestamp;SensorId;Value) to wide format
            if 'SensorId' in input_dataframe.columns and 'Value' in input_dataframe.columns:
                logging.info("Reshaping input data from long to wide format.")
                features_df = input_dataframe.pivot(index='Timestamp', columns='SensorId', values='Value')
                features_df = features_df.reset_index()
                
                # Drop Timestamp if your training steps did not include it
                if 'Timestamp' in features_df.columns:
                    features_df = features_df.drop(columns=['Timestamp'])
            else:
                features_df = input_dataframe

            # 3. Load preprocessor to verify feature alignment
            preprocessor = self.utils.load_object(file_path=self.prediction_pipeline_config.preprocessor_path)
            
            # 4. Extract expected feature names from scikit-learn pipeline or imputer metadata
            if hasattr(preprocessor, 'feature_names_in_'):
                expected_features = preprocessor.feature_names_in_
            elif hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    expected_features = preprocessor.get_feature_names_out()
                except:
                    # Fallback check on individual steps if wrapped inside a basic sequential Pipeline object
                    expected_features = preprocessor.steps[0][1].feature_names_in_
            else:
                raise Exception("Could not automatically retrieve feature names from preprocessor layout. Please ensure feature_names_in_ is exposed.")

            # 5. Expand and align columns from your input to match all 590 expected training dimensions.
            # Missing columns are automatically initialized with NaN values for SimpleImputer to resolve.
            features_df = features_df.reindex(columns=expected_features)

            # 6. Run prediction with the structurally corrected dataset shape
            predictions = self.Predict(features_df)
            
            # 7. Format output mapping and append prediction results back
            features_df[prediction_column_name] = [pred for pred in predictions]
            
            target_column_mapping = {0: 'bad', 1: 'good'}
            features_df[prediction_column_name] = features_df[prediction_column_name].map(target_column_mapping)
            
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)
            features_df.to_csv(self.prediction_pipeline_config.prediction_file_path, index=False)
            logging.info("prediction completed")
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files() 
            self.get_predited_dataframe(input_csv_path)
            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(e, sys)
