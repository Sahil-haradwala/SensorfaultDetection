import sys
import os
import numpy as np
import pandas as pd
from src.constant import *
from src.excep.exception import CustomException
from src.logger import logging
from src.utils.mail_utils import MainUtils
from dataclasses import dataclass

from src.components.data_ingetion import DataIngetion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngetion()
            feature_store_file_path = data_ingestion.intiate_data_ingetion()
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self, feature_store_file_path):
        try:
            data_transformation = DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr, test_arr, preprocessor_path = data_transformation.intiate_data_transformation()
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer = ModelTrainer()
            model_file_path = model_trainer.initiate_model_trainer(train_arr, test_arr)
            return model_file_path
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        try: 
            logging.info("Starting Pipeline...")
            
            feature_store_file_path = self.start_data_ingestion()
            
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(feature_store_file_path)
            
            model_file_path = self.start_model_training(train_arr, test_arr)
            
            print(f"Training Completed successfully. Model saved at: {model_file_path}")

        except Exception as e:
            raise CustomException(e, sys)