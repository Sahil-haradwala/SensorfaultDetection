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
    def start_data_ingetion(self):
        try:
            data_ingetion=DataIngetion()
            feature_store_file_path=data_ingetion.intiate_data_ingetion()
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_tansformation(self,feature_store_file_path):
        try:
            data_transformation=DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr,test_arr,preprocessor_path=data_transformation.intiate_data_transformation()
            return train_arr,test_arr,preprocessor_path
        except Exception as e :
            raise CustomException(e,sys)
        
    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer=ModelTrainer()
            model_score=model_trainer.intiate_model_trainer(train_arr,test_arr)
            
            return model_score
        except Exception as e:
            raise CustomException(e,sys)
        
    def run_pipeline(self):
        try: 
            feature_store_file_path=self.start_data_ingetion()
            train_arr,test_arr,preprocessor_path=self.start_data_tansformation(feature_store_file_path)
            
            r2_square=self.start_model_training(train_arr,test_arr)
            
            print("Training Completed Model score is ",r2_square)

        except Exception as e :
            raise CustomException(e,sys) 
                   
            