import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from pymongo.mongo_client import MongoClient
from zipfile import Path
from dataclasses import dataclass

from src.constant import *
from src.excep.exception import CustomException 
from src.logger import logging
from src.utils.mail_utils import MainUtils


@dataclass
class DataIngetionconfig:
    artifact_folder: str = os.path.join(os.getcwd(), "artifacts") 
    
    
class DataIngetion:
    def __init__(self):
        self.data_ingetion_config = DataIngetionconfig()
        self.utils = MainUtils()
        
    def export_collection_as_dataframe(self, collection_name, db_name):
        try:
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[db_name][collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na": np.nan}, inplace=True)
            
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_data_into_feature_store_file_path(self) -> pd.DataFrame:
        try:
            logging.info("Exporting data from mongodb")
            raw_file_path = self.data_ingetion_config.artifact_folder
            
            os.makedirs(raw_file_path, exist_ok=True)
            sensor_data = self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME, 
                db_name=MONGO_DATABASE_NAME
            )
            
            logging.info(f"saving exported data into feature store file path: {raw_file_path}")
            
            feature_store_file_path = os.path.join(raw_file_path, "wafer_fault.csv")
            sensor_data.to_csv(feature_store_file_path, index=False)
            
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)
        
    def intiate_data_ingetion(self) -> Path:
        logging.info("entered Intiated data ingetion method of data integration process class")
        try:
            feature_store_file_path = self.export_data_into_feature_store_file_path()
            logging.info("got the data mongodb ")
            logging.info("exited and intiate data ingetion method of data ingetion class")
            
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)

