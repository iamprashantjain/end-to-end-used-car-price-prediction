import os
import sys
from src.logger.logging import logging
from src.exception.exception import customexception
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# print(f"Current working directory: {os.getcwd()}")
# logging.info(f"Current working directory: {os.getcwd()}")

# #creating an object of dataingestion
# obj=DataIngestion()

# #data ingestion returns: train_data_path,test_data_path
# train_data_path,test_data_path=obj.initiate_data_ingestion()
# print(f"Inside Training Pipeline: {train_data_path}, {test_data_path}")
# logging.info(f"Inside Training Pipeline: {train_data_path}, {test_data_path}")

# #creating data transformatoion object
# data_transformation=DataTransformation()

# #data transformatoion returns: train_arr,test_arr & takes input : train_data_path,test_data_path
# train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)
# print(f"Inside Training Pipeline: {train_arr}, {test_arr}")
# logging.info(f"Inside Training Pipeline: {train_arr}, {test_arr}")

# #creating object of ModelTrainer
# model_trainer_obj=ModelTrainer()

# #ModelTrainer takes input train_arr,test_arr
# model_trainer_obj.initate_model_training(train_arr,test_arr)


class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            return train_data_path, test_data_path
        except Exception as e:
            raise customexception(e,sys)
        
    
    def start_data_transformation(self,train_data_path, test_data_path):
        try:
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
            return train_arr, test_arr
        except Exception as e:
            raise customexception(e,sys)
        
    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer_obj = ModelTrainer()
            model_trainer_obj.initate_model_training(train_arr, test_arr)
        except Exception as e:
            raise customexception(e,sys)
        
    #ensemble every stage
    def start_training(self):
        try:
            train_data_path, test_data_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            self.start_model_training(train_arr, test_arr)
        except Exception as e:
            raise customexception(e,sys)