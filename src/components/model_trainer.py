# import pandas as pd
# import numpy as np
# from src.logger.logging import logging
# from src.exception.exception import customexception
# import os
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from src.utils.utils import save_object, evaluate_model
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import r2_score
# import joblib

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join('artifacts', 'model.pkl')
#     trained_model_type_path = os.path.join('artifacts', 'model_type.pkl')  # Add path to save model type


# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def calculate_adjusted_r2(self, model, X, y):
#         """
#         Calculate the Adjusted R2 score for a given model.
#         """
#         # Fit the model to the data
#         model.fit(X, y)
#         r2 = r2_score(y, model.predict(X))

#         # Number of samples and features
#         n = X.shape[0]
#         p = X.shape[1]

#         # Adjusted R2 formula
#         adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#         return adjusted_r2

#     def initate_model_training(self, train_array, test_array):
#             try:
#                 logging.info('Splitting Dependent and Independent variables from train and test data')
#                 X_train, y_train, X_test, y_test = (
#                     train_array[:, :-1],  # Features from the training data
#                     train_array[:, -1],   # Target from the training data
#                     test_array[:, :-1],   # Features from the test data
#                     test_array[:, -1]     # Target from the test data
#                 )

#                 # Define the models to be tested
#                 models = {
#                     'LinearRegression': LinearRegression(),
#                     'Lasso': Lasso(),
#                     'Ridge': Ridge(),
#                     'ElasticNet': ElasticNet(),
#                     'DecisionTreeRegressor': DecisionTreeRegressor(),
#                     'RandomForestRegressor': RandomForestRegressor(),
#                     'SVR': SVR(),
#                     'KNeighborsRegressor': KNeighborsRegressor(),
#                     'GaussianProcessRegressor': GaussianProcessRegressor(),
#                     'XGBoost': xgb.XGBRegressor(),
#                     'LightGBM': lgb.LGBMRegressor(),
#                     'CatBoost': CatBoostRegressor(verbose=0)
#                 }

#                 # Use the evaluate_model function frim utils to get the model report
#                 model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

#                 # Print the model report
#                 print(model_report)
#                 logging.info(f'Model Report: {model_report}')

#                 # Select the best model based on Adjusted R2 score
#                 best_model_name = max(model_report, key=lambda k: model_report[k]['Adjusted R2 Score'])
#                 best_model_score = model_report[best_model_name]['Adjusted R2 Score']
#                 best_model = models[best_model_name]

#                 print(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')
#                 logging.info(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')

#                 # Save the best model
#                 save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
#                 logging.info("model.pkl file saved in artifacts")
#                 joblib.dump(best_model_name, self.model_trainer_config.trained_model_type_path)  # Save model type

#             except Exception as e:
#                 logging.info('Exception occurred during model training')
#                 raise customexception(e, sys)



# =================== with mlflow implimented =====================

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import save_object, evaluate_model
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    trained_model_type_path = os.path.join('artifacts', 'model_type.pkl')
    mlflow_tracking_uri = None

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def calculate_adjusted_r2(self, model, X, y):
        """
        Calculate the Adjusted R2 score for a given model.
        """
        # Fit the model to the data
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))

        # Number of samples and features
        n = X.shape[0]
        p = X.shape[1]

        # Adjusted R2 formula
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features from the training data
                train_array[:, -1],   # Target from the training data
                test_array[:, :-1],   # Features from the test data
                test_array[:, -1]     # Target from the test data
            )

            # Define the models to be tested
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'SVR': SVR(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'GaussianProcessRegressor': GaussianProcessRegressor(),
                'XGBoost': xgb.XGBRegressor(),
                'LightGBM': lgb.LGBMRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0)
            }

            # Use the evaluate_model function to get the model report
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Print the model report
            print(model_report)
            logging.info(f'Model Report: {model_report}')

            # Select the best model based on Adjusted R2 score
            best_model_name = max(model_report, key=lambda k: model_report[k]['Adjusted R2 Score'])
            best_model_score = model_report[best_model_name]['Adjusted R2 Score']
            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')
            logging.info(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')

            # If MLflow tracking URI is provided, log experiment and model to MLflow
            if self.model_trainer_config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.model_trainer_config.mlflow_tracking_uri)
                mlflow.start_run()
                logging.info("MLflow experiment started")

                # Log parameters, metrics, and model to MLflow
                mlflow.log_param('Model Type', best_model_name)
                mlflow.log_metric('Adjusted R2 Score', best_model_score)

                # Log the trained model
                mlflow.sklearn.log_model(best_model, "model")
                logging.info(f'Model {best_model_name} logged to MLflow')

                # Register the model to MLflow Model Registry (optional)
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", best_model_name)

                mlflow.end_run()
                logging.info("MLflow experiment ended")

            else:
                # If no MLflow URI is provided, simulate the same MLflow logging behavior
                # Set a local URI (in case we still want to track the models locally using MLflow)
                local_mlflow_uri = "file://" + os.path.join(os.getcwd(), 'mlruns')
                mlflow.set_tracking_uri(local_mlflow_uri)
                mlflow.start_run()
                logging.info("Simulated MLflow experiment started (local)")

                # Log parameters, metrics, and model to MLflow
                mlflow.log_param('Model Type', best_model_name)
                mlflow.log_metric('Adjusted R2 Score', best_model_score)

                # Log the trained model
                mlflow.sklearn.log_model(best_model, "model")
                logging.info(f'Model {best_model_name} logged to simulated MLflow')

                # Register the model to MLflow Model Registry (optional)
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", best_model_name)

                mlflow.end_run()
                logging.info("Simulated MLflow experiment ended (local)")

                # Save the model locally (outside MLflow logging)
                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
                logging.info("Model saved locally as model.pkl")

                # Save model type as well (for reference)
                joblib.dump(best_model_name, self.model_trainer_config.trained_model_type_path)
                logging.info(f"Model type saved as {self.model_trainer_config.trained_model_type_path}")

        except Exception as e:
            logging.info('Exception occurred during model training')
            raise customexception(e, sys)