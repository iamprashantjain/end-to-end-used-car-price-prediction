import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customexception
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the models using Cross-Val score and Adjusted R2 Score.
    Returns a dictionary with model names as keys and performance metrics as values.
    """
    try:
        report = {}  # Dictionary to store model evaluation results

        # Iterate through the provided models
        for model_name, model in models.items():
            
            # Cross-validation score (R²) using 5-fold cross-validation
            cross_val_score_result = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cross_val_score_mean = np.mean(cross_val_score_result)

            # Fit the model to the training data
            model.fit(X_train, y_train)

            # Predict testing data using the trained model
            y_test_pred = model.predict(X_test)

            # Calculate R2 score on the test data
            test_model_score = r2_score(y_test, y_test_pred)

            # Calculate Adjusted R2 Score
            n = X_train.shape[0]  # Number of samples
            p = X_train.shape[1]  # Number of features
            adjusted_r2 = 1 - (1 - test_model_score) * (n - 1) / (n - p - 1)

            # Store the evaluation metrics in the report dictionary
            report[model_name] = {
                'Cross-Val Score': cross_val_score_mean,
                'Adjusted R2 Score': adjusted_r2,
                'Test R2 Score': test_model_score  # Store the R2 score on the test data
            }

            # Log model performance
            logging.info(f"Model {model_name} - Cross-Val Score: {cross_val_score_mean:.4f}, "
                         f"Test R2 Score: {test_model_score:.4f}, Adjusted R2 Score: {adjusted_r2:.4f}")

        return report

    except Exception as e:
        logging.error('Exception occurred during model evaluation')
        raise customexception(e, sys)

    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)