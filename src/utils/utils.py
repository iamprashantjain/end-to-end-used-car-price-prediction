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
    
    
    
    
import numpy as np
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def calculate_95_ci(model, X, y, model_type):
    """
    Calculate the 95% Confidence Interval (CI) for the predictions of the given model.
    The method varies depending on the model type.

    :param model: The trained model (e.g., RandomForest, LinearRegression)
    :param X: The feature data for which to make predictions
    :param y: The actual values (target variable) for the test set (optional for some models)
    :param model_type: Type of model used (e.g., 'RandomForest', 'LinearRegression')
    :return: Lower and upper bounds of the 95% CI
    """

    # Generate predictions
    predictions = model.predict(X)

    # For Linear Models (Linear Regression, Lasso, Ridge, etc.)
    if model_type in ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet']:
        # Calculate the residuals (errors)
        residuals = y - predictions
        # Calculate the standard error of the residuals
        se = np.std(residuals)
        # Calculate the t-statistic for 95% confidence (2-tailed, assuming large sample size)
        t_stat = stats.t.ppf(0.975, df=len(X)-1)  # 95% confidence for a two-tailed distribution

        # Calculate the margin of error
        margin_of_error = t_stat * se

        # Calculate the CI bounds
        ci_lower = predictions - margin_of_error
        ci_upper = predictions + margin_of_error

    # For Tree-based Models (Random Forest, Decision Trees)
    elif model_type in ['RandomForestRegressor', 'DecisionTreeRegressor']:
        # For tree-based models, we can use a bootstrapping approach for CI estimation
        n_bootstrap = 1000  # Number of bootstrap samples
        bootstrap_preds = []
        
        for _ in range(n_bootstrap):
            # Resample X and y with replacement
            indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_resampled = X[indices]
            y_resampled = y[indices]
            
            # Make predictions with the resampled data
            model.fit(X_resampled, y_resampled)
            bootstrap_preds.append(model.predict(X))
        
        # Convert list of predictions to a numpy array for easier processing
        bootstrap_preds = np.array(bootstrap_preds)
        
        # Calculate the lower and upper bounds of the 95% CI
        ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)  # 2.5th percentile (lower bound)
        ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)  # 97.5th percentile (upper bound)

    # For Support Vector Regression (SVR)
    elif model_type == 'SVR':
        # Use a bootstrapping approach for SVR
        n_bootstrap = 1000
        bootstrap_preds = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_resampled = X[indices]
            y_resampled = y[indices]

            # Refit SVR model and predict
            model.fit(X_resampled, y_resampled)
            bootstrap_preds.append(model.predict(X))

        bootstrap_preds = np.array(bootstrap_preds)
        ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)

    # For K-Nearest Neighbors Regressor (KNeighborsRegressor)
    elif model_type == 'KNeighborsRegressor':
        # Use bootstrapping for KNN as well
        n_bootstrap = 1000
        bootstrap_preds = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_resampled = X[indices]
            y_resampled = y[indices]

            model.fit(X_resampled, y_resampled)
            bootstrap_preds.append(model.predict(X))

        bootstrap_preds = np.array(bootstrap_preds)
        ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)

    # For Gaussian Process Regressor (GaussianProcessRegressor)
    elif isinstance(model, GaussianProcessRegressor):
        # For Gaussian processes, the CI can be derived from the predicted mean and standard deviation
        mean, std = model.predict(X, return_std=True)
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

    # For Boosting Models (XGBoost, LightGBM, CatBoost)
    elif model_type in ['XGBoost', 'LightGBM', 'CatBoost']:
        n_bootstrap = 1000
        bootstrap_preds = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_resampled = X[indices]
            y_resampled = y[indices]

            # Refit the boosting model and predict
            model.fit(X_resampled, y_resampled)
            bootstrap_preds.append(model.predict(X))

        bootstrap_preds = np.array(bootstrap_preds)
        ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)

    else:
        ci_lower = predictions - 5000
        ci_upper = predictions + 5000

    return ci_lower, ci_upper