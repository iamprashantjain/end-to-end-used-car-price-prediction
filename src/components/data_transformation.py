import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_preprocessing(self, X: pd.DataFrame):
        try:
            logging.info('Preprocessing initiated')

            # Identify categorical and numerical columns
            cat_cols = X.select_dtypes(include="object").columns.tolist()

            # Add binary columns (those with exactly 2 unique values) to categorical columns
            binary_cols = [col for col in X.columns if set(X[col].dropna().unique()) == {0, 1}]
            cat_cols.extend(binary_cols)

            # Identify numerical columns (remaining columns after removing categorical ones)
            num_cols = [col for col in X.columns if col not in cat_cols]

            logging.info(f"Categorical Columns: {cat_cols}")
            logging.info(f"Numerical Columns: {num_cols}")

            # Find the optimal n_neighbors for KNNImputer using cross-validation
            best_k = self.select_best_k(X[num_cols + cat_cols])

            # Numerical Pipeline using KNNImputer with best k
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", KNNImputer(n_neighbors=best_k)),  # KNN imputation
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline using KNNImputer with best k
            # We encode categorical variables first and then apply KNN imputation
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", KNNImputer(n_neighbors=best_k)),  # KNN imputation
                    ("encoder", OneHotEncoder(sparse_output=False, drop='first'))
                ]
            )

            # Create the preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )
            
            return preprocessor

        except Exception as e:
            logging.error("Exception occurred in get_data_preprocessing")
            raise customexception(e, sys)


    def select_best_k(self, X: pd.DataFrame):
        """
        Select the best k for KNNImputer using cross-validation.
        """
        best_k = None
        best_score = -1

        # Test k values from 1 to 10
        for k in range(1, 11):
            try:
                # Initialize the KNN imputer
                knn_imputer = KNNImputer(n_neighbors=k)

                # Impute missing values for the combined DataFrame
                X_imputed = pd.DataFrame(knn_imputer.fit_transform(X), columns=X.columns)

                # Evaluate model using cross-validation
                pipeline = Pipeline(steps=[
                    ('scaler', StandardScaler()),
                    ('regressor', RandomForestRegressor(random_state=42))
                ])

                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                score = cross_val_score(pipeline, X_imputed, y, cv=cv, scoring='r2').mean()

                logging.info(f'k: {k}, Cross-validated score: {score:.4f}')
                
                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                logging.error(f"Error in selecting best k: {k}")
                raise customexception(e, sys)

        logging.info(f'Best k: {best_k} with score: {best_score:.4f}')
        return best_k

    def initialize_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading train & test data started")

            # Load the datasets
            train_df = pd.read_excel(train_path)
            test_df = pd.read_excel(test_path)

            logging.info(f"Reading train & test data completed")
            logging.info(f"Train data sample: \n{train_df.sample(5)}")
            logging.info(f"Test data sample: \n{test_df.sample(5)}")
            
            logging.info(f"column names:{train_df.columns}")
            
            # Define the target column name
            target_column_name = 'content.onRoadPrice'

            try:
                # Get the preprocessing object
                preprocessing_obj = self.get_data_preprocessing(train_df)
            except Exception as e:
                logging.info('unable to read train_df')
                raise customexception(e,sys)

            # Drop the target column from both train and test sets
            drop_columns = [target_column_name]
            logging.info("Dropping target column and preparing features and target")

            X_train = train_df.drop(columns=drop_columns, axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=drop_columns, axis=1)
            y_test = test_df[target_column_name]

            # Apply preprocessing
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Concatenate transformed features with the target variable to form final datasets
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save the preprocessor object to disk
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing pickle file saved")

            return train_arr, test_arr

        except Exception as e:
            logging.error("Exception occurred in initialize_data_transformation")
            raise customexception(e, sys)


if __name__ == "__main__":
    train_data_path = r"H:\CampusX_DS\week43 - My Projects Aug 2024\end-to-end-used-car-price-prediction\artifacts\train.xlsx"
    test_data_path = r"H:\CampusX_DS\week43 - My Projects Aug 2024\end-to-end-used-car-price-prediction\artifacts\test.xlsx"

    data_transformation = DataTransformation()

    try:
        train_array, test_array = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
        logging.info("Training and testing data transformation completed successfully.")
        logging.info(f"Training array shape: {train_array.shape}")
        logging.info(f"Testing array shape: {test_array.shape}")

    except Exception as e:
        logging.error(f"Error during data transformation: {str(e)}")
        raise customexception(e, sys)
    
    
if __name__ == "__main__":
    train_data_path = r"H:\CampusX_DS\week43 - My Projects Aug 2024\end-to-end-used-car-price-prediction\artifacts\train.xlsx"
    test_data_path = r"H:\CampusX_DS\week43 - My Projects Aug 2024\end-to-end-used-car-price-prediction\artifacts\test.xlsx"

    data_transformation = DataTransformation()

    try:
        train_array, test_array = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
        logging.info("Training and testing data transformation completed successfully.")
        logging.info(f"Training array shape: {train_array.shape}")
        logging.info(f"Testing array shape: {test_array.shape}")

    except Exception as e:
        logging.error(f"Error during data transformation: {str(e)}")
        raise customexception(e, sys)