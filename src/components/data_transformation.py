import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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
            logging.info('removing appointmentId from dataframe')
            X.drop(columns=['content.appointmentId'], inplace=True)
                        
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

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
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


# data = data[["ABSAntilockBrakingSystem","AirConditioner","Airbags","Bootspacelitres","Displacementcc","FueltankCapacitylitres","GroundClearancemm","Tyre_WARN","MaxPowerbhp","MaxPowerrpm","MaxTorqueNm","SeatingCapacity","content.bodyType","content.duplicateKey","content.fitnessUpto_months_remaining","content.fuelType","content.insuranceExpiry_months_remaining","content.insuranceType","content.make","content.model","content.odometerReading","content.ownerNumber","content.transmission","content.year","defects","repainted",]]
#currently facing error: columns not available in the index.. bcoz of feature engineering steps not included in cleaning
