from datetime import datetime
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.xlsx")
    train_data_path: str = os.path.join("artifacts", "train.xlsx")
    test_data_path: str = os.path.join("artifacts", "test.xlsx")


class DataIngestion:
    def __init__(self, raw_data_path: str = "cars24_final_data.xlsx"):
        # Initialize data ingestion output paths
        self.ingestion_config = DataIngestionConfig()
        self.raw_data_path = raw_data_path  # Allow flexible data path for input

    def safe_to_datetime(self, date_value):
        """
        Safely converts a value to datetime, handling both Unix timestamps and formatted dates.
        """
        try:
            # If it's a Unix timestamp (integer or float), convert it
            if isinstance(date_value, (int, float)) and date_value > 1000000000:  # Unix timestamps are large integers
                return pd.to_datetime(date_value, unit='s')
            else:
                # Attempt to convert using the specified format
                return pd.to_datetime(date_value, format='%d-%b-%Y', errors='coerce')  # 'coerce' will turn invalid parsing into NaT
        except Exception as e:
            # If all fails, return NaT (Not a Time)
            return pd.NaT

    def initial_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement the initial data cleaning steps.
        This can include handling missing values, removing duplicates,
        and other necessary preprocessing steps.
        """
        logging.info("Data cleaning started")

        # Perform data cleaning on the input df

        # Step 1: Initial Cleaning

        # Drop duplicates and unwanted rows
        df.drop_duplicates(inplace=True)
        df = df[df['specs_tag'] == "available"]
        df = df[df[['Displacementcc', 'GearBoxNumberOfGears']].notna().all(axis=1)]
        df.drop(columns=['TransmissionType', 'specs_tag'], inplace=True)

        # Remove unnecessary columns
        df.drop(columns=['content.city'], inplace=True)

        # Step 2: Handle NaN and categorical replacements

        # Replace NaN with "not available" for selected columns
        cols_to_replace = [
            'ABSAntilockBrakingSystem', 'PowerWindowsFront', 'PowerWindowsRear', 
            'AirConditioner', '12VPowerOutlet', 'MultifunctionDisplayScreen', 
            'EntertainmentDisplayScreen', 'VoiceRecognition', 'SmartCardSmartKey', 
            'AmbientLighting', 'SunroofMoonroof', 'WirelessChargingPad', 
            'VentilatedSeatsRear', 'HVACControl', '360DegreeCamera', 
            'ParkingAssistSide', 'DriverSeatAdjustmentElectric'
        ]
        df[cols_to_replace] = df[cols_to_replace].fillna("not available")

        # Step 3: Handle column values (like "not available" to 0)
        def update_columns(df):
            for col in df.columns:
                unique_values = set(df[col].astype(str).unique())
                if unique_values == {'1', 'not available'}:
                    df[col] = df[col].replace('not available', 0)
                elif unique_values == {'Available', 'Not available'}:
                    df[col] = df[col].replace({'Available': 1, 'Not available': 0})
                elif unique_values == {'0', 'not available'}:
                    df[col] = df[col].replace('not available', 0)
    
        # Apply the function to your DataFrame
        update_columns(df)

        # Replace "OK" with 1 and "WARN" with 0 in specified columns
        columns_to_replace = ['Left Front Tyre', 'Right Front Tyre', 'Left Rear Tyre', 'Right Rear Tyre', 'Spare Tyre']
        df[columns_to_replace] = df[columns_to_replace].replace({'OK': 1, 'WARN': 0})

        # Step 4: Feature Engineering

        # Create 'Tyre_Health' and 'tyre_health_pct'
        df['Tyre_Health'] = df[['Left Front Tyre', 'Right Front Tyre', 'Left Rear Tyre', 'Right Rear Tyre', 'Spare Tyre']].sum(axis=1)
        total_tyres = df[['Left Front Tyre', 'Right Front Tyre', 'Left Rear Tyre', 'Right Rear Tyre', 'Spare Tyre']].shape[1]
        df['tyre_health_pct'] = (df['Tyre_Health'] / total_tyres) * 100

        # Create the 'content.duplicateKey' column
        df['content.duplicateKey'] = df['content.duplicateKey'].replace({True: 1, False: 0})

        # Calculate months remaining for 'content.fitnessUpto', 'content.insuranceExpiry', and 'content.lastServicedAt'
        today = pd.to_datetime(datetime.now().date())

        # Fitness Upto - months remaining
        df['content.fitnessUpto'] = df['content.fitnessUpto'].apply(self.safe_to_datetime)
        df['content.fitnessUpto_months_remaining'] = ((df['content.fitnessUpto'].dt.year - today.year) * 12 + 
                                                      (df['content.fitnessUpto'].dt.month - today.month))

        # Insurance Expiry - months remaining
        df['content.insuranceExpiry'] = df['content.insuranceExpiry'].apply(self.safe_to_datetime)
        df['content.insuranceExpiry_months_remaining'] = ((df['content.insuranceExpiry'].dt.year - today.year) * 12 + 
                                                          (df['content.insuranceExpiry'].dt.month - today.month))
        df['content.insuranceExpiry_months_remaining'] = df['content.insuranceExpiry_months_remaining'].clip(lower=0)

        # Last Serviced At - months remaining
        df['content.lastServicedAt'] = df['content.lastServicedAt'].apply(self.safe_to_datetime)
        df['content.lastServicedAt_months_remaining'] = ((df['content.lastServicedAt'].dt.year - today.year) * 12 + 
                                                        (df['content.lastServicedAt'].dt.month - today.month))
        df['content.lastServicedAt_months_remaining'] = df['content.lastServicedAt_months_remaining'].abs()

        # Step 5: Drop columns that are no longer needed after feature engineering
        df.drop(columns=['content.fitnessUpto', 'content.insuranceExpiry', 'content.lastServicedAt'], inplace=True)
        df.drop(columns=['content.registrationNumber', 'content.listingPrice'], inplace=True)

        # Step 6: Extract state from 'content.cityRto' (move this before dropping the column)
        if 'content.cityRto' in df.columns:
            df['car_state'] = df['content.cityRto'].str.strip().str[:2]
        else:
            logging.warning("Column 'content.cityRto' is missing. Unable to extract car state.")

        # Drop the 'content.cityRto' column after using it
        df.drop(columns=['content.cityRto'], inplace=True)

        # Final Cleanup
        # Strip whitespace from all string columns
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)      

        logging.info("Data cleaning completed")
        return df

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Read raw data from the source (e.g., a local file or cloud)
            data = pd.read_excel(self.raw_data_path)
            logging.info(f"Reading data from source: {self.raw_data_path}")

            # Clean the data
            data = self.initial_data_cleaning(data)

            # Ensure the directory for saving files exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the cleaned data to the specified path
            data.to_excel(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            # Perform train-test split (75% train, 25% test)
            logging.info("Train-test split started")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train-test split completed")

            # Save the train and test datasets to specified paths
            train_data.to_excel(self.ingestion_config.train_data_path, index=False)
            test_data.to_excel(self.ingestion_config.test_data_path, index=False)
            logging.info(f"Train data saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to: {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed")
            
            return {
                "train_data": self.ingestion_config.train_data_path,
                "test_data": self.ingestion_config.test_data_path
            }

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise customexception(e, sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()