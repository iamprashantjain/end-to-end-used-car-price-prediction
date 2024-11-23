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

    def initial_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        impliment basic data cleaning, feature engineering here except missing value imputation, scaling & encoding
        """
        logging.info("Data cleaning started")

        #drop all duplicate rows
        df.drop_duplicates(inplace=True)
        
        ## while scraping data from cars24, I've noticed that for some cars specs details are not available so we can avoid those
        ## thats why I've added a column "specs_tag" with values available & not available which can be used to filter those rows
        df = df[df['specs_tag'] == "available"]
        
        # It looks like there are rows where Top features are also unavailable so we can remove rows where
        # all specs missing even after specs_tag filter to avoid wrong data
        # aplying more filter to remove all rows where all specs are missing meaning spec features is not available on website
        df = df[df[['Displacementcc', 'GearBoxNumberOfGears']].notna().all(axis=1)]
        
        df.drop(columns=['TransmissionType'], inplace=True)
        
        ## while assesing data, I've noticed rows where data contains 1.0, 0.0 & NaN.. where 1.0 means availble,
        ## 0.0 measn not available & NaN means not availble, after checking I got to know its bcoz that data is not
        ## available on the website like even though car contains AC it says NaN bcoz data is not available on website
        ## which will create problem at the time of data analysis & will hamper prediction results so we are removing all those
        ## NaN rows where column contains all & only 3 values [1.0,0.0,NaN]


        def is_valid_column(col):
            unique_values = set(col.dropna())  # Exclude NaN for checking unique values
            return unique_values.issubset({1.0, 0.0}) and \
                all(val in unique_values for val in [1.0, 0.0]) and \
                col.isnull().any()  # Ensure at least one NaN is present

        # Identify columns that meet the criteria
        valid_columns = df.columns[df.apply(is_valid_column, axis=0)]

        # Create a new DataFrame with only the valid columns
        filtered_df = df[valid_columns]
        
        df.drop(columns=filtered_df.columns.tolist(), inplace=True)
        
        ## removing filter column which is of no use in analysis & model building
        df.drop(columns=['specs_tag'], inplace=True)
        
        
        from datetime import datetime

        # Convert to datetime
        df['content.fitnessUpto'] = pd.to_datetime(df['content.fitnessUpto'], format='%d-%b-%Y')

        # Get today's date
        today = pd.to_datetime(datetime.now().date())

        df['content.fitnessUpto_months_remaining'] = ((df['content.fitnessUpto'].dt.year - today.year) * 12 + (df['content.fitnessUpto'].dt.month - today.month))
        
        
        # Convert Unix timestamp to datetime
        df['content.insuranceExpiry'] = pd.to_datetime(df['content.insuranceExpiry'], unit='s')

        # Get today's date
        today = pd.to_datetime(datetime.now().date())

        # Calculate the number of months remaining
        df['content.insuranceExpiry_months_remaining'] = ((df['content.insuranceExpiry'].dt.year - today.year) * 12 + (df['content.insuranceExpiry'].dt.month - today.month))

        # Change all values less than 0 to 0
        df['content.insuranceExpiry_months_remaining'] = df['content.insuranceExpiry_months_remaining'].clip(lower=0)

        
        # Convert to datetime (auto-detect format)
        df['content.lastServicedAt'] = pd.to_datetime(df['content.lastServicedAt'])

        # Get today's date
        today = pd.to_datetime(datetime.now().date())

        # Calculate the number of months remaining since the last service
        df['content.lastServicedAt_months_remaining'] = ((df['content.lastServicedAt'].dt.year - today.year) * 12 + (df['content.lastServicedAt'].dt.month - today.month))

        # Change all negative values to positive (use abs())
        df['content.lastServicedAt_months_remaining'] = df['content.lastServicedAt_months_remaining'].abs()

        ## removing columns as already been feature engineered
        df.drop(columns=['content.fitnessUpto', 'content.insuranceExpiry','content.lastServicedAt'], inplace=True)
        
        
        # remove content.city
        df.drop(columns=['content.city'], inplace=True)
        
        
        # extract state code from content.cityRto
        df['content.staterto'] = df['content.cityRto'].str[:2]
        
        #drop state rto
        df.drop(columns=['content.staterto'], inplace=True)
        
        #change NaN to "not available" for selected columns
        cols_to_replace = ['ABSAntilockBrakingSystem', 'PowerWindowsFront', 'PowerWindowsRear', 
                        'AirConditioner', '12VPowerOutlet', 'MultifunctionDisplayScreen', 
                        'EntertainmentDisplayScreen', 'VoiceRecognition', 'SmartCardSmartKey', 
                        'AmbientLighting', 'SunroofMoonroof', 'WirelessChargingPad', 
                        'VentilatedSeatsRear', 'HVACControl', '360DegreeCamera', 
                        'ParkingAssistSide', 'DriverSeatAdjustmentElectric']

        df[cols_to_replace] = df[cols_to_replace].fillna("not available")
        
        
        #missing values in HeadlampBulbTypeHighBeam can be filled using HeadlampBulbTypeLowBeam column
        df['HeadlampBulbTypeHighBeam'] = df['HeadlampBulbTypeHighBeam'].fillna(df['HeadlampBulbTypeLowBeam'])
        
        #strip whole dataset
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        #Change all not available to 0 for better visualization
        # To ensure that the changes are applied only when the columns contain exactly two unique values as specified
        def update_columns(df):
            for col in df.columns:
                unique_values = set(df[col].astype(str).unique())
                #print(f"Processing column: {col}, Unique values: {unique_values}")  # Debugging line
                
                if unique_values == {'1', 'not available'}:
                    df[col] = df[col].replace('not available', 0)
                
                elif unique_values == {'Available', 'Not available'}:
                    df[col] = df[col].replace({'Available': 1, 'Not available': 0})
                    
                elif unique_values == {'0', 'not available'}:
                    df[col] = df[col].replace('not available', 0)
        
        
        # Apply the function to your DataFrame
        update_columns(df)
        
        
        # Change content.duplicatekey True/False to 1/0
        df['content.duplicateKey'] = df['content.duplicateKey'].replace({True: 1, False: 0})
        
        # Extract State from cityrto
        df['car_state'] = df['content.cityRto'].str.strip().str[:2]
        
        # content.cityRto content.registrationNumber content.listingPrice - remove
        df.drop(columns=['content.cityRto','content.registrationNumber','content.listingPrice'], inplace=True)
        
        df.drop(columns=['content.appointmentId'], inplace=True)
        
        # Replace "OK" with 1 and "WARN" with 0 in specified columns
        columns_to_replace = ['Left Front Tyre', 'Right Front Tyre', 'Left Rear Tyre', 'Right Rear Tyre', 'Spare Tyre']
        df[columns_to_replace] = df[columns_to_replace].replace({'OK': 1, 'WARN': 0})
        
        
        # Calculate the sum of 'OK' values
        df['Tyre_Health'] = df[['Left Front Tyre', 'Right Front Tyre', 'Left Rear Tyre', 'Right Rear Tyre', 'Spare Tyre']].sum(axis=1)

        # Calculate the total number of tyres (columns)
        total_tyres = df[['Left Front Tyre', 'Right Front Tyre', 'Left Rear Tyre', 'Right Rear Tyre', 'Spare Tyre']].shape[1]

        # Calculate the percentage of 'OK' values
        df['tyre_health_pct'] = (df['Tyre_Health'] / total_tyres) * 100
        
        #drop feature engineered columns
        df.drop(columns=['Left Front Tyre', 'Right Front Tyre', 'Left Rear Tyre', 'Right Rear Tyre', 'Spare Tyre', 'Tyre_Health'], inplace=True)
        
        #change not available in ABSAntilockBrakingSystem to 0, as bcoz of this i was facing issues
        # For numeric conversion after replacement, you can use pd.to_numeric like this:
        df['ABSAntilockBrakingSystem'] = pd.to_numeric(df['ABSAntilockBrakingSystem'].str.replace("not available", "0"), errors='coerce')

        
        #final df to to used for model building
        df = df[['ABSAntilockBrakingSystem','AirConditioner','Airbags','Bootspacelitres','Displacementcc','FueltankCapacitylitres','GroundClearancemm','tyre_health_pct','MaxPowerbhp','MaxPowerrpm','MaxTorqueNm','SeatingCapacity','content.bodyType','content.duplicateKey','content.fitnessUpto_months_remaining','content.fuelType','content.insuranceExpiry_months_remaining','content.insuranceType','content.make','content.odometerReading','content.ownerNumber','content.transmission','content.year','defects','repainted','content.onRoadPrice']]
        logging.info(f"Final Columns to be used in model building: {df.columns}")
        logging.info(f"current missing value count: {df.isnull().sum()}")        
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
            
            #return train &* test data path
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise customexception(e, sys)
        
        
# if __name__ == "__main__":
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()