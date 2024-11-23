import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object, calculate_95_ci
import joblib

class PredictPipeline:

    def __init__(self):
        print("Initializing PredictPipeline object")

    def predict(self, features):
        """
        Makes a prediction based on the input features by first applying the preprocessor,
        then using the trained model to make predictions.

        :param features: The input features to make predictions on.
        :return: The predicted results and their 95% confidence interval.
        """
        try:
            # Define the paths for the preprocessor and model
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            # Load the preprocessor and model using a utility function
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Scale the input features using the preprocessor
            scaled_features = preprocessor.transform(features)

            # Make predictions using the trained model
            predictions = model.predict(scaled_features)

            # Get model type (you can customize this as per your implementation)
            model_type = type(model).__name__

            # Calculate 95% confidence interval (lower and upper bounds)
            ci_lower, ci_upper = calculate_95_ci(model, scaled_features, predictions, model_type)

            return predictions, ci_lower, ci_upper


        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 ABSAntilockBrakingSystem: float,
                 AirConditioner: float,
                 Airbags: float,
                 Bootspacelitres: float,
                 Displacementcc: float,
                 FueltankCapacitylitres: float,
                 GroundClearancemm: float,
                 tyre_health_pct: float,
                 MaxPowerbhp: float,
                 MaxPowerrpm: float,
                 MaxTorqueNm: float,
                 SeatingCapacity: float,
                 bodyType: str,
                 duplicateKey: float,
                 fitnessUpto_months_remaining: float,
                 fuelType: str,
                 insuranceExpiry_months_remaining: float,
                 insuranceType: str,
                 make: str,
                 odometerReading: float,
                 ownerNumber: int,
                 transmission: str,
                 year: int,
                 defects: float,
                 repainted: float
                 ):

        # Initialize the features
        self.ABSAntilockBrakingSystem = ABSAntilockBrakingSystem
        self.AirConditioner = AirConditioner
        self.Airbags = Airbags
        self.Bootspacelitres = Bootspacelitres
        self.Displacementcc = Displacementcc
        self.FueltankCapacitylitres = FueltankCapacitylitres
        self.GroundClearancemm = GroundClearancemm
        self.tyre_health_pct = tyre_health_pct
        self.MaxPowerbhp = MaxPowerbhp
        self.MaxPowerrpm = MaxPowerrpm
        self.MaxTorqueNm = MaxTorqueNm
        self.SeatingCapacity = SeatingCapacity
        self.bodyType = bodyType
        self.duplicateKey = duplicateKey
        self.fitnessUpto_months_remaining = fitnessUpto_months_remaining
        self.fuelType = fuelType
        self.insuranceExpiry_months_remaining = insuranceExpiry_months_remaining
        self.insuranceType = insuranceType
        self.make = make
        self.odometerReading = odometerReading
        self.ownerNumber = ownerNumber
        self.transmission = transmission
        self.year = year
        self.defects = defects
        self.repainted = repainted

    def get_data_as_dataframe(self):
        try:
            # Convert the features into a dictionary
            custom_data_input_dict = {
                'ABSAntilockBrakingSystem': [self.ABSAntilockBrakingSystem],
                'AirConditioner': [self.AirConditioner],
                'Airbags': [self.Airbags],
                'Bootspacelitres': [self.Bootspacelitres],
                'Displacementcc': [self.Displacementcc],
                'FueltankCapacitylitres': [self.FueltankCapacitylitres],
                'GroundClearancemm': [self.GroundClearancemm],
                'tyre_health_pct': [self.tyre_health_pct],
                'MaxPowerbhp': [self.MaxPowerbhp],
                'MaxPowerrpm': [self.MaxPowerrpm],
                'MaxTorqueNm': [self.MaxTorqueNm],
                'SeatingCapacity': [self.SeatingCapacity],
                'content.bodyType': [self.bodyType],
                'content.duplicateKey': [self.duplicateKey],
                'content.fitnessUpto_months_remaining': [self.fitnessUpto_months_remaining],
                'content.fuelType': [self.fuelType],
                'content.insuranceExpiry_months_remaining': [self.insuranceExpiry_months_remaining],
                'content.insuranceType': [self.insuranceType],
                'content.make': [self.make],
                'content.odometerReading': [self.odometerReading],
                'content.ownerNumber': [self.ownerNumber],
                'content.transmission': [self.transmission],
                'content.year': [self.year],
                'defects': [self.defects],
                'repainted': [self.repainted],
            }

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Custom input data converted to dataframe successfully')

            return df
        except Exception as e:
            logging.error('Exception occurred while converting data to DataFrame')
            raise customexception(e, sys)