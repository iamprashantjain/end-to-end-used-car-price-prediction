from src.logger.logging import logging
from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('home.html')

@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Step 1: Collect form data (user input)
            input_data = {
                'ABSAntilockBrakingSystem': float(request.form['ABSAntilockBrakingSystem']),
                'AirConditioner': float(request.form['AirConditioner']),
                'Airbags': float(request.form['Airbags']),
                'Bootspacelitres': float(request.form['Bootspacelitres']),
                'Displacementcc': float(request.form['Displacementcc']),
                'FueltankCapacitylitres': float(request.form['FueltankCapacitylitres']),
                'GroundClearancemm': float(request.form['GroundClearancemm']),
                'tyre_health_pct': float(request.form['tyre_health_pct']),
                'MaxPowerbhp': float(request.form['MaxPowerbhp']),
                'MaxPowerrpm': float(request.form['MaxPowerrpm']),
                'MaxTorqueNm': float(request.form['MaxTorqueNm']),
                'SeatingCapacity': float(request.form['SeatingCapacity']),
                'bodyType': request.form['bodyType'],
                'duplicateKey': float(request.form['duplicateKey']),
                'fitnessUpto_months_remaining': float(request.form['fitnessUpto_months_remaining']),
                'fuelType': request.form['fuelType'],
                'insuranceExpiry_months_remaining': float(request.form['insuranceExpiry_months_remaining']),
                'insuranceType': request.form['insuranceType'],
                'make': request.form['make'],
                'odometerReading': float(request.form['odometerReading']),
                'ownerNumber': float(request.form['ownerNumber']),
                'transmission': request.form['transmission'],
                'year': int(request.form['year']),
                'defects': float(request.form['defects']),
                'repainted': float(request.form['repainted'])
            }

            # Step 2: Convert input data to DataFrame
            custom_data = CustomData(**input_data)
            test_data = custom_data.get_data_as_dataframe()
            logging.info(f"Inputs: {test_data}")            #dont push this to production
            
            # Step 3: Make predictions using the pipeline
            predictions, ci_lower, ci_upper = predict_pipeline.predict(test_data)
            logging.info(f"Predicted Amount: {predictions[0]}")     #dont push this to production

            # Step 4: Return the result to the user
            return render_template('index.html', predictions=predictions[0], ci_lower=ci_lower[0], ci_upper=ci_upper[0])

        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html', predictions=None, ci_lower=None, ci_upper=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=80, debug=True)