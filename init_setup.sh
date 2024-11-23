#!/bin/bash

# Get the current date and time
date=$(date)

echo "[$date]: START"
echo "[$date]: creating venv"

# Create the virtual environment
virtualenv -p python venv 

echo "[$date]: activating environment"

# Activate the virtual environment
source venv/Scripts/activate

echo "[$date]: installing dev requirements"
pip install -r requirements_dev.txt

echo "[$date]: pip upgrade"
python.exe -m pip install --upgrade pip


echo "[$date]: END"