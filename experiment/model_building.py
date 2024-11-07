import pandas as pd
import numpy as np
import warnings;warnings.filterwarnings('ignore')


# display max row & columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)


df = pd.read_excel(r"H:\CampusX_DS\week43 - My Projects Aug 2024\end-to-end-used-car-price-prediction\cars24_final_data.xlsx")


## cleaning

df.drop_duplicates(inplace=True)

df = df[df['specs_tag'] == "available"]
df = df[df[['Displacementcc', 'GearBoxNumberOfGears']].notna().all(axis=1)]
df.drop(columns=['TransmissionType'], inplace=True)

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

# Convert Unix timestamps to datetime
df['content.insuranceExpiry'] = pd.to_datetime(df['content.insuranceExpiry'], unit='s')

# Format the datetime to the desired string format
df['content.insuranceExpiry'] = df['content.insuranceExpiry'].dt.strftime('%d-%b-%Y')

# Convert the datetime strings to datetime objects
df['content.lastServicedAt'] = pd.to_datetime(df['content.lastServicedAt'])

# Format the datetime to the desired string format
df['content.lastServicedAt'] = df['content.lastServicedAt'].dt.strftime('%d-%b-%Y')

## removing filter column which is of no use in analysis & model building

df.drop(columns=['specs_tag'], inplace=True)

## change content.fitnessUpto, content.insuranceExpiry & content.lastServicedAt column to months from today

from datetime import datetime

# Convert to datetime
df['content.fitnessUpto'] = pd.to_datetime(df['content.fitnessUpto'], format='%d-%b-%Y')

# Get today's date
today = pd.to_datetime(datetime.now().date())

df['content.fitnessUpto_months_remaining'] = ((df['content.fitnessUpto'].dt.year - today.year) * 12 + 
                          (df['content.fitnessUpto'].dt.month - today.month))

from datetime import datetime

# Convert to datetime
df['content.insuranceExpiry'] = pd.to_datetime(df['content.insuranceExpiry'], format='%d-%b-%Y')

# Get today's date
today = pd.to_datetime(datetime.now().date())

df['content.insuranceExpiry_months_remaining'] = ((df['content.insuranceExpiry'].dt.year - today.year) * 12 + (df['content.insuranceExpiry'].dt.month - today.month))
df
## change all values less then 0 to 0
df['content.insuranceExpiry_months_remaining'] = df['content.insuranceExpiry_months_remaining'].clip(lower=0)

from datetime import datetime

# Convert to datetime
df['content.lastServicedAt'] = pd.to_datetime(df['content.lastServicedAt'], format='%d-%b-%Y')

# Get today's date
today = pd.to_datetime(datetime.now().date())

df['content.lastServicedAt_months_remaining'] = ((df['content.lastServicedAt'].dt.year - today.year) * 12 + (df['content.lastServicedAt'].dt.month - today.month))

## change all negative values to positive
df['content.lastServicedAt_months_remaining'] = df['content.lastServicedAt_months_remaining'].abs()

## removing columns as already been feature engineered

df.drop(columns=['content.fitnessUpto', 'content.insuranceExpiry','content.lastServicedAt'], inplace=True)

##remove content.city
df.drop(columns=['content.city'], inplace=True)

## extract state code from content.cityRto
df['content.staterto'] = df['content.cityRto'].str[:2]

df.drop(columns=['content.staterto'], inplace=True)

## change NaN to "not available" for selected columns

cols_to_replace = ['ABSAntilockBrakingSystem', 'PowerWindowsFront', 'PowerWindowsRear', 
                   'AirConditioner', '12VPowerOutlet', 'MultifunctionDisplayScreen', 
                   'EntertainmentDisplayScreen', 'VoiceRecognition', 'SmartCardSmartKey', 
                   'AmbientLighting', 'SunroofMoonroof', 'WirelessChargingPad', 
                   'VentilatedSeatsRear', 'HVACControl', '360DegreeCamera', 
                   'ParkingAssistSide', 'DriverSeatAdjustmentElectric']

df[cols_to_replace] = df[cols_to_replace].fillna("not available")


## missing values in HeadlampBulbTypeHighBeam can be filled using HeadlampBulbTypeLowBeam column
df['HeadlampBulbTypeHighBeam'] = df['HeadlampBulbTypeHighBeam'].fillna(df['HeadlampBulbTypeLowBeam'])

## strip whole dataset
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

## Change all not available to 0 for better visualization
## To ensure that the changes are applied only when the columns contain exactly two unique values as specified

def update_columns(df):
    for col in df.columns:
        unique_values = set(df[col].astype(str).unique())
#         print(f"Processing column: {col}, Unique values: {unique_values}")  # Debugging line
        
        if unique_values == {'1', 'not available'}:
            df[col] = df[col].replace('not available', 0)
        
        elif unique_values == {'Available', 'Not available'}:
            df[col] = df[col].replace({'Available': 1, 'Not available': 0})
            
        elif unique_values == {'0', 'not available'}:
            df[col] = df[col].replace('not available', 0)
            
# Apply the function to your DataFrame
update_columns(df)


## Change content.duplicatekey True/False to 1/0
df['content.duplicateKey'] = df['content.duplicateKey'].replace({True: 1, False: 0})


## Extract State from cityrto
df['car_state'] = df['content.cityRto'].str.strip().str[:2]


## content.cityRto content.registrationNumber content.listingPrice - remove
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


df.drop(columns=['Left Front Tyre', 'Right Front Tyre', 'Left Rear Tyre', 'Right Rear Tyre', 'Spare Tyre', 'Tyre_Health'], inplace=True)


df['ABSAntilockBrakingSystem'] = df['ABSAntilockBrakingSystem'].replace('not available', 0)



## check & drop based on corealtion
# Calculate the correlation matrix
correlation_matrix = df[['MaxPowerbhp', 'MaxPowerrpm', 'MaxTorqueNm']].corr()

# Set a threshold for correlation
threshold = 0.9

# Identify columns to drop
columns_to_drop = set()

# Loop through the correlation matrix to find pairs with correlation above the threshold
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            columns_to_drop.add(colname)

# Drop the identified columns from the DataFrame
df.drop(columns=columns_to_drop, inplace=True)

df = df[['ABSAntilockBrakingSystem','AirConditioner','Airbags','Bootspacelitres','Displacementcc','FueltankCapacitylitres','GroundClearancemm','tyre_health_pct','MaxPowerbhp','MaxPowerrpm','MaxTorqueNm','SeatingCapacity','content.bodyType','content.duplicateKey','content.fitnessUpto_months_remaining','content.fuelType','content.insuranceExpiry_months_remaining','content.insuranceType','content.make','content.odometerReading','content.ownerNumber','content.transmission','content.year','defects','repainted','content.onRoadPrice']]


##pre-processing
## In preprocessing: missing value imputation, encoding, scaling etc

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming your df is a DataFrame with the data

# Separate features (X) and target (y)
X = df.drop(columns=['content.onRoadPrice'])
y = df['content.onRoadPrice']

# Identify categorical columns (object type and binary 1/0 columns)
cat_cols = X.select_dtypes(include="object").columns.tolist()

# Add columns with exactly two unique values (0, 1) to categorical columns
binary_cols = [col for col in X.columns if set(X[col].dropna().unique()) == {0, 1}]
cat_cols.extend(binary_cols)

# Identify numerical columns (remaining columns after removing categorical ones)
num_cols = [col for col in X.columns if col not in cat_cols]

# Create separate pipelines for numeric and categorical transformations
num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),  # Use mean imputation for numerical data
        ("scaler", StandardScaler())  # Standardize numerical features
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Use most frequent value for categorical imputation
        ("encoder", OneHotEncoder(sparse_output=False, drop='first'))  # One-hot encode, drop first to avoid multicollinearity
    ]
)

# Create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num_pipeline', num_pipeline, num_cols),  # Apply num_pipeline to numerical columns
        ('cat_pipeline', cat_pipeline, cat_cols)   # Apply cat_pipeline to categorical columns
    ]
)

# Split the data into train and test sets (be sure to pass y_train, not X_train, for the target variable)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform transformations on the training data
X_train_transformed = preprocessor.fit_transform(X_train)

# Apply the transformations to the test data
X_test_transformed = preprocessor.transform(X_test)

# Get the feature names after transformations
feature_names = preprocessor.get_feature_names_out()

# Convert the transformed data back to a DataFrame with the appropriate column names
X_train = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test = pd.DataFrame(X_test_transformed, columns=feature_names)



import pdb;pdb.set_trace()