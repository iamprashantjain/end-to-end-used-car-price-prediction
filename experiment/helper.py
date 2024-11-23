#data analysis eda imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


#visualize missing values
import missingno as msno
msno.bar(df)



# Set display option to format floating point numbers
pd.set_option('display.float_format', lambda x: '%.2f' % x)


#drop missing values for specific cols
df = df.dropna(subset=['rating', 'reviews'])


#find rows contains 'warranty' & avoid na values & case sensitivity
df[df['other_info'].str.contains('Warranty', case=False, na=False)]

#multiple words
pattern = 'Year|Months'
df[df['other_info'].str.contains(pattern, case=False, na=False)]



#change multiple rows values
# Update rows where 'processor' contains 'Qualcomm Snapdragon'
df.loc[df['processor'].str.contains("Qualcomm Snapdragon", case=False, na=False), 'processor'] = "Qualcomm Snapdragon"

# Update rows where 'processor' contains 'MTK'
df.loc[df['processor'].str.contains("MTK", case=False, na=False), 'processor'] = "MediaTek"


#rename column name
df.rename(columns={'rear_camera':'rear_camera_in_MP'})


#fill nae
df['expandable'].fillna("Not Available", inplace=True)



#change data type
# Step 1: Replace non-numeric values with NaN and handle them if needed
df['battery_mAh'] = pd.to_numeric(df['battery_mAh'], errors='coerce')

# Step 2: Convert the column to integer (after handling NaNs)
df['battery_mAh'] = df['battery_mAh'].fillna(0).astype('int')


#univariate analysis of numerical col
def univariate_numerical(df, col, bins):
    # Print descriptive statistics
    print(df[col].describe())
    
    # Set up a figure with subplots for multiple plots
    plt.figure(figsize=(14, 10))
    
    # Histogram
    plt.subplot(2, 2, 1)
    df[col].plot(kind='hist', bins=bins, ax=plt.gca(), color='skyblue', edgecolor='black')
    plt.title('Histogram')
    
    # KDE plot
    plt.subplot(2, 2, 2)
    df[col].plot(kind='kde', ax=plt.gca(), color='orange')
    plt.title('KDE Plot')
    
    # Box plot
    plt.subplot(2, 2, 3)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title('Box Plot')
    
    # Print skewness
    skewness = df[col].skew()
    print(f'Skewness: {skewness:.2f}')
    
    plt.tight_layout()
    plt.show()




# find outlier based on box plot
def find_outliers(df, column_name):
    """
    Identify outliers in a DataFrame column using the IQR method.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to analyze.

    Returns:
    - pd.DataFrame: DataFrame containing the outliers.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Calculate quartiles and IQR
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]

    return outliers



#univariate categorical
def univariate_categorical(df, col):
    """
    Analyzes and visualizes a categorical column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col (str): The name of the categorical column to analyze.

    Returns:
    - None: Displays plots and prints the number of missing values.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")
    
    # Get value counts
    counts = df[col].value_counts()
    
    # Print value counts
    print("Value Counts:\n", counts)
    
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    counts.plot(kind='bar', color='skyblue')
    plt.title(f'Bar Chart of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    
    # Plot pie chart
    plt.subplot(1, 2, 2)
    counts.plot(kind='pie', autopct='%0.1f%%', colors=plt.cm.Paired(range(len(counts))))
    plt.title(f'Pie Chart of {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Print number of missing values
    missing_values = df[col].isnull().sum()
    print(f"Number of missing values in '{col}': {missing_values}")



# display max row & columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)



#regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def adjusted_r2_score(y_true, y_pred, num_predictors):
    """
    Calculate the Adjusted R-squared score.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.
    num_predictors (int): Number of predictors (features) used in the model.
    
    Returns:
    float: Adjusted R-squared score.
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - num_predictors - 1)

def evaluate_regression_metrics(y_test, y_pred, num_predictors):
    """
    Calculate various regression metrics to evaluate model performance.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.
    num_predictors (int): Number of predictors (features) used in the model.

    Returns:
    dict: A dictionary containing the regression metrics.
    """
    metrics = {}

    # Calculate Mean Absolute Error
    metrics['MAE'] = mean_absolute_error(y_test, y_pred)

    # Calculate Mean Squared Error
    metrics['MSE'] = mean_squared_error(y_test, y_pred)

    # Calculate Root Mean Squared Error
    metrics['RMSE'] = np.sqrt(metrics['MSE'])

    # Calculate R-squared (Coefficient of Determination)
    metrics['R2'] = r2_score(y_test, y_pred)

    # Calculate Adjusted R-squared
    metrics['Adjusted R2'] = adjusted_r2_score(y_test, y_pred, num_predictors)

    return metrics




#linearity assumptions
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, omni_normtest
from statsmodels.stats.outliers_influence import variance_inflation_factor

def analyze_regression(df, target):
    # Extract column names and predictors
    columns = df.columns.tolist()
    predictors = [col for col in columns if col != target]

    # 1. Normality of Target Variable (Optional)
    plt.figure(figsize=(14, 6))
    
    # Scatter Plot of Target vs First Predictor (if needed)
    if predictors:
        plt.subplot(1, 2, 1)
        plt.scatter(df[predictors[0]], df[target], alpha=0.5)
        plt.xlabel(predictors[0])
        plt.ylabel(target)
        plt.title(f'Scatter Plot of {target} vs {predictors[0]}')
    
    # Residual Plot (Optional)
    plt.subplot(1, 2, 2)
    model = smf.ols(f'{target} ~ {predictors[0]}', data=df).fit()
    residuals = model.resid
    fitted = model.fittedvalues
    plt.scatter(fitted, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Fitted ({predictors[0]})')
    
    plt.tight_layout()
    plt.show()

    for predictor in predictors:
        # Fit the model
        formula = f'{target} ~ {predictor}'
        model = smf.ols(formula, data=df).fit()
        residuals = model.resid
        fitted = model.fittedvalues
        
        # 2. Normality of Residuals
        plt.figure(figsize=(15, 10))
        
        # Histogram/KDE plot of residuals
        plt.subplot(2, 2, 1)
        sns.histplot(residuals, kde=True)
        plt.title(f'Histogram/KDE of Residuals ({predictor})')
        
        # Q-Q Plot of residuals
        plt.subplot(2, 2, 2)
        qqplot(residuals, line='s', ax=plt.gca())
        plt.title(f'Q-Q Plot of Residuals ({predictor})')
        
        # Residuals vs Fitted values plot
        plt.subplot(2, 2, 3)
        plt.scatter(fitted, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title(f'Residuals vs Fitted ({predictor})')
        
        # Apply Omnibus or Jarque-Bera test
        try:
            jb_stat, jb_pvalue, *_ = jarque_bera(residuals)  # Handle additional values if present
            print(f"Jarque-Bera Test for {predictor}: Statistic={jb_stat:.2f}, p-value={jb_pvalue:.2f}")
            print("    Interpretation: p-value < 0.05 indicates residuals are not normally distributed.")
        except ValueError as e:
            print(f"Jarque-Bera Test for {predictor} failed: {e}")
        
        try:
            omni_stat, omni_pvalue = omni_normtest(residuals)
            print(f"Omnibus Test for {predictor}: Statistic={omni_stat:.2f}, p-value={omni_pvalue:.2f}")
            print("    Interpretation: p-value < 0.05 indicates residuals are not normally distributed.")
        except ValueError as e:
            print(f"Omnibus Test for {predictor} failed: {e}")
        
        plt.tight_layout()
        plt.show()

    # 3. Homoscedasticity
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(fitted, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    
    # Perform Breusch-Pagan test
    bp_test = het_breuschpagan(residuals, model.model.exog)
    print(f"Breusch-Pagan test for homoscedasticity:")
    print(f"  LM statistic: {bp_test[0]:.2f}")
    print(f"  LM p-value: {bp_test[1]:.2f}")
    print("    Interpretation: p-value < 0.05 indicates heteroscedasticity.")
    
    # 4. Multicollinearity
    plt.subplot(1, 2, 2)
    # Heatmap of Correlation Matrix
    corr_matrix = df[predictors].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap of Correlation Matrix')
    plt.show()

    # Compute VIF
    exog = df[predictors]
    vif_data = pd.DataFrame()
    vif_data['Variable'] = exog.columns
    vif_data['VIF'] = [variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])]
    print("Variance Inflation Factor (VIF) for each feature:")
    print(vif_data)
    print("    Interpretation: VIF > 10 indicates high multicollinearity.")





#fill na values only
extracted_values_graphics_card1 = temp_df[temp_df['graphics_card'].isnull()]['features'].str.split(",").str[5].str.split("u2009")

temp_df.loc[temp_df['graphics_card'].isnull(), 'graphics_card'] = extracted_values_graphics_card1    






#shift horizontally
condition = df['feature_2'].str.contains('Threads', na=False)
df.loc[condition, 'feature_2'] = df.loc[condition, 'feature_2'].shift(1, fill_value='')



#shift rows horizontally
#shift rows horizontally

values_to_shift = [
    '16GB LPDDR5X RAM', '32GB RAM', '16GB LPDDR5x RAM', '32GB LPDDR5X RAM',
    '8GB DDR4 RAM', '16GB LPDDR3 RAM', '32GB DDR5 RAM', '8GB RAM', '4GB LPDDR4 RAM', '32GB  RAM', '8GB  RAM'
]


# Define the condition to filter rows
condition = df1['cores'].isin(values_to_shift)

# Function to shift row values to the right
def shift_row(row):
    new_row = [''] + row[:-1].tolist()  # Shift values to the right
    return pd.Series(new_row, index=row.index)

# Apply the shift_row function to rows meeting the condition
df1.loc[condition] = df1.loc[condition].apply(shift_row, axis=1)






#update missing 
graphics_terms = ['NVIDIA GeForce RTX 3050', 'Apple M1 Integrated Graphics', 'NVIDIA GeForce RTX 4070','8-Core GPU', '10 Core GPU','Qualcomm Adreno GPU', 'Arm Mali-G72 MP3', '40 Core GPU', 'NVIDIA','40 Core GPU','NVIDIA GeForce GTX 1650','NVIDIA GeForce RTX 4080','14 Core GPU','ARM Mali G72', 'Intel Arc Graphics', 'Intel Integrated Integrated', 'AMD Radeon Graphics']

# Function to update missing core values
def update_graphics(features_str, existing_cores):
    if pd.notna(existing_cores):
        return existing_cores  # Return existing value if not NaN
    
    features_list = eval(features_str)  # Convert string representation of list to list
    for feature in features_list:
        for term in graphics_terms:
            if term.lower() in feature.lower():  # Case-insensitive match
                return term
    return np.nan  # Return NaN if no term is found


df['graphics'] = df.apply(lambda row: update_graphics(row['features'], row['cores']), axis=1)    




#view all duplicates
all_duplicates = df1[df1.duplicated(keep=False)]




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def bivariate_analysis_num_num(df, col1, col2):
    """
    Perform bivariate analysis between two numerical columns of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    col1 (str): Name of the first numerical column.
    col2 (str): Name of the second numerical column.
    
    Returns:
    None
    """
    # Check if columns exist
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns '{col1}' or '{col2}' do not exist in the DataFrame")
    
    # Extract columns
    x = df[col1]
    y = df[col2]
    
    # Scatter Plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(x, y, alpha=0.5)
    plt.title(f'Scatter Plot of {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    
    # 2D Histogram Plot
    plt.subplot(2, 2, 2)
    plt.hist2d(x, y, bins=30, cmap='Blues', alpha=0.7)
    plt.colorbar(label='Frequency')
    plt.title(f'2D Histogram of {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    
    # KDE Plot
    plt.subplot(2, 2, 3)
    sns.kdeplot(x, y, cmap='Blues', fill=True)
    plt.title(f'KDE Plot of {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    
    # Correlation Coefficient
    correlation, _ = pearsonr(x, y)
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.5, f'Pearson Correlation Coefficient:\n{correlation:.2f}', fontsize=15)
    plt.title('Correlation Analysis')
    
    plt.tight_layout()
    plt.show()





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def bivariate_analysis_num_cat(df, numerical_col, categorical_col):
    """
    Perform bivariate analysis between a numerical and a categorical column.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    numerical_col (str): Name of the numerical column.
    categorical_col (str): Name of the categorical column.
    
    Returns:
    None
    """
    # Check if columns exist
    if numerical_col not in df.columns or categorical_col not in df.columns:
        raise ValueError(f"Columns '{numerical_col}' or '{categorical_col}' do not exist in the DataFrame")

    # Set up the figure and axes
    plt.figure(figsize=(20, 18))

    # Bar Plot (Mean of Numerical Data by Category)
    plt.subplot(3, 2, 1)
    sns.barplot(x=categorical_col, y=numerical_col, data=df, estimator=np.mean, ci=None, palette='viridis')
    plt.title(f'Bar Plot: Mean {numerical_col} by {categorical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel(f'Mean {numerical_col}')
    plt.xticks(rotation=45)

    # Box Plot
    plt.subplot(3, 2, 2)
    sns.boxplot(x=categorical_col, y=numerical_col, data=df, palette='viridis')
    plt.title(f'Box Plot: {numerical_col} by {categorical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel(numerical_col)
    plt.xticks(rotation=45)

    # Violin Plot
    plt.subplot(3, 2, 3)
    sns.violinplot(x=categorical_col, y=numerical_col, data=df, palette='viridis')
    plt.title(f'Violin Plot: {numerical_col} by {categorical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel(numerical_col)
    plt.xticks(rotation=45)

    # KDE Plot
    plt.subplot(3, 2, 4)
    sns.kdeplot(data=df, x=numerical_col, hue=categorical_col, common_norm=False, palette='viridis', fill=True)
    plt.title(f'KDE Plot: {numerical_col} by {categorical_col}')
    plt.xlabel(numerical_col)
    plt.ylabel('Density')

    # Swarm Plot
    plt.subplot(3, 2, 5)
    sns.swarmplot(x=categorical_col, y=numerical_col, data=df, color='k', alpha=0.5)
    plt.title(f'Swarm Plot: {numerical_col} by {categorical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel(numerical_col)
    plt.xticks(rotation=45)

    # Strip Plot
    plt.subplot(3, 2, 6)
    sns.stripplot(x=categorical_col, y=numerical_col, data=df, jitter=True, palette='viridis', alpha=0.6)
    plt.title(f'Strip Plot: {numerical_col} by {categorical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel(numerical_col)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squarify  # For treemaps

def bivariate_analysis_cat_cat(df, cat_col1, cat_col2):
    """
    Perform bivariate analysis between two categorical columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    cat_col1 (str): Name of the first categorical column.
    cat_col2 (str): Name of the second categorical column.
    
    Returns:
    None
    """
    # Check if columns exist
    if cat_col1 not in df.columns or cat_col2 not in df.columns:
        raise ValueError(f"Columns '{cat_col1}' or '{cat_col2}' do not exist in the DataFrame")

    # Create a contingency table (crosstab)
    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
    print("Contingency Table:")
    print(contingency_table)

    # Set up the figure for plots
    plt.figure(figsize=(18, 12))

    # Heatmap of the contingency table
    plt.subplot(2, 2, 1)
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='viridis')
    plt.title(f'Heatmap: {cat_col1} vs {cat_col2}')
    plt.xlabel(cat_col2)
    plt.ylabel(cat_col1)

    # Stacked Bar Plot
    plt.subplot(2, 2, 2)
    contingency_table.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
    plt.title(f'Stacked Bar Plot: {cat_col1} vs {cat_col2}')
    plt.xlabel(cat_col1)
    plt.ylabel('Counts')
    plt.xticks(rotation=45)

    # Treemap
    plt.subplot(2, 2, 3)
    sizes = contingency_table.values.flatten()
    labels = [f"{i}\n{j}" for i in contingency_table.index for j in contingency_table.columns]
    squarify.plot(sizes=sizes, label=labels, alpha=.7, color=sns.color_palette('viridis', len(sizes)))
    plt.title(f'Treemap: {cat_col1} vs {cat_col2}')
    plt.axis('off')  # Hide the axes

    plt.tight_layout()
    plt.show()



# Identify and view duplicate rows
duplicates = df[df.duplicated(keep=False)]  # keep=False shows all duplicates    


#warnings
import warnings;warnings.filterwarnings('ignore')



# -----------------------------------------------------------------------------------------------------------------------------------

# Pair Plot: Use a pair plot for a quick overview of relationships between several numerical variables and the target.
# multi-variate

sns.pairplot(data_df_v2, vars=other_numerical_cols + ['content.onRoadPrice'], diag_kind='kde')
plt.show()




# Facet Grid: Use sns.FacetGrid to visualize the relationship of multiple categorical features with the target variable.
# multi-variate

# To use sns.FacetGrid with multiple categorical columns for visualizing content.onRoadPrice,
# we need to create a single grid that visualizes all categorical columns at once.

# Melt the DataFrame to long format for FacetGrid
melted_df = data_df_v2.melt(id_vars='content.onRoadPrice', value_vars=categorical_columns, 
                             var_name='Categorical Column', value_name='Category')

# Create the FacetGrid
g = sns.FacetGrid(melted_df, col='Categorical Column', col_wrap=4, height=4)
g.map(sns.boxplot, 'Category', 'content.onRoadPrice')
g.set_titles(col_template="{col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Boxplots of content.onRoadPrice by Categorical Variables', fontsize=16)
plt.xticks(rotation=45)
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------
#check corelation of numerical column with other columns


# List of other numerical columns to include
other_numerical_cols = [
    'content.onRoadPrice',
    'content.ownerNumber',
    'content.odometerReading',
    'Airbags',
    'NumberOfSpeakers',
    'Displacementcc',
    'Cylinders',
    'GearBoxNumberOfGears',
    'NumberOfDiscBrakes',
    'GroundClearancemm',
    'SeatingCapacity',
    'Bootspacelitres',
    'Widthmm',
    'Lengthmm',
    'WheelBasemm',
    'FueltankCapacitylitres',
    'MaxPowerbhp',
    'MaxPowerrpm',
    'MaxTorqueNm',
    'defects',
    'repainted',
    'MultifunctionDisplayScreenSizein',
    'EntertainmentDisplayScreenSizein',
    'content.fitnessUpto_months_remaining',
    'content.insuranceExpiry_months_remaining',
    'content.lastServicedAt_months_remaining'
]

# Filter the DataFrame to include only the specified numerical columns
filtered_df = data_df_v2[other_numerical_cols]

# Calculate the correlation matrix for the filtered DataFrame
correlation_matrix = filtered_df.corr()

# Sort the correlation values for 'Airbags'
sorted_correlations = correlation_matrix['Airbags'].sort_values(ascending=False)

# Create a DataFrame from the sorted correlations
sorted_correlations_df = sorted_correlations.to_frame()

# Create a heatmap focused on the correlation of Airbags
plt.figure(figsize=(10, 6))
plt.title('Correlation of Airbags with Other Numerical Features')

# Plot the heatmap without annotations
sns.heatmap(sorted_correlations_df, annot=False, cmap='coolwarm', fmt='.2f', square=True)

# Show the plot
plt.show()

#----------------------------------------------------------------------------------------------


#visualize the relationship of one specific categorical column with each of the other columns categorical & numerical

# Define your target categorical column
target_categorical_col = 'InstrumentPanelType'

# Count plots for categorical columns
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data_df_v2, x=target_categorical_col, hue=col)
    plt.title(f'Count of {target_categorical_col} by {col}')
    plt.xlabel(target_categorical_col)
    plt.ylabel('Count')
    plt.legend(title=col)
    plt.show()

# Box plots for numerical columns
for col in other_numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_df_v2, x=target_categorical_col, y=col)
    plt.title(f'Boxplot of {col} by {target_categorical_col}')
    plt.xlabel(target_categorical_col)
    plt.ylabel(col)
    plt.show()
