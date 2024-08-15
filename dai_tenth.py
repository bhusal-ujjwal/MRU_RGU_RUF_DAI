import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm, chi2_contingency
from sklearn.metrics import roc_curve, auc
from google.colab import files

# Upload csv file to colab for using it
upload_file = files.upload()

# Use the uploaded file in colab
ruf10 = pd.read_csv("Dai_tenth_task.csv")
ruf10.head()

# Actual values
actual = ruf10['Gap_length_surgery']

# Predicted values from two machines
predicted_MRU = ruf10['Distraction_defect_MRU']
predicted_RGU = ruf10['distraction_defect_RGU']

# Calculating Absolute Error
ruf10['AE_MRU'] = abs(actual - predicted_MRU)
ruf10['AE_RGU'] = abs(actual - predicted_RGU)

# Calculating Relative Error
ruf10['RE_MRU'] = ruf10['AE_MRU'] / actual
ruf10['RE_RGU'] = ruf10['AE_RGU'] / actual

# Calculating Error Percentage
ruf10['EP_MRU'] = ruf10['RE_MRU'] * 100
ruf10['EP_RGU'] = ruf10['RE_RGU'] * 100

# Display the first 20 rows with the calculated columns
print("Dataset with Calculated Errors:")
print(ruf10.head(20).to_string(index=False))

# Statistical summary of the errors
error_summary = ruf10[['AE_MRU', 'AE_RGU', 'RE_MRU', 'RE_RGU', 'EP_MRU', 'EP_RGU']].describe()

print("\nError Summary:")
print(error_summary)

# Standard Error calculations
n = len(ruf10)

# MRU
se_MRU = np.std(ruf10['AE_MRU'], ddof=1) / np.sqrt(n)
# RGU
se_RGU = np.std(ruf10['AE_RGU'], ddof=1) / np.sqrt(n)

print("\nStandard Error (SE) for MRU predictions:", se_MRU)
print("Standard Error (SE) for RGU predictions:", se_RGU)

mean_actual = np.mean(actual)

# SEP for MRU
sep_MRU = (se_MRU / mean_actual) * 100

# SEP for RGU
sep_RGU = (se_RGU / mean_actual) * 100

print("\nStandard Error Percentage (SEP) for MRU predictions:", sep_MRU)
print("Standard Error Percentage (SEP) for RGU predictions:", sep_RGU)

