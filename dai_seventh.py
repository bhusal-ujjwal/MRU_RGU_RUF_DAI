# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2_contingency
from sklearn.metrics import roc_curve, auc
from google.colab import files
from scipy import stats

# Upload csv file to colab for using it
upload_file = files.upload()

# Use the uploaded file in colab
ruf7 = pd.read_csv("Dai_seventh_task.csv")
ruf7.head()

# Check column names to ensure 'Surgical_Steps' is correctly referenced
print("Columns in DataFrame:", ruf7.columns)

# Strip any leading/trailing spaces from the column names to avoid referencing issues
ruf7.columns = ruf7.columns.str.strip()

# Separate the data based on the Surgical_Steps value
Surgical_Steps_1 = ruf7[ruf7['Surgical_Steps'] == 1]
Surgical_Steps_2 = ruf7[ruf7['Surgical_Steps'] == 2]


# Remove 'Surgical_Steps' column from calculation
Surgical_Steps_1 = Surgical_Steps_1.drop(columns='Surgical_Steps')
Surgical_Steps_2 = Surgical_Steps_2.drop(columns='Surgical_Steps')


# Use describe function to get statistical summary for both groups
describe_steps_1 = Surgical_Steps_1.describe()
describe_steps_2 = Surgical_Steps_2.describe()


# Display the results
print("Statistical Summary for Surgical Steps 1:")
print(describe_steps_1)

# Display the results
print("Statistical Summary for Surgical Steps 2:")
print(describe_steps_2)

# Extract only count and mean for each Surgical Steps
count_mean_steps_1 = describe_steps_1.loc[['count', 'mean']]
count_mean_steps_2 = describe_steps_2.loc[['count', 'mean']]

# Display the results
print("Count and Mean for Surgical Steps 1:")
print(count_mean_steps_1)

print("\nCount and Mean for Surgical Steps 2:")
print(count_mean_steps_2)

# Calculate mean values for each Surgical Steps
mean_steps_1 = count_mean_steps_1.loc['mean'].values[0]
mean_steps_2 = count_mean_steps_2.loc['mean'].values[0]

# Count the number of values less than and greater than the mean for each Surgical Steps
less_than_mean_steps_1 = (Surgical_Steps_1['Pubourethral_stump_l_MRU'] < mean_steps_1).sum()
greater_than_mean_steps_1 = (Surgical_Steps_1['Pubourethral_stump_l_MRU'] > mean_steps_1).sum()

less_than_mean_steps_2 = (Surgical_Steps_2['Pubourethral_stump_l_MRU'] < mean_steps_2).sum()
greater_than_mean_steps_2 = (Surgical_Steps_2['Pubourethral_stump_l_MRU'] > mean_steps_2).sum()

# Display the results
print(f"\nSurgical Steps = 1")
print(f"Number of values less than mean value(FN): {less_than_mean_steps_1}")
print(f"Number of values greater than mean value(TN): {greater_than_mean_steps_1}")

print(f"\nSurgical Steps = 2")
print(f"Number of values less than mean value(TP): {less_than_mean_steps_2}")
print(f"Number of values greater than mean value(FP): {greater_than_mean_steps_2}")

# Calculate the fixed threshold value
threshold_value = 2.475

# Count the number of values less than and greater than the fixed threshold for each Surgical Steps
less_than_threshold_steps_1 = (Surgical_Steps_1['Pubourethral_stump_l_MRU'] < threshold_value).sum()
greater_than_threshold_steps_1 = (Surgical_Steps_1['Pubourethral_stump_l_MRU'] > threshold_value).sum()

less_than_threshold_steps_2 = (Surgical_Steps_2['Pubourethral_stump_l_MRU'] < threshold_value).sum()
greater_than_threshold_steps_2 = (Surgical_Steps_2['Pubourethral_stump_l_MRU'] > threshold_value).sum()

# Display the results
print(f"\nSurgical Steps = 1")
print(f"Number of values less than threshold value (FN): {less_than_threshold_steps_1}")
print(f"Number of values greater than threshold value (TN): {greater_than_threshold_steps_1}")

print(f"\nSurgical Steps = 2")
print(f"Number of values less than threshold value (TP): {less_than_threshold_steps_2}")
print(f"Number of values greater than threshold value (FP): {greater_than_threshold_steps_2}")

from scipy import stats
# Perform Mann-Whitney U test comparing the two groups
from scipy.stats import mannwhitneyu
u_statistic, p_value = stats.mannwhitneyu(
    Surgical_Steps_1['Pubourethral_stump_l_MRU'],
    Surgical_Steps_2['Pubourethral_stump_l_MRU']
)

# Display the results
print("Mann-Whitney Result for Comparison of Surgical Steps 1 and 2:")
print("Pubourethral_stump_l_MRU vs Surgical Steps 1 vs Surgical Steps 2")
print("U Statistic:", u_statistic)
print("p-Value:", p_value)

#when the average is taken for both Surgical Steps value for mean TN, FN , TP , FP.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Updated values
TP = 4
FN = 7
TN = 6
FP = 3


# Sensitivity and Specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Plotting the ROC curve
# For simplicity, let's use these values as the ROC points
fpr = [0, 1 - specificity, 1]  # False Positive Rate
tpr = [0, sensitivity, 1]      # True Positive Rate

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Print Sensitivity and AUC
print(f"Sensitivity: {sensitivity:.3f}")
print(f"AUC: {roc_auc:.3f}")

# Plot distribution plots with normal distribution curve
columns_to_plot = ruf7.columns[:-1]  # Exclude 'Surgical_Steps'

# plt.figure(figsize=(10, 20))

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(len(columns_to_plot), 1, i)  # Create a vertical layout
    sns.distplot(ruf7[col], fit=norm, kde=False)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency (occurrences)')

plt.tight_layout()
plt.show()

