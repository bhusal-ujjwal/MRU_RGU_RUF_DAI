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
ruf4 = pd.read_csv("Fourth_task.csv")
ruf4.head()

# Check column names to ensure 'Surgical_Steps' is correctly referenced
print("Columns in DataFrame:", ruf4.columns)

# Strip any leading/trailing spaces from the column names to avoid referencing issues
ruf4.columns = ruf4.columns.str.strip()

# Separate the data based on the surgical_steps value
surgical_steps_1 = ruf4[ruf4['Surgical_Steps'] == 1]
surgical_steps_2 = ruf4[ruf4['Surgical_Steps'] == 2]
surgical_steps_3 = ruf4[ruf4['Surgical_Steps'] == 3]
surgical_steps_4 = ruf4[ruf4['Surgical_Steps'] == 4]


# Remove 'Surgical_Steps' column from calculation
surgical_steps_1 = surgical_steps_1.drop(columns='Surgical_Steps')
surgical_steps_2 = surgical_steps_2.drop(columns='Surgical_Steps')
surgical_steps_3 = surgical_steps_3.drop(columns='Surgical_Steps')
surgical_steps_4 = surgical_steps_4.drop(columns='Surgical_Steps')


# Use describe function to get statistical summary for both groups
describe_steps_1 = surgical_steps_1.describe()
describe_steps_2 = surgical_steps_2.describe()
describe_steps_3 = surgical_steps_3.describe()
describe_steps_4 = surgical_steps_4.describe()

# Display the results
print("Statistical Summary for Surgical Steps 1:")
print(describe_steps_1)

# Display the results
print("Statistical Summary for Surgical Steps 2:")
print(describe_steps_2)

# Display the results
print("Statistical Summary for Surgical Steps 3:")
print(describe_steps_3)

# Display the results
print("Statistical Summary for Surgical Steps 4:")
print(describe_steps_4)

# Extract only count and mean for each surgical step
count_mean_steps_1 = describe_steps_1.loc[['count', 'mean']]
count_mean_steps_2 = describe_steps_2.loc[['count', 'mean']]
count_mean_steps_3 = describe_steps_3.loc[['count', 'mean']]
count_mean_steps_4 = describe_steps_4.loc[['count', 'mean']]

# Display the results
print("Count and Mean for Surgical Steps 1:")
print(count_mean_steps_1)

print("\nCount and Mean for Surgical Steps 2:")
print(count_mean_steps_2)

print("\nCount and Mean for Surgical Steps 3:")
print(count_mean_steps_3)

print("\nCount and Mean for Surgical Steps 4:")
print(count_mean_steps_4)

# Calculate the new means for combined surgical steps 1 and 2
mean_steps_1_2_L_distraction_defect_MRU = (1.838 + 2.45375) / 2
mean_steps_1_2_L_distraction_defect_RGU = (2.274 + 2.89875) / 2

# Calculate the new means for combined surgical steps 3 and 4
mean_steps_3_4_L_distraction_defect_MRU = (3.4 + 3.608) / 2
mean_steps_3_4_L_distraction_defect_RGU = (4.0 + 3.862) / 2

# Create new DataFrames for the combined groups
combined_steps_1_2 = pd.DataFrame({
    'L_distraction_defect_MRU': [mean_steps_1_2_L_distraction_defect_MRU],
    'L_distraction_defect_RGU': [mean_steps_1_2_L_distraction_defect_RGU]
}, index=['mean'])

combined_steps_3_4 = pd.DataFrame({
    'L_distraction_defect_MRU': [mean_steps_3_4_L_distraction_defect_MRU],
    'L_distraction_defect_RGU': [mean_steps_3_4_L_distraction_defect_RGU]
}, index=['mean'])

# Display the results
print("New Mean for Combined Surgical Steps 1 and 2:")
print(combined_steps_1_2)

print("\nNew Mean for Combined Surgical Steps 3 and 4:")
print(combined_steps_3_4)

# Separate the data based on the surgical_steps value
surgical_steps = {}
for step in ruf4['Surgical_Steps'].unique():
    surgical_steps[step] = ruf4[ruf4['Surgical_Steps'] == step].drop(columns='Surgical_Steps')



#yo chai data combine garera nikaleko mean mathiko jasto mean average naliera sidhai data batai garkeo

from scipy.stats import mannwhitneyu

# Combine data for Surgical Steps 1 and 2
combined_data_1_2 = pd.concat([surgical_steps_1, surgical_steps_2])

# Combine data for Surgical Steps 3 and 4
combined_data_3_4 = pd.concat([surgical_steps_3, surgical_steps_4])

# Calculate the new means for combined groups
mean_steps_1_2_L_distraction_defect_MRU = combined_data_1_2['L_distraction_defect_MRU'].mean()
mean_steps_1_2_L_distraction_defect_RGU = combined_data_1_2['L_distraction_defect_RGU'].mean()

mean_steps_3_4_L_distraction_defect_MRU = combined_data_3_4['L_distraction_defect_MRU'].mean()
mean_steps_3_4_L_distraction_defect_RGU = combined_data_3_4['L_distraction_defect_RGU'].mean()

# Display the new means
combined_steps_1_2 = pd.DataFrame({
    'L_distraction_defect_MRU': [mean_steps_1_2_L_distraction_defect_MRU],
    'L_distraction_defect_RGU': [mean_steps_1_2_L_distraction_defect_RGU]
}, index=['mean'])

combined_steps_3_4 = pd.DataFrame({
    'L_distraction_defect_MRU': [mean_steps_3_4_L_distraction_defect_MRU],
    'L_distraction_defect_RGU': [mean_steps_3_4_L_distraction_defect_RGU]
}, index=['mean'])

print("New Mean for Combined Surgical Steps 1 and 2:")
print(combined_steps_1_2)

print("\nNew Mean for Combined Surgical Steps 3 and 4:")
print(combined_steps_3_4)

#ani mathillo steps ma combine gareko data ko adhar ma
from scipy.stats import mannwhitneyu

# Mann-Whitney U Test function
def mann_whitney_test(data1, data2):
    stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
    return stat, p

# Perform Mann-Whitney U test for combined groups 1 and 2
stat_MRU_1_2, p_MRU_1_2 = mann_whitney_test(
    combined_data_1_2['L_distraction_defect_MRU'],
    combined_data_1_2['L_distraction_defect_RGU']
)
stat_RGU_1_2, p_RGU_1_2 = mann_whitney_test(
    combined_data_1_2['L_distraction_defect_RGU'],
    combined_data_1_2['L_distraction_defect_MRU']
)

# Perform Mann-Whitney U test for combined groups 3 and 4
stat_MRU_3_4, p_MRU_3_4 = mann_whitney_test(
    combined_data_3_4['L_distraction_defect_MRU'],
    combined_data_3_4['L_distraction_defect_RGU']
)
stat_RGU_3_4, p_RGU_3_4 = mann_whitney_test(
    combined_data_3_4['L_distraction_defect_RGU'],
    combined_data_3_4['L_distraction_defect_MRU']
)

# Display results
print("Mann-Whitney Result for Surgical Steps 1 and 2 Combination:")
print(f"L_distraction_defect_MRU vs L_distraction_defect_RGU")
print(f"U Statistic: {stat_MRU_1_2:.2f}")
print(f"p-Value: {p_MRU_1_2:.4f}\n")

print("Mann-Whitney Result for Surgical Steps 3 and 4 Combination:")
print(f"L_distraction_defect_MRU vs L_distraction_defect_RGU")
print(f"U Statistic: {stat_MRU_3_4:.2f}")
print(f"p-Value: {p_MRU_3_4:.4f}")







# Upload csv file to colab for using it
upload_file = files.upload()

# Use the uploaded file in colab
ruf5 = pd.read_csv("dai_fifth.csv")
ruf5.head()

# Check column names to ensure 'Surgical_Steps' is correctly referenced
print("Columns in DataFrame:", ruf5.columns)

# Strip any leading/trailing spaces from the column names to avoid referencing issues
ruf5.columns = ruf5.columns.str.strip()

# Separate the data based on the surgical_steps value
surgical_steps_1 = ruf5[ruf5['Surgical_Steps'] == 1]
surgical_steps_2 = ruf5[ruf5['Surgical_Steps'] == 2]


# Remove 'Surgical_Steps' column from calculation
surgical_steps_1 = surgical_steps_1.drop(columns='Surgical_Steps')
surgical_steps_2 = surgical_steps_2.drop(columns='Surgical_Steps')


# Use describe function to get statistical summary for both groups
describe_steps_1 = surgical_steps_1.describe()
describe_steps_2 = surgical_steps_2.describe()


# Display the results
print("Statistical Summary for Surgical Steps 1:")
print(describe_steps_1)

# Display the results
print("Statistical Summary for Surgical Steps 2:")
print(describe_steps_2)

# Extract only count and mean for each surgical step
count_mean_steps_1 = describe_steps_1.loc[['count', 'mean']]
count_mean_steps_2 = describe_steps_2.loc[['count', 'mean']]

# Display the results
print("Count and Mean for Surgical Steps 1:")
print(count_mean_steps_1)

print("\nCount and Mean for Surgical Steps 2:")
print(count_mean_steps_2)

# Calculate mean values for each surgical step
mean_steps_1 = count_mean_steps_1.loc['mean'].values[0]
mean_steps_2 = count_mean_steps_2.loc['mean'].values[0]

# Count the number of values less than and greater than the mean for each surgical step
less_than_mean_steps_1 = (surgical_steps_1['Pubourethral_vertical_distance_MRU'] < mean_steps_1).sum()
greater_than_mean_steps_1 = (surgical_steps_1['Pubourethral_vertical_distance_MRU'] > mean_steps_1).sum()

less_than_mean_steps_2 = (surgical_steps_2['Pubourethral_vertical_distance_MRU'] < mean_steps_2).sum()
greater_than_mean_steps_2 = (surgical_steps_2['Pubourethral_vertical_distance_MRU'] > mean_steps_2).sum()

# Display the results
print(f"\nSurgical Steps = 1")
print(f"Number of values less than mean value(FN): {less_than_mean_steps_1}")
print(f"Number of values greater than mean value(TN): {greater_than_mean_steps_1}")

print(f"\nSurgical Steps = 2")
print(f"Number of values less than mean value(TP): {less_than_mean_steps_2}")
print(f"Number of values greater than mean value(FP): {greater_than_mean_steps_2}")

# Calculate the fixed threshold value
threshold_value = 1.44

# Count the number of values less than and greater than the fixed threshold for each surgical step
less_than_threshold_steps_1 = (surgical_steps_1['Pubourethral_vertical_distance_MRU'] < threshold_value).sum()
greater_than_threshold_steps_1 = (surgical_steps_1['Pubourethral_vertical_distance_MRU'] > threshold_value).sum()

less_than_threshold_steps_2 = (surgical_steps_2['Pubourethral_vertical_distance_MRU'] < threshold_value).sum()
greater_than_threshold_steps_2 = (surgical_steps_2['Pubourethral_vertical_distance_MRU'] > threshold_value).sum()

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
    surgical_steps_1['Pubourethral_vertical_distance_MRU'],
    surgical_steps_2['Pubourethral_vertical_distance_MRU']
)

# Display the results
print("Mann-Whitney Result for Comparison of Surgical Steps 1 and 2:")
print("Pubourethral_vertical_distance_MRU vs Surgical Steps 1 vs Surgical Steps 2")
print("U Statistic:", u_statistic)
print("p-Value:", p_value)

#when the average is taken for both surgical steps value for mean TN, FN , TP , FP.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Updated values
TP = 6
FN = 5
TN = 8
FP = 1


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
columns_to_plot = ruf5.columns[:-1]  # Exclude 'Surgical_Steps'

# plt.figure(figsize=(10, 20))

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(len(columns_to_plot), 1, i)  # Create a vertical layout
    sns.distplot(ruf5[col], fit=norm, kde=False)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency (occurrences)')

plt.tight_layout()
plt.show()

