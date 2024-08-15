# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2_contingency, ttest_ind, spearmanr
from sklearn.metrics import roc_curve, auc
from google.colab import files

# Upload csv file to colab for using it
upload_file = files.upload()

# Use the uploaded file in colab
ruf = pd.read_csv("Thesis_Ishory_Dai_Modified.csv")
ruf.head()

# View the details of the dataset to check if data contains null values
ruf.info()

# View the statistical values
ruf.describe()

# Plot distribution plots with normal distribution curve
columns_to_plot = ruf.columns[:-1]  # Exclude 'Surgical_Steps'

# Calculate correlation matrix
correlation_matrix = ruf.corr()

# Ensure correct column name for 'Surgical_Steps'
surgical_steps_column = [col for col in ruf.columns if 'Surgical_Steps' in col][0]

# Display the correlation of each column with 'Surgical_Steps'
correlation_with_surgical_steps = correlation_matrix[surgical_steps_column].drop(surgical_steps_column)

# Create a DataFrame to display the correlation values
correlation_table = pd.DataFrame(correlation_with_surgical_steps).reset_index()
correlation_table.columns = ['Column', 'Correlation with Surgical_Steps']

# Display the correlation table
print(correlation_table)

# Categorize each column based on mean value and perform chi-square test
chi_square_results = []
mean_comparison_results = []

for col in columns_to_plot:
    # Categorize based on mean
    mean_val = ruf[col].mean()
    ruf[f'{col}_cat'] = np.where(ruf[col] > mean_val, 'Above Mean', 'Below Mean')

    # Create contingency table
    contingency_table = pd.crosstab(ruf[f'{col}_cat'], ruf[surgical_steps_column])

    # Perform chi-square test
    chi2, p, dof, ex = chi2_contingency(contingency_table)

    # Store the chi-square test results
    chi_square_results.append({
        'Column': col,
        'Chi2 Statistic': chi2,
        'p-value': p,
        'Degrees of Freedom': dof
    })

    # Mean comparison results
    above_mean = ruf[ruf[col] > mean_val]
    below_mean = ruf[ruf[col] <= mean_val]

    above_mean_count = len(above_mean)
    below_mean_count = len(below_mean)

    above_mean_steps = above_mean[surgical_steps_column].value_counts().to_dict()
    below_mean_steps = below_mean[surgical_steps_column].value_counts().to_dict()

    mean_comparison_results.append({
        'Column': col,
        'Mean': mean_val,
        'Above Mean Count': above_mean_count,
        'Below Mean Count': below_mean_count,
        'Above Mean Surgical Steps': above_mean_steps,
        'Below Mean Surgical Steps': below_mean_steps
    })

# Convert results to DataFrame
chi_square_df = pd.DataFrame(chi_square_results)

# Display the chi-square test results
print("Chi-Square Test Results:")
print(chi_square_df)

# Restructure the DataFrame for better readability
structured_results = []

for row in mean_comparison_results:
    structured_results.append({
        'Column': row['Column'],
        'Mean': row['Mean'],
        'AM_Count': row['Above Mean Count'],
        'BM_Count': row['Below Mean Count'],
        'AMSS_1': row['Above Mean Surgical Steps'].get(1, 0),
        'AMSS_2': row['Above Mean Surgical Steps'].get(2, 0),
        'BMSS_1': row['Below Mean Surgical Steps'].get(1, 0),
        'BMSS_2': row['Below Mean Surgical Steps'].get(2, 0),
    })

# Convert structured results to DataFrame
structured_df = pd.DataFrame(structured_results)

# Display the restructured mean comparison results
print("Mean Comparison Results:")
print(structured_df.to_string(index=False))

# Strip extra spaces from column names in structured_df
structured_df.columns = structured_df.columns.str.strip()

# Verify the columns again
print("Columns in structured_df after cleaning:")
print(structured_df['Column'].unique())

# Define the confusion matrix mapping
confusion_matrix_mapping = {
    'Distraction_defect_MRU': {'TP': 'AMSS_2', 'FP': 'AMSS_1', 'TN': 'BMSS_1', 'FN': 'BMSS_2'},
    'distraction_defect_RGU': {'TP': 'AMSS_2', 'FP': 'AMSS_1', 'TN': 'BMSS_1', 'FN': 'BMSS_2'},
    'PVD_MRU': {'TP': 'BMSS_2', 'FP': 'BMSS_1', 'TN': 'AMSS_1', 'FN': 'AMSS_2'},
    'RUMD_MRU': {'TP': 'BMSS_1', 'FP': 'AMSS_2', 'TN': 'AMSS_1', 'FN': 'BMSS_2'},
    'PUL_MRU': {'TP': 'AMSS_2', 'FP': 'BMSS_2', 'TN': 'BMSS_1', 'FN': 'AMSS_1'},
    'Gap_index_MRU ': {'TP': 'AMSS_1', 'FP': 'AMSS_2', 'TN': 'BMSS_2', 'FN': 'BMSS_1'},
}

# Construct the confusion matrix
confusion_matrix = {}
for column, mapping in confusion_matrix_mapping.items():
    if column in structured_df['Column'].values:
        confusion_matrix[column] = {key: structured_df.loc[structured_df['Column'] == column, value].values[0]
                                    for key, value in mapping.items()}

# Check the confusion matrix for completeness
print("Confusion Matrix:")
for column, values in confusion_matrix.items():
    print(f"{column}: {values}")

# Calculate ROC curve and sensitivity
roc_results = []
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

for i, (column, values) in enumerate(confusion_matrix.items()):
    TP, FP, TN, FN = values['TP'], values['FP'], values['TN'], values['FN']

    # Sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Compute Youden Index
    youden_index = sensitivity + specificity - 1

    # Create ROC curve points
    roc_x = [0, 1 - specificity, 1]
    roc_y = [0, sensitivity, 1]

    # Calculate AUC
    roc_auc = auc(roc_x, roc_y)

    roc_results.append({
        'Column': column,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Youden Index': youden_index,
        'AUC': roc_auc
    })

# Convert ROC results to DataFrame
roc_df = pd.DataFrame(roc_results)

# Display the ROC results
print("ROC Results:")
print(roc_df.to_string(index=False))

# Define colors for plotting
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# Plot Combined ROC Curves
plt.figure(figsize=(10, 8))

for i, row in enumerate(roc_results):
    column = row['Column']
    roc_x = [0, 1 - row['Specificity'], 1]
    roc_y = [0, row['Sensitivity'], 1]
    plt.plot(roc_x, roc_y, color=colors[i % len(colors)], marker='o',
             label=f'ROC curve for {column} (Sensitivity={row["Sensitivity"]:.2f}, 1-Specificity={1 - row["Specificity"]:.2f}, AUC={row["AUC"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Combined ROC Curves')
plt.legend(loc="lower right")
plt.show()

# Plot Separate ROC Curves
for i, row in enumerate(roc_results):
    column = row['Column']
    roc_x = [0, 1 - row['Specificity'], 1]
    roc_y = [0, row['Sensitivity'], 1]

    plt.figure(figsize=(8, 6))
    plt.plot(roc_x, roc_y, color=colors[i % len(colors)], marker='o',
             label=f'ROC curve for {column} (Sensitivity={row["Sensitivity"]:.2f}, 1-Specificity={1 - row["Specificity"]:.2f}, AUC={row["AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'ROC Curve for {column}')
    plt.legend(loc="lower right")
    plt.show()

# Import the necessary library for the Mann-Whitney U test
from scipy.stats import mannwhitneyu

# Perform Mann-Whitney U test for each column against the 'Surgical_Steps' column
mann_whitney_results = []

for col in columns_to_plot:
    group1 = ruf[ruf[surgical_steps_column] == 1][col]
    group2 = ruf[ruf[surgical_steps_column] == 2][col]

    # Perform the Mann-Whitney U test
    stat, p_value = mannwhitneyu(group1, group2)

    # Store the results
    mann_whitney_results.append({
        'Column': col,
        'U Statistic': stat,
        'p-value': p_value
    })

# Convert results to DataFrame
mann_whitney_df = pd.DataFrame(mann_whitney_results)

# Display the Mann-Whitney U test results
print("Mann-Whitney U Test Results:")
print(mann_whitney_df.to_string(index=False))

# Import additional libraries
from scipy.stats import ttest_ind, spearmanr
from sklearn.metrics import roc_curve, roc_auc_score

# Two-sample t-test
ttest_results = []
for col in columns_to_plot:
    group0 = ruf[ruf[surgical_steps_column] == 1][col]
    group1 = ruf[ruf[surgical_steps_column] == 2][col]

    # Perform the t-test
    t_stat, p_value = ttest_ind(group0, group1, equal_var=True)

    ttest_results.append({
        'Column': col,
        'T-Statistic': t_stat,
        'p-value': p_value
    })

# Convert t-test results to DataFrame
ttest_df = pd.DataFrame(ttest_results)

# Display the t-test results
print("T-Test Results:")
print(ttest_df.to_string(index=False))

# Spearman's Rank Correlation
spearman_results = []

for col in columns_to_plot:
    rho, p_value = spearmanr(ruf[col], ruf[surgical_steps_column])
    spearman_results.append({
        'Column': col,
        'Spearman rho': rho,
        'p-value': p_value
    })

# Convert Spearman results to DataFrame
spearman_df = pd.DataFrame(spearman_results)

# Display the Spearman results
print("Spearman's Rank Correlation Results:")
print(spearman_df.to_string(index=False))

# Optimal Cutpoint Estimation using Youden Index
cutpoint_results = []

for col in columns_to_plot:
    y_true = ruf[surgical_steps_column]
    y_scores = ruf[col]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=2)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Compute Youden Index
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]

    cutpoint_results.append({
        'Column': col,
        'Optimal Cutpoint': optimal_threshold,
        'Youden Index': youden_index[optimal_idx],
        'Sensitivity at Cutpoint': sensitivity,
        'Specificity at Cutpoint': specificity,
        'AUC': roc_auc
    })

# Convert cutpoint results to DataFrame
cutpoint_df = pd.DataFrame(cutpoint_results)

# Display the cutpoint results
print("Optimal Cutpoint Estimation Results:")
print(cutpoint_df.to_string(index=False))

# to fix inf values
cutpoint_results = []

for col in columns_to_plot:
    y_true = ruf[surgical_steps_column]
    y_scores = ruf[col]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=2)

    # Filter out inf values from thresholds
    valid_indices = np.isfinite(thresholds)  # Boolean array indicating valid thresholds
    fpr_valid = fpr[valid_indices]
    tpr_valid = tpr[valid_indices]
    thresholds_valid = thresholds[valid_indices]

    # If no valid thresholds, handle gracefully
    if len(thresholds_valid) == 0:
        cutpoint_results.append({
            'Column': col,
            'Optimal Cutpoint': np.nan,
            'Youden Index': np.nan,
            'Sensitivity at Cutpoint': np.nan,
            'Specificity at Cutpoint': np.nan,
            'AUC': np.nan
        })
        continue

    # Compute Youden Index
    youden_index = tpr_valid - fpr_valid
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds_valid[optimal_idx]
    sensitivity = tpr_valid[optimal_idx]
    specificity = 1 - fpr_valid[optimal_idx]
    roc_auc = roc_auc_score(y_true, y_scores)

    cutpoint_results.append({
        'Column': col,
        'Optimal Cutpoint': optimal_threshold,
        'Youden Index': youden_index[optimal_idx],
        'Sensitivity at Cutpoint': sensitivity,
        'Specificity at Cutpoint': specificity,
        'AUC': roc_auc
    })

# Convert cutpoint results to DataFrame
cutpoint_df = pd.DataFrame(cutpoint_results)

# Display the cutpoint results
print("Optimal Cutpoint Estimation Results:")
print(cutpoint_df.to_string(index=False))