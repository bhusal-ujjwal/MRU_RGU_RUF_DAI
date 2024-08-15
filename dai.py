# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2_contingency
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

plt.figure(figsize=(10, 40))

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(len(columns_to_plot), 1, i)  # Create a vertical layout
    sns.distplot(ruf[col], fit=norm, kde=False)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency (occurrences)')

plt.tight_layout()
plt.show()

# Calculate correlation matrix
correlation_matrix = ruf.corr()

# Plot heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

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

# Define the mapping for the confusion matrix
confusion_matrix_mapping = {
    'Distraction_defect_MRU': {'TP': 'AMSS_2', 'FP': 'AMSS_1', 'TN': 'BMSS_1', 'FN': 'BMSS_2'},
    'distraction_defect_RGU': {'TP': 'AMSS_2', 'FP': 'AMSS_1', 'TN': 'BMSS_1', 'FN': 'BMSS_2'},
    'PVD_MRU': {'TP': 'BMSS_2', 'FP': 'BMSS_1', 'TN': 'AMSS_1', 'FN': 'AMSS_2'},
    'RUMD_MRU': {'TP': 'BMSS_1', 'FP': 'AMSS_2', 'TN': 'AMSS_1', 'FN': 'BMSS_2'},
    'PUL_MRU': {'TP': 'AMSS_2', 'FP': 'BMSS_2', 'TN': 'BMSS_1', 'FN': 'AMSS_1'},
    'Gap_index_MRU': {'TP': 'AMSS_1', 'FP': 'AMSS_2', 'TN': 'BMSS_2', 'FN': 'BMSS_1'},
}

# Construct the confusion matrix
confusion_matrix = {}
for column, mapping in confusion_matrix_mapping.items():
    if column in structured_df['Column'].values:
        confusion_matrix[column] = {key: structured_df.loc[structured_df['Column'] == column, value].values[0]
                                    for key, value in mapping.items()}

# Calculate ROC curve and sensitivity
roc_results = []

for column, values in confusion_matrix.items():
    TP, FP, TN, FN = values['TP'], values['FP'], values['TN'], values['FN']

    # Sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    roc_results.append({
        'Column': column,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    })

# Convert ROC results to DataFrame
roc_df = pd.DataFrame(roc_results)

# Display the ROC results
print("ROC Results:")
print(roc_df.to_string(index=False))

# Plot ROC curves
plt.figure(figsize=(10, 8))

for row in roc_results:
    column = row['Column']
    TP, FP, TN, FN = row['TP'], row['FP'], row['TN'], row['FN']

    # Create labels and scores for ROC curve
    y_true = [1] * TP + [0] * FN + [0] * FP + [1] * TN
    y_scores = [1] * (TP + FN) + [0] * (FP + TN)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'ROC curve for {column} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

# Plotting curves separately to check if curves are overlapping and not visible
# Plot ROC curve for Distraction_defect_MRU
plt.figure(figsize=(8, 6))
TP, FP, TN, FN = roc_results[0]['TP'], roc_results[0]['FP'], roc_results[0]['TN'], roc_results[0]['FN']
y_true = [1] * TP + [0] * FN + [0] * FP + [1] * TN
y_scores = [1] * (TP + FN) + [0] * (FP + TN)
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f'ROC curve for Distraction_defect_MRU (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Distraction_defect_MRU')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve for distraction_defect_RGU
plt.figure(figsize=(8, 6))
TP, FP, TN, FN = roc_results[1]['TP'], roc_results[1]['FP'], roc_results[1]['TN'], roc_results[1]['FN']
y_true = [1] * TP + [0] * FN + [0] * FP + [1] * TN
y_scores = [1] * (TP + FN) + [0] * (FP + TN)
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='orange', label=f'ROC curve for distraction_defect_RGU (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for distraction_defect_RGU')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve for each column
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

for i, row in enumerate(roc_results):
    column = row['Column']
    TP, FP, TN, FN = row['TP'], row['FP'], row['TN'], row['FN']

    # Create labels and scores for ROC curve
    y_true = [1] * TP + [0] * FN + [0] * FP + [1] * TN
    y_scores = [1] * (TP + FN) + [0] * (FP + TN)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current column
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=colors[i % len(colors)], label=f'ROC curve for {column} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
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

# Exclude 'distraction_defect_RGU' from the plot
columns_to_plot_for_graph = [col for col in columns_to_plot if col != 'distraction_defect_RGU']
mann_whitney_df_for_graph = mann_whitney_df[mann_whitney_df['Column'].isin(columns_to_plot_for_graph)]

# Plot p-values from Mann-Whitney U test
plt.figure(figsize=(10, 6))
plt.barh(mann_whitney_df_for_graph['Column'], mann_whitney_df_for_graph['p-value'], color='skyblue')
plt.xlabel('p-value')
plt.ylabel('Column')
plt.title('Mann-Whitney U Test p-values for Each Column')
plt.axvline(x=0.05, color='r', linestyle='--', linewidth=1.5, label='Significance Level (0.05)')
plt.legend()
plt.gca().invert_yaxis()
plt.show()

# # Plot p-values from Mann-Whitney U test
# plt.figure(figsize=(10, 6))
# plt.barh(mann_whitney_df['Column'], mann_whitney_df['p-value'], color='skyblue')
# plt.xlabel('p-value')
# plt.ylabel('Column')
# plt.title('Mann-Whitney U Test p-values for Each Column')
# plt.axvline(x=0.05, color='r', linestyle='--', linewidth=1.5, label='Significance Level (0.05)')
# plt.legend()
# plt.gca().invert_yaxis()  # Invert y-axis to have higher p-values at the top
# plt.show()

# Upload csv file to colab for using it
upload_file = files.upload()

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Extract the feature (X) and target variable (y)
X = ruf2[['Gap_length_surgery']]  # feature
y = ruf2['Distraction_defect_MRU']  # target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression object
model = LinearRegression()

# Fit the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients:', model.coef_)
# The mean squared error
print('Mean squared error:', mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(X_test, y_test,  color='black')  # Actual data points
plt.plot(X_test, y_pred, color='blue', linewidth=3)  # Predicted values

plt.title('Actual vs Predicted')
plt.xlabel('Gap_length_surgery')
plt.ylabel('Distraction_defect_MRU')

plt.show()

# Calculate Absolute Error
ruf2['Absolute_Error'] = (ruf2['Gap_length_surgery'] - ruf2['Distraction_defect_MRU']).abs()

# Calculate Relative Error
ruf2['Relative_Error'] = ruf2['Absolute_Error'] / ruf2['Gap_length_surgery']

# Calculate Error Percentage
ruf2['Error_Percentage'] = ruf2['Relative_Error'] * 100

# Display the calculated errors
print(ruf2[['Gap_length_surgery', 'Distraction_defect_MRU', 'Absolute_Error', 'Relative_Error', 'Error_Percentage']])

# Statistical summary of errors
error_summary = ruf2[['Absolute_Error', 'Relative_Error', 'Error_Percentage']].describe()
print("\nError Summary:")
print(error_summary)

# Upload csv file to colab for using it
upload_file = files.upload()

# Use the uploaded file in colab
ruf3 = pd.read_csv("third copy.csv")
ruf3.head()

# Check column names to ensure 'Surgical_Steps' is correctly referenced
print("Columns in DataFrame:", ruf3.columns)

# Strip any leading/trailing spaces from the column names to avoid referencing issues
ruf3.columns = ruf3.columns.str.strip()

# Separate the data based on the surgical_steps value
surgical_steps_1 = ruf3[ruf3['Surgical_Steps'] == 1]
surgical_steps_2 = ruf3[ruf3['Surgical_Steps'] == 2]

# Remove 'Surgical_Steps' column from calculation
surgical_steps_1 = surgical_steps_1.drop(columns='Surgical_Steps')
surgical_steps_2 = surgical_steps_2.drop(columns='Surgical_Steps')

# Use describe function to get statistical summary for both groups
describe_steps_1 = surgical_steps_1.describe()
describe_steps_2 = surgical_steps_2.describe()

# Display the results
print("Statistical Summary for Surgical Steps 1:")
print(describe_steps_1)

print("\nStatistical Summary for Surgical Steps 2:")
print(describe_steps_2)