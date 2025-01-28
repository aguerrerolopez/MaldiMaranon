import pandas as pd

# Load the CSV file
csv_file = "/export/usuarios01/alexjorguer/Datos/MaldiMaranon/results/test_case_results_table3.csv"
results_df = pd.read_csv(csv_file)

# Define the mapping of tests to LaTeX labels
test_cases = [
    {"Test": "Test 0", "Media": "Medio Ch", "Week": "Semana 1"},
    {"Test": "Test 2", "Media": "Medio Ch", "Week": "Semana 2"},
    {"Test": "Test 3", "Media": "Medio Ch", "Week": "Semana 3"},
    {"Test": "Test 4", "Media": "Medio Br", "Week": "Semana 1"},
    {"Test": "Test 5", "Media": "Medio Cl", "Week": "Semana 1"},
    {"Test": "Test 6", "Media": "Medio Sc", "Week": "Semana 1"},
    {"Test": "Test 7", "Media": "GU", "Week": "Semana 1"},
]

# Start building the LaTeX table
latex_table = """
\begin{table*}
    \centering
    \adjustbox{max width=\textwidth}{
    \begin{tabular}{|c|c|c|c|c|c|c|c|c|} \cline{4-9}
    \multicolumn{3}{c|}{}& \textbf{RF} & \textbf{SVM + RBF} & \textbf{SVM + Linear} & \textbf{KNN} & \textbf{LGBM} & \textbf{Mean} \\ \hline
"""

# Define rows for test cases
for idx, test_case in enumerate(test_cases):
    test_label = test_case["Test"]
    media = test_case["Media"]
    week = test_case["Week"]


    # Filter results for the current test (the test_label must be contained)
    test_results = results_df[results_df["Test"].str.contains(test_label)]
    if test_results.empty:
        # Leave blank for PE and 24H PE
        row = f"& & {media} & {week} & & & & & \\ \cline{2-9}\n"
    else:
        # Extract metrics
        rf_auc = test_results["RF_AUC"].values[0]
        rf_acc = test_results["RF_Accuracy"].values[0]
        svm_rbf_auc = test_results["SVM_RBF_AUC"].values[0]
        svm_rbf_acc = test_results["SVM_RBF_Accuracy"].values[0]
        svm_linear_auc = test_results["SVM_Linear_AUC"].values[0]
        svm_linear_acc = test_results["SVM_Linear_Accuracy"].values[0]
        knn_auc = test_results["KNN_AUC"].values[0]
        knn_acc = test_results["KNN_Accuracy"].values[0]
        lgbm_auc = test_results["LGBM_AUC"].values[0]
        lgbm_acc = test_results["LGBM_Accuracy"].values[0]

        # Compute mean values for AUC and Accuracy
        mean_auc = (rf_auc + svm_rbf_auc + svm_linear_auc + knn_auc + lgbm_auc) / 5
        mean_acc = (rf_acc + svm_rbf_acc + svm_linear_acc + knn_acc + lgbm_acc) / 5

        # Format the row
        row = (f"& & {media} & {week} & {rf_auc:.2f} ({rf_acc:.2f}) & {svm_rbf_auc:.2f} ({svm_rbf_acc:.2f}) & "
               f"{svm_linear_auc:.2f} ({svm_linear_acc:.2f}) & {knn_auc:.2f} ({knn_acc:.2f}) & "
               f"{lgbm_auc:.2f} ({lgbm_acc:.2f}) & {mean_auc:.2f} ({mean_acc:.2f}) \\ \cline{{2-9}}\n")
    
    latex_table += row

# Calculate column means for AUC and Accuracy
mean_rf_auc = results_df["RF_AUC"].mean()
mean_rf_acc = results_df["RF_Accuracy"].mean()
mean_svm_rbf_auc = results_df["SVM_RBF_AUC"].mean()
mean_svm_rbf_acc = results_df["SVM_RBF_Accuracy"].mean()
mean_svm_linear_auc = results_df["SVM_Linear_AUC"].mean()
mean_svm_linear_acc = results_df["SVM_Linear_Accuracy"].mean()
mean_knn_auc = results_df["KNN_AUC"].mean()
mean_knn_acc = results_df["KNN_Accuracy"].mean()
mean_lgbm_auc = results_df["LGBM_AUC"].mean()
mean_lgbm_acc = results_df["LGBM_Accuracy"].mean()

# Compute overall mean for AUC and Accuracy
overall_mean_auc = (mean_rf_auc + mean_svm_rbf_auc + mean_svm_linear_auc + mean_knn_auc + mean_lgbm_auc) / 5
overall_mean_acc = (mean_rf_acc + mean_svm_rbf_acc + mean_svm_linear_acc + mean_knn_acc + mean_lgbm_acc) / 5

# Add final row for means
mean_row = (f"& & \textbf{{Mean}} & & {mean_rf_auc:.2f} ({mean_rf_acc:.2f}) & {mean_svm_rbf_auc:.2f} ({mean_svm_rbf_acc:.2f}) & "
            f"{mean_svm_linear_auc:.2f} ({mean_svm_linear_acc:.2f}) & {mean_knn_auc:.2f} ({mean_knn_acc:.2f}) & "
            f"{mean_lgbm_auc:.2f} ({mean_lgbm_acc:.2f}) & {overall_mean_auc:.2f} ({overall_mean_acc:.2f}) \\ \hline\n")

latex_table += mean_row

# Add a final row with std
std_row = (f"& & \textbf{{Std}} & & {results_df['RF_AUC'].std():.2f} ({results_df['RF_Accuracy'].std():.2f}) & "
           f"{results_df['SVM_RBF_AUC'].std():.2f} ({results_df['SVM_RBF_Accuracy'].std():.2f}) & "
           f"{results_df['SVM_Linear_AUC'].std():.2f} ({results_df['SVM_Linear_Accuracy'].std():.2f}) & "
           f"{results_df['KNN_AUC'].std():.2f} ({results_df['KNN_Accuracy'].std():.2f}) & "
           f"{results_df['LGBM_AUC'].std():.2f} ({results_df['LGBM_Accuracy'].std():.2f}) & \\ \hline\n")
latex_table += std_row

# Close the table
latex_table += """
    \end{tabular}
    }
    \caption{Performance metrics for different media and weeks.}
    \label{tab:media_week_performance}
\end{table*}
"""

# Print the LaTeX table
print(latex_table)

