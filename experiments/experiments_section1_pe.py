from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm
import numpy as np
from dataloader.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer
from dataloader.DataAugmenter import DataAugmenter
import pymzml
import os
from dataloader.SpectrumObject import SpectrumObject
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import itertools  # For combinatory testing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Global constants for dataset structure
WEEKS = ['Semana 1', 'Semana 2', 'Semana 3']
CLASSES = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']
MEDIA = ['Chx', 'Brx', 'Clx', 'SCx', 'GU']

# Initialize lists for data
Y_train, Y_test = [], []
baseline_samples, test_samples = [], []

semanas = ['Semana 1', 'Semana 2', 'Semana 3']
clases = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']
medios = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU']

Y_train = []
baseline_samples = []
test_samples = []
Y_test = []
baseline_media_label = []
baseline_semana_label = []
baseline_extraction_label = []
baseline_id_label = []
test_media_label = []
test_semana_label = []
test_extraction_label = []
test_id_label = []

for medio in medios:
    for semana in semanas:
        for clase in clases:
            ruta = f'/export/data_ml4ds/bacteria_id/C_diff/Reproducibilidad/ClostiRepro/ClostriRepro/Reproducibilidad No extracción/{medio}/{semana}/{clase}'
            ruta_pe = f'/export/data_ml4ds/bacteria_id/C_diff/Reproducibilidad/ClostiRepro/ClostriRepro/Reproducibilidad Extracción/{medio}/{semana}/{clase}' 
            if os.path.exists(ruta):
                for f in os.listdir(ruta):
                    ruta_f = os.path.join(ruta, f)
                    if medio == 'Medio Ch'and semana == 'Semana 1':
                        baseline_id_label.append(f.split('_')[0])
                        baseline_media_label.append(medio)
                        baseline_semana_label.append(semana)
                        baseline_extraction_label.append(0)
                        if 'mzml' in ruta_f:
                            run = pymzml.run.Reader(ruta_f)
                            spectro = [r for r in run]
                            s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                            baseline_samples.append(s)
                            Y_train.append(clase)
                        else: 
                            carpetas = [subf for subf in os.listdir(ruta_f)]
                            if carpetas:
                                ruta_f = os.path.join(ruta_f, carpetas[0])
                                # Buscar archivos 'fid' y 'acqu' en las subcarpetas
                                fid_files = glob(os.path.join(ruta_f, '*', '1SLin', 'fid'))
                                acqu_files = glob(os.path.join(ruta_f, '*', '1SLin', 'acqu'))

                                ruta_fid = fid_files[0]
                                ruta_acqu = acqu_files[0]
                                s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                                baseline_samples.append(s)
                                Y_train.append(clase)       
                    
                    else:
                        test_id_label.append(f.split('_')[0])
                        test_media_label.append(medio)
                        test_semana_label.append(semana)
                        # Si el archivo es un .mzml
                        if 'mzml' in ruta_f:
                            run = pymzml.run.Reader(ruta_f)
                            
                            spectro = [r for r in run]
                            s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                            test_samples.append(s)
                            Y_test.append(clase)
                                    
                                    
                        else: 
                            carpetas = [subf for subf in os.listdir(ruta_f)]
                            
                            ruta_f = os.path.join(ruta, f, carpetas[0])
                            fid_files = glob(os.path.join(ruta_f, '*', '1SLin', 'fid'))
                            acqu_files = glob(os.path.join(ruta_f, '*', '1SLin', 'acqu'))

                            ruta_fid = fid_files[0]
                            ruta_acqu = acqu_files[0]
                            s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                            test_samples.append(s)
                            Y_test.append(clase)
                for f in os.listdir(ruta_pe):
                    ruta_f = os.path.join(ruta, f)
                    test_id_label.append(f.split('_')[0])
                    test_media_label.append(medio)
                    test_semana_label.append(semana)
                    # Si el archivo es un .mzml
                    if 'mzml' in ruta_f:
                        run = pymzml.run.Reader(ruta_f)
                        spectro = [r for r in run]
                        s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                        test_samples.append(s)
                        Y_test.append(clase)
                    else:
                        carpetas = [subf for subf in os.listdir(ruta_f)]
                        ruta_f = os.path.join(ruta, f, carpetas[0])
                        fid_files = glob(os.path.join(ruta_f, '*', '1SLin', 'fid'))
                        acqu_files = glob(os.path.join(ruta_f, '*', '1SLin', 'acqu'))
                        ruta_fid = fid_files[0]
                        ruta_acqu = acqu_files[0]
                        s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                        test_samples.append(s)
                        Y_test.append(clase)
# Encode labels
label_mapping = {label: idx for idx, label in enumerate(CLASSES)}
Y_train = np.array([label_mapping[label] for label in Y_train])
Y_test = np.array([label_mapping[label] for label in Y_test])

# Preprocessing pipeline
binning_step = 3
preprocess_pipeline = SequentialPreprocessor(
    VarStabilizer(method="sqrt"),
    Smoother(halfwindow=10),
    BaselineCorrecter(method="SNIP", snip_n_iter=10),
    Normalizer(sum=1),
    Trimmer(),
    Binner(step=binning_step),
)

# Preprocess data
print("Preprocessing data...")
baseline_samples = [preprocess_pipeline(s) for s in tqdm(baseline_samples, desc="Baseline samples")]
test_samples = [preprocess_pipeline(s) for s in tqdm(test_samples, desc="Test samples")]


test_cases = [
    {"Media": "Medio Ch", "Week": "Semana 1"},
    {"Media": "Medio Ch", "Week": "Semana 2"},
    {"Media": "Medio Ch", "Week": "Semana 3"},
    {"Media": "Medio Br", "Week": "Semana 1"},
    {"Media": "Medio Cl", "Week": "Semana 1"},
    {"Media": "Medio Sc", "Week": "Semana 1"},
    {"Media": "GU", "Week": "Semana 1"},
]

# Model selection flags
USE_RF = True
USE_SVM_LINEAR = True
USE_SVM_RBF = True
USE_KNN = True
USE_LGBM = True

# Predefine parameter grids
rf_grid = {"n_estimators": [100, 200, 300]}
svm_linear_grid = {"C": [0.01, 0.1, 1, 10, 100]}
svm_rbf_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
}
knn_grid = {"n_neighbors": list(range(3, 21))}
lgbm_grid = {"n_estimators": [100, 200, 300], "num_leaves": [31, 50, 100]}

# Final results storage
all_results = []

# Function to train and evaluate models
def train_and_evaluate(model_name, model, param_grid, X_train, Y_train_loo, X_test, Y_test):
    grid_model = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc_ovr", n_jobs=-1, verbose=0)
    grid_model.fit(X_train, Y_train_loo)
    best_model = grid_model.best_estimator_
    predictions = best_model.predict_proba(X_test)
    auc = roc_auc_score(Y_test, predictions, multi_class="ovr")
    acc = accuracy_score(Y_test, np.argmax(predictions, axis=1))
    return auc, acc

# Loop over test cases
for test_idx, test_case in enumerate(test_cases, start=1):
    selected_media = test_case["Media"]
    selected_week = test_case["Week"]

    print(f"\nRunning Test {test_idx}: Media={selected_media}, Week={selected_week}...")

    # Filter test samples
    filtered_test_indices = [
        i for i, (media, week) in enumerate(zip(test_media_label, test_semana_label))
        if media == selected_media and week == selected_week
    ]
    filtered_test_samples = [test_samples[i] for i in filtered_test_indices]
    filtered_Y_test = Y_test[filtered_test_indices]
    filtered_test_id_labels = [test_id_label[i] for i in filtered_test_indices]

    # Skip if no test samples match
    if len(filtered_test_samples) == 0:
        print(f"Test {test_idx} skipped: No matching test samples.")
        continue

    # Results for this test case
    test_results = {"Test": f"Test {test_idx}"}

    # Train and evaluate each model
    for model_name, use_model, model, param_grid in [
        ("RF", USE_RF, RandomForestClassifier(random_state=42), rf_grid),
        ("SVM_Linear", USE_SVM_LINEAR, SVC(kernel="linear", probability=True, random_state=42), svm_linear_grid),
        ("SVM_RBF", USE_SVM_RBF, SVC(kernel="rbf", probability=True, random_state=42), svm_rbf_grid),
        ("KNN", USE_KNN, KNeighborsClassifier(), knn_grid),
        ("LGBM", USE_LGBM, LGBMClassifier(random_state=42), lgbm_grid),
    ]:
        if not use_model:
            continue

        print(f"Training {model_name} for Test {test_idx}...")
        auc_scores, acc_scores = [], []

        for idx, left_out_id in tqdm(enumerate(baseline_id_label), total=len(baseline_id_label), desc=f"{model_name} LOO Progress"):
            # Filter training data
            train_samples = [
                baseline_samples[i] for i in range(len(baseline_samples)) if baseline_id_label[i] != left_out_id
            ]
            Y_train_loo = [
                Y_train[i] for i in range(len(baseline_samples)) if baseline_id_label[i] != left_out_id
            ]

            # Prepare test data
            loo_test_indices = [
                i for i, test_id in enumerate(filtered_test_id_labels) if test_id == left_out_id
            ]
            X_test_loo = [s.intensity for i, s in enumerate(filtered_test_samples) if i not in loo_test_indices]
            Y_test_loo = [
                filtered_Y_test[i] for i in range(len(filtered_test_samples)) if i not in loo_test_indices
            ]

            # Skip if no valid test samples remain
            if len(X_test_loo) == 0:
                continue

            # Prepare training data
            X_train = [s.intensity for s in train_samples]

            # Train and evaluate the model
            auc, acc = train_and_evaluate(model_name, model, param_grid, X_train, Y_train_loo, X_test_loo, Y_test_loo)
            auc_scores.append(auc)
            acc_scores.append(acc)

        # Compute final metrics
        final_auc = np.mean(auc_scores) if auc_scores else 0
        final_acc = np.mean(acc_scores) if acc_scores else 0

        # Add to test results
        test_results[f"{model_name}_AUC"] = final_auc
        test_results[f"{model_name}_Accuracy"] = final_acc

        print(f"{model_name} - AUC: {final_auc:.4f}, Accuracy: {final_acc:.4f}")

    # Append test results
    all_results.append(test_results)

# Save all results to CSV
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv("results/test_case_results.csv", index=False)

# Print summary of all test cases
print("\nFinal Results Summary:")
print(all_results_df)