#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm
import numpy as np
from dataloader.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer
import pymzml
import os
from dataloader.SpectrumObject import SpectrumObject
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

###########################################
# Global constants and configuration
###########################################
# List of classes for encoding
CLASSES = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']

# For this new script, the training (baseline) data is defined as:
#    - Week (Semana) = "Semana 1"
#    - Media = "Medio Sc"
# All other combinations are considered for testing.

###########################################
# Data Lists and Labels
###########################################
baseline_samples = []
baseline_id_label = []
baseline_media_label = []
baseline_semana_label = []
Y_train = []

test_samples = []
test_id_label = []
test_media_label = []
test_semana_label = []
Y_test = []

###########################################
# Directories and parameters for loading
###########################################
# Define weeks, classes and medios in your directory structure.
# Note: Your folder names may need to be adjusted as per your system.
semanas = ['Semana 1', 'Semana 2', 'Semana 3']
clases = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']
medios = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU']

# Define a base path (adjust if needed)
base_path = '/export/data_ml4ds/bacteria_id/C_diff/Reproducibilidad/ClostiRepro/ClostriRepro/Reproducibilidad No extracción'

###########################################
# Load data from directory
###########################################
for medio in medios:
    for semana in semanas:
        for clase in clases:
            ruta = f"{base_path}/{medio}/{semana}/{clase}"
            if os.path.exists(ruta):
                for f in os.listdir(ruta):
                    ruta_f = os.path.join(ruta, f)
                    # For training (baseline) data: load only if medio=="Medio Sc" and semana=="Semana 1"
                    if medio == "Medio Sc" and semana == "Semana 1":
                        baseline_id_label.append(f.split('_')[0])
                        baseline_media_label.append(medio)
                        baseline_semana_label.append(semana)
                        if 'mzml' in ruta_f:
                            run = pymzml.run.Reader(ruta_f)
                            spectro = [r for r in run]
                            s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                            baseline_samples.append(s)
                            Y_train.append(clase)
                        else:
                            carpetas = [subf for subf in os.listdir(ruta_f)]
                            if carpetas:
                                ruta_sub = os.path.join(ruta_f, carpetas[0])
                                fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
                                acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
                                if fid_files and acqu_files:
                                    ruta_fid = fid_files[0]
                                    ruta_acqu = acqu_files[0]
                                    s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                                    baseline_samples.append(s)
                                    Y_train.append(clase)
                    # All other cases will be used as test data.
                    else:
                        test_id_label.append(f.split('_')[0])
                        test_media_label.append(medio)
                        test_semana_label.append(semana)
                        if 'mzml' in ruta_f:
                            run = pymzml.run.Reader(ruta_f)
                            spectro = [r for r in run]
                            s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                            test_samples.append(s)
                            Y_test.append(clase)
                        else:
                            carpetas = [subf for subf in os.listdir(ruta_f)]
                            if carpetas:
                                ruta_sub = os.path.join(ruta_f, carpetas[0])
                                fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
                                acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
                                if fid_files and acqu_files:
                                    ruta_fid = fid_files[0]
                                    ruta_acqu = acqu_files[0]
                                    s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                                    test_samples.append(s)
                                    Y_test.append(clase)

# Encode labels using defined order in CLASSES
label_mapping = {label: idx for idx, label in enumerate(CLASSES)}
Y_train = np.array([label_mapping[label] for label in Y_train])
Y_test = np.array([label_mapping[label] for label in Y_test])

###########################################
# Preprocessing Pipeline
###########################################
binning_step = 3
preprocess_pipeline = SequentialPreprocessor(
    VarStabilizer(method="sqrt"),
    Smoother(halfwindow=10),
    BaselineCorrecter(method="SNIP", snip_n_iter=10),
    Normalizer(sum=1),
    Trimmer(),
    Binner(step=binning_step),
)

print("Preprocessing training (baseline) data...")
baseline_samples = [preprocess_pipeline(s) for s in tqdm(baseline_samples, desc="Baseline Samples")]

print("Preprocessing test data...")
test_samples = [preprocess_pipeline(s) for s in tqdm(test_samples, desc="Test Samples")]

###########################################
# Model Training and Leave-One-Out Helper Functions
###########################################
def get_predictions(model, param_grid, X_train, Y_train_loo, X_test):
    grid_model = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc_ovr", n_jobs=-1, verbose=0)
    grid_model.fit(X_train, Y_train_loo)
    best_model = grid_model.best_estimator_
    preds = best_model.predict_proba(X_test)
    return preds

###########################################
# Model Definitions, Flags and Parameter Grids
###########################################
USE_RF = True
USE_SVM_LINEAR = True
USE_SVM_RBF = True
USE_KNN = True
USE_LGBM = True

rf_grid = {"n_estimators": [100, 200, 300]}
svm_linear_grid = {"C": [0.01, 0.1, 1, 10, 100]}
svm_rbf_grid = {"C": [0.01, 0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}
knn_grid = {"n_neighbors": list(range(3, 21))}
lgbm_grid = {"n_estimators": [100, 200, 300], "num_leaves": [31, 50, 100]}

models_info = {
    "RF": (USE_RF, RandomForestClassifier(random_state=42), rf_grid),
    "SVM_Linear": (USE_SVM_LINEAR, SVC(kernel="linear", probability=True, random_state=42), svm_linear_grid),
    "SVM_RBF": (USE_SVM_RBF, SVC(kernel="rbf", probability=True, random_state=42), svm_rbf_grid),
    "KNN": (USE_KNN, KNeighborsClassifier(), knn_grid),
    "LGBM": (USE_LGBM, LGBMClassifier(random_state=42), lgbm_grid),
}

###########################################
# Test 0: Baseline Leave-One-Out (using training data)
###########################################
print("\nRunning Test 0: Baseline (LOO over training data from Semana 1, Medio Sc)...")
results_test0 = {}

for model_name, (use_model, model, param_grid) in models_info.items():
    if not use_model:
        continue
    print(f"\nTraining {model_name} for Test 0...")
    preds_all = []   # accumulate prediction probabilities
    ytrue_all = []   # accumulate corresponding true labels

    # Leave-One-Out on baseline samples (by unique IDs)
    for left_out_id in tqdm(np.unique(baseline_id_label), desc=f"{model_name} LOO Progress"):
        # Training: all baseline samples that do not match left_out_id
        X_train = [s.intensity for i, s in enumerate(baseline_samples) if baseline_id_label[i] != left_out_id]
        Y_train_loo = [Y_train[i] for i in range(len(baseline_samples)) if baseline_id_label[i] != left_out_id]

        # Testing: samples that match left_out_id
        X_test_loo = [s.intensity for i, s in enumerate(baseline_samples) if baseline_id_label[i] == left_out_id]
        Y_test_loo = [Y_train[i] for i in range(len(baseline_samples)) if baseline_id_label[i] == left_out_id]

        if len(X_test_loo) == 0:
            continue

        preds = get_predictions(model, param_grid, X_train, Y_train_loo, X_test_loo)
        preds_all.append(preds)
        ytrue_all.extend(Y_test_loo)
    
    if len(preds_all) > 0:
        preds_all = np.concatenate(preds_all, axis=0)
        if len(np.unique(ytrue_all)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(ytrue_all, preds_all, multi_class="ovr")
        acc = accuracy_score(ytrue_all, np.argmax(preds_all, axis=1))
    else:
        auc, acc = np.nan, np.nan

    results_test0[f"{model_name}_AUC"] = auc
    results_test0[f"{model_name}_Accuracy"] = acc
    print(f"{model_name} (Test 0) - AUC: {auc:.4f}, Accuracy: {acc:.4f}")

###########################################
# Test Cases
###########################################
test_cases = [
    {"Media": "Medio Sc", "Week": "Semana 1"},
    {"Media": "Medio Sc", "Week": "Semana 2"},
    {"Media": "Medio Sc", "Week": "Semana 3"},
    {"Media": "Medio Br", "Week": "Semana 1"},
    {"Media": "Medio Cl", "Week": "Semana 1"},
    {"Media": "Medio Ch", "Week": "Semana 1"},
    {"Media": "GU", "Week": "Semana 1"},
]

# Collect results starting with Test 0
all_results = [dict(Test="Test 0 (Baseline LOO)", **results_test0)]

for test_idx, test_case in enumerate(test_cases, start=1):
    selected_media = test_case["Media"]
    selected_week = test_case["Week"]
    print(f"\nRunning Test {test_idx}: Media={selected_media}, Week={selected_week}...")

    # Filter test samples based on media and week
    filtered_test_indices = [
        i for i, (media, week) in enumerate(zip(test_media_label, test_semana_label))
        if media == selected_media and week == selected_week
    ]
    filtered_test_samples = [test_samples[i] for i in filtered_test_indices]
    filtered_Y_test = [Y_test[i] for i in filtered_test_indices]
    filtered_test_id_labels = [test_id_label[i] for i in filtered_test_indices]

    if len(filtered_test_samples) == 0:
        print(f"Test {test_idx} skipped: No matching test samples.")
        continue

    results_test = {}
    # For each model, accumulate predictions and true labels over LOO iterations on baseline IDs.
    for model_name, (use_model, model, param_grid) in models_info.items():
        if not use_model:
            continue

        print(f"Training {model_name} for Test {test_idx}...")
        preds_all = []
        ytrue_all = []

        # Use the baseline IDs to create LOO splits for training
        for left_out_id in tqdm(np.unique(baseline_id_label), desc=f"{model_name} LOO Progress"):
            # Build training data from baseline (not equal to left_out_id)
            X_train = [s.intensity for i, s in enumerate(baseline_samples) if baseline_id_label[i] != left_out_id]
            Y_train_loo = [Y_train[i] for i in range(len(baseline_samples)) if baseline_id_label[i] != left_out_id]

            # For this iteration, from the filtered test data, remove samples with left_out_id.
            indices_for_test = [i for i, tid in enumerate(filtered_test_id_labels) if tid != left_out_id]
            if len(indices_for_test) == 0:
                continue

            X_test_loo = [filtered_test_samples[i].intensity for i in indices_for_test]
            Y_test_loo = [filtered_Y_test[i] for i in indices_for_test]

            preds = get_predictions(model, param_grid, X_train, Y_train_loo, X_test_loo)
            preds_all.append(preds)
            ytrue_all.extend(Y_test_loo)
        
        if len(preds_all) > 0:
            preds_all = np.concatenate(preds_all, axis=0)
            if len(np.unique(ytrue_all)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(ytrue_all, preds_all, multi_class="ovr")
            acc = accuracy_score(ytrue_all, np.argmax(preds_all, axis=1))
        else:
            auc, acc = np.nan, np.nan

        results_test[f"{model_name}_AUC"] = auc
        results_test[f"{model_name}_Accuracy"] = acc
        print(f"{model_name} - AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    results_test["Test"] = f"Test {test_idx}: {selected_media}, {selected_week}"
    all_results.append(results_test)

###########################################
# Save and Display Results
###########################################
all_results_df = pd.DataFrame(all_results)
output_csv = "results/test_case_results_mediaSc.csv"
all_results_df.to_csv(output_csv, index=False)

print("\nFinal Results Summary:")
print(all_results_df)
