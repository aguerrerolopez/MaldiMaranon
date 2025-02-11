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
MEDIA = ['Chx', 'Brx', 'Clx', 'Scx', 'GU']

# Initialize lists for data
Y_train, Y_test = [], []
baseline_samples, test_samples = [], []

semanas = ['Semana 1', 'Semana 2', 'Semana 3']
clases = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']
medios = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU']
medios_pe = ['Chx', 'Brx', 'Clx', 'Scx', 'GU']

Y_train = []
baseline_samples = []
test_samples = []
Y_test = []
baseline_media_label = []
baseline_semana_label = []
baseline_id_label = []
test_media_label = []
test_pe_label = []
test_semana_label = []
test_id_label = []

for i, medio in enumerate(medios):
    for semana in semanas:
        for clase in clases:
            ruta = f'/export/data_ml4ds/bacteria_id/C_diff/Reproducibilidad/ClostiRepro/ClostriRepro/Reproducibilidad No extracción/{medio}/{semana}/{clase}' 
            ruta_pe = f'/export/data_ml4ds/bacteria_id/C_diff/Reproducibilidad/ClostiRepro/ClostriRepro/Reproducibilidad Extracción/{medios_pe[i]}/{clase}'
            
            if os.path.exists(ruta):
                for f in os.listdir(ruta):
                    ruta_f = os.path.join(ruta, f)
                    
                    if medio == 'Medio Ch' and semana == 'Semana 1':
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
                        test_pe_label.append(0)
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
                    ruta_f = os.path.join(ruta_pe, f)
                    test_id_label.append(f.split('_')[0])
                    test_pe_label.append(1)
                    test_media_label.append(medio)
                    test_semana_label.append('Semana 1')
                    # Si el archivo es un .mzml
                    if 'mzml' in ruta_f:
                        run = pymzml.run.Reader(ruta_f)
                        spectro = [r for r in run]
                        s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                        test_samples.append(s)
                        Y_test.append(clase)
                                
                    else: 
                        carpetas = [subf for subf in os.listdir(ruta_f)]
                        ruta_f = os.path.join(ruta_pe, f, carpetas[0])
                        fid_files = glob(os.path.join(ruta_f, '*', '1SLin', 'fid'))
                        acqu_files = glob(os.path.join(ruta_f, '*', '1SLin', 'acqu'))

                        ruta_fid = fid_files[0]
                        ruta_acqu = acqu_files[0]
                        s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                        test_samples.append(s)
                        Y_test.append(clase)
                        # If spectrum is emtpy or nan remove it from everywhere
                        if np.isnan(s.intensity).any() or np.isnan(s.mz).any():
                            test_samples.pop()
                            Y_test.pop()
                            test_id_label.pop()
                            test_pe_label.pop()
                            test_media_label.pop()
                            test_semana_label.pop()
                        


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

##############################################################
# Function to train model and return predictions for a given split
##############################################################
def get_predictions(model, param_grid, X_train, Y_train_loo, X_test):
    grid_model = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc_ovr", n_jobs=-1, verbose=0)
    grid_model.fit(X_train, Y_train_loo)
    best_model = grid_model.best_estimator_
    preds = best_model.predict_proba(X_test)
    return preds

##############################################################
# Test 0: Leave-One-Out on the Baseline Data
##############################################################
print("\nRunning Test 0: Baseline (Leave-One-Out on training data)...")
test0_results = {"Test": "Test 0 (Baseline LOO)"}

# Model selection flags
USE_RF = True
USE_SVM_LINEAR = True
USE_SVM_RBF = True
USE_KNN = True
USE_LGBM = True

# Parameter grids
rf_grid = {"n_estimators": [100, 200, 300]}
svm_linear_grid = {"C": [0.01, 0.1, 1, 10, 100]}
svm_rbf_grid = {"C": [0.01, 0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}
knn_grid = {"n_neighbors": list(range(3, 21))}
lgbm_grid = {"n_estimators": [100, 200, 300]}

# Prepare a dictionary that maps model names to tuples (flag, model instance, parameter grid)
models_info = {
    "RF": (USE_RF, RandomForestClassifier(random_state=42), rf_grid),
    "SVM_Linear": (USE_SVM_LINEAR, SVC(kernel="linear", probability=True, random_state=42), svm_linear_grid),
    "SVM_RBF": (USE_SVM_RBF, SVC(kernel="rbf", probability=True, random_state=42), svm_rbf_grid),
    "KNN": (USE_KNN, KNeighborsClassifier(), knn_grid),
    "LGBM": (USE_LGBM, LGBMClassifier(random_state=42), lgbm_grid),
}

# # For each model, store accumulated predictions and true labels
# results_test0 = {}

# for model_name, (use_model, model, param_grid) in models_info.items():
#     if not use_model:
#         continue

#     print(f"\nTraining {model_name} for Test 0...")
#     preds_all = []   # to accumulate prediction probabilities
#     ytrue_all = []   # to accumulate the corresponding true labels

#     # Leave-One-Out on the baseline samples
#     for left_out_id in tqdm(np.unique(baseline_id_label), desc=f"{model_name} LOO Progress"):
#         # Build training data (all baseline samples not matching left_out_id)
#         X_train = [s.intensity for i, s in enumerate(baseline_samples) if baseline_id_label[i] != left_out_id]
#         Y_train_loo = [Y_train[i] for i in range(len(baseline_samples)) if baseline_id_label[i] != left_out_id]

#         # Build test data (all baseline samples with left_out_id)
#         X_test_loo = [s.intensity for i, s in enumerate(baseline_samples) if baseline_id_label[i] == left_out_id]
#         Y_test_loo = [Y_train[i] for i in range(len(baseline_samples)) if baseline_id_label[i] == left_out_id]

#         # Skip iteration if no test data (should not happen)
#         if len(X_test_loo) == 0:
#             continue

#         # Get predictions for this fold
#         preds = get_predictions(model, param_grid, X_train, Y_train_loo, X_test_loo)
#         preds_all.append(preds)
#         ytrue_all.extend(Y_test_loo)

#     # Concatenate all predictions and compute metrics
#     if len(preds_all) > 0:
#         # preds_all is a list of numpy arrays so we concatenate along axis 0
#         preds_all = np.concatenate(preds_all, axis=0)
#         # If more than one class is present, compute ROC AUC; otherwise, set as nan.
#         if len(np.unique(ytrue_all)) < 2:
#             auc = np.nan
#         else:
#             auc = roc_auc_score(ytrue_all, preds_all, multi_class="ovr")
#         # Accuracy: compare the argmax predictions vs accumulated true labels.
#         acc = accuracy_score(ytrue_all, np.argmax(preds_all, axis=1))
#     else:
#         auc, acc = np.nan, np.nan

#     results_test0[f"{model_name}_AUC"] = auc
#     results_test0[f"{model_name}_Accuracy"] = acc
#     print(f"{model_name} (Test 0) - AUC: {auc:.4f}, Accuracy: {acc:.4f}")

##############################################################
# Test Cases using filtered test samples
##############################################################
test_cases = [
    # {"Media": "Medio Ch", "Week": "Semana 1", "PE": 0},
    {"Media": "Medio Ch", "Week": "Semana 1", "PE": 1},
    # {"Media": "Medio Ch", "Week": "Semana 2", "PE": 0},
    # {"Media": "Medio Ch", "Week": "Semana 3", "PE": 0},
    # {"Media": "Medio Br", "Week": "Semana 1", "PE": 0},
    # {"Media": "Medio Cl", "Week": "Semana 1", "PE": 0},
    # {"Media": "Medio Sc", "Week": "Semana 1", "PE": 0},
    # {"Media": "GU", "Week": "Semana 1"},
]

# List to collect all results; start with Test 0 results.
# all_results = [dict(Test="Test 0 (Baseline LOO)", **results_test0)]
all_results = []
for test_idx, test_case in enumerate(test_cases, start=1):
    selected_media = test_case["Media"]
    selected_week = test_case["Week"]
    selected_pe = test_case["PE"]

    print(f"\nRunning Test {test_idx}: Media={selected_media}, Week={selected_week}, PE={selected_pe}...")

    # Filter test samples by media and week.
    filtered_test_indices = [
        i for i, (media, week, pe) in enumerate(zip(test_media_label, test_semana_label, test_pe_label))
        if media == selected_media and week == selected_week and pe == selected_pe
    ]
    filtered_test_samples = [test_samples[i] for i in filtered_test_indices]
    filtered_Y_test = [Y_test[i] for i in filtered_test_indices]
    filtered_test_id_labels = [test_id_label[i] for i in filtered_test_indices]

    if len(filtered_test_samples) == 0:
        print(f"Test {test_idx} skipped: No matching test samples.")
        continue

    # For each model, accumulate predictions and true labels over LOO iterations.
    results_test = {}
    for model_name, (use_model, model, param_grid) in models_info.items():
        if not use_model:
            continue

        print(f"Training {model_name} for Test {test_idx}...")
        preds_all = []   # predictions across iterations
        ytrue_all = []   # true labels across iterations

        # LOO on baseline_id for training set.
        for left_out_id in tqdm(np.unique(baseline_id_label), desc=f"{model_name} LOO Progress"):
            # Build training data: baseline samples not having left_out_id.
            X_train = [s.intensity for i, s in enumerate(baseline_samples) if baseline_id_label[i] != left_out_id]
            Y_train_loo = [Y_train[i] for i in range(len(baseline_samples)) if baseline_id_label[i] != left_out_id]

            # Build test data for this iteration:
            # Remove from filtered test samples all with the left_out_id.
            indices_for_test = [i for i, tid in enumerate(filtered_test_id_labels) if tid != left_out_id]
            if len(indices_for_test) == 0:
                continue

            X_test_loo = [filtered_test_samples[i].intensity for i in indices_for_test]
            Y_test_loo = [filtered_Y_test[i] for i in indices_for_test]
            
            # Check for nans and drop them
            nan_indices = [i for i, s in enumerate(X_test_loo) if np.isnan(s).any()]
            if len(nan_indices) > 0:
                # Drop them from X and Y
                X_test_loo = [X_test_loo[i] for i in range(len(X_test_loo)) if i not in nan_indices]
                Y_test_loo = [Y_test_loo[i] for i in range(len(Y_test_loo)) if i not in nan_indices]

            # Get predictions on filtered test samples for the current LOO iteration.
            preds = get_predictions(model, param_grid, X_train, Y_train_loo, X_test_loo)
            preds_all.append(preds)
            ytrue_all.extend(Y_test_loo)

        # Concatenate predictions and compute final metrics.
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

    # Append results with a label for the test.
    results_test["Test"] = f"Test {test_idx}: {selected_media}, {selected_week}"
    all_results.append(results_test)

# Save all results to CSV.
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv("results/test_case_results_table1.csv", index=False)

# Print summary of all test cases.
print("\nFinal Results Summary:")
print(all_results_df)
