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

# check for nan in test samples and drop them
nan_indices = [i for i, s in enumerate(test_samples) if np.isnan(s.intensity).any() or np.isnan(s.mz).any()]
test_samples = [s for i, s in enumerate(test_samples) if i not in nan_indices]
test_id_label = [id_label for i, id_label in enumerate(test_id_label) if i not in nan_indices]
test_media_label = [media_label for i, media_label in enumerate(test_media_label) if i not in nan_indices]
Y_test = np.delete(Y_test, nan_indices)

# Function to generate augmented datasets of size `N`
def generate_augmented_dataset(baseline_samples, Y_labels, augmenter, N, augmentation_fn=None, **kwargs):
    if N == 1:  # No augmentation, return original samples repeated
        augmented_samples = baseline_samples
        augmented_labels = Y_labels
    else:
        # The dataset is the same dataset and then the augmentations
        augmented_samples = baseline_samples.copy()
        augmented_labels = np.copy(Y_labels)
        for i in range(N - 1):
            # Select a random sample to augment
            sample_idx = np.random.randint(0, len(baseline_samples))
            sample = baseline_samples[sample_idx]
            sample_to_augment = SpectrumObject(mz=np.copy(sample.mz), intensity=np.copy(sample.intensity))

            # Augment the sample
            augmented_sample = augmentation_fn(sample_to_augment, **kwargs)
            augmented_samples.append(augmented_sample)
            augmented_labels = np.append(augmented_labels, Y_labels[sample_idx])
        

    return augmented_samples, augmented_labels

# Define experiment configurations
augmentation_techniques = {
    "intensity_variability": {
        "fn": lambda s, **kwargs: augmenter.intensity_variability(s, **kwargs),
        "params": list(itertools.product([0.10, 0.40, 0.60], [0.01, 0.1, 0.5, 1])),  # Combinations of p1 and v1
    },
    "proportional_shifting": {
        "fn": lambda s, **kwargs: augmenter.proportional_shifting(s, **kwargs),
        "params": {"p0": [0.2, 0.4, 0.8, 1]},  # Fraction of peaks to shift
    },
    "machine_variability": {
        "fn": lambda s, **kwargs: augmenter.machine_variability(s, **kwargs),
        "params": {"v3": [0.01, 0.05, 0.1, 0.2]},
    },
}

augmenter = DataAugmenter(random_state=42)

# Dataset sizes for augmentation
dataset_multiples = [1, 3, 5, 10, 20] # From N to 10*N


# Baseline AUC without any augmentation
baseline_rf_model = RandomForestClassifier(random_state=42)
grid_rf = {'n_estimators': [100, 200, 300]}
baseline_rf_model = GridSearchCV(baseline_rf_model, grid_rf, cv=5, n_jobs=-1, verbose=1)


X_train_baseline = [s.intensity for s in baseline_samples]
X_test_baseline = [s.intensity for s in test_samples]

baseline_rf_model.fit(X_train_baseline, Y_train)
baseline_rf_auc = roc_auc_score(Y_test, baseline_rf_model.best_estimator_.predict_proba(X_test_baseline), multi_class='ovr')
baseline_rf_acc = baseline_rf_model.best_estimator_.score(X_test_baseline, Y_test)

# File to store experiment results
csv_filename = "results/augmentation_experiment_combinations.csv"
# Load existing results if CSV exists, otherwise initialize an empty list
if os.path.exists(csv_filename):
    existing_results_df = pd.read_csv(csv_filename)
    results = existing_results_df.to_dict("records")
else:
    results = []

for technique, config in augmentation_techniques.items():
    print(f"Running experiments for {technique}...")
    params_list = config["params"]

    if isinstance(params_list, dict):  # Handle single-parameter cases
        params_list = [dict(zip(params_list.keys(), values)) for values in itertools.product(*params_list.values())]

    for params in tqdm(params_list, desc=f"Testing {technique} parameters"):
        for multiple in dataset_multiples:
            N = multiple * len(baseline_samples)
            # Format parameters consistently
            if technique == "intensity_variability":
                params_new_format = dict(zip(["p1", "v1"], params))
            else:
                params_new_format = params

            # Check if the current experiment combination already exists
            combination_exists = any(
                (r["Technique"] == technique) and 
                (r["Parameters"] == str(params_new_format)) and 
                (r["N"] == N) and 
                (r["Model"] == "RF")
                for r in results
            )
            if combination_exists:
                print(f"Skipping combination: Technique={technique}, Parameters={params_new_format}, N={N}")
                continue

            # Initialize storage for global predictions and true labels
            rf_predictions_all = []
            Y_test_all = []

            for idx, left_out_id in tqdm(enumerate(baseline_id_label), 
                             total=len(baseline_id_label), 
                             desc="Leave-One-Out Epochs"):
                # Leave-One-Out Split
                baseline_samples_loo = [
                    SpectrumObject(mz=np.copy(s.mz), intensity=np.copy(s.intensity)) 
                    for i, s in enumerate(baseline_samples) if baseline_id_label[i] != left_out_id
                ]
                Y_train_loo = np.array([label for i, label in enumerate(Y_train) if baseline_id_label[i] != left_out_id])
                
                # Generate augmented dataset
                augmented_samples, augmented_labels = generate_augmented_dataset(
                    baseline_samples_loo, Y_train_loo, augmenter, N, config["fn"], **(params_new_format if isinstance(params_new_format, dict) else {})
                )
                X_train_loo = [s.intensity for s in augmented_samples]
                Y_train_loo = augmented_labels
                
                # Filter test samples excluding those sharing `test_id_label` with the left-out training sample
                test_excluded_ids = {
                    test_id_label[i]
                    for i, base_id in enumerate(baseline_id_label) if base_id == left_out_id
                }
                X_test = [
                    s.intensity for i, s in enumerate(test_samples) if test_id_label[i] not in test_excluded_ids
                ]
                Y_test_loo = [
                    label for i, label in enumerate(Y_test) if test_id_label[i] not in test_excluded_ids
                ]

                # Hyperparameter grid search
                rf_model = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    grid_rf,
                    cv=5,
                    scoring="roc_auc_ovr",
                    n_jobs=-1,
                )
                rf_model.fit(X_train_loo, Y_train_loo)
                rf_predictions = rf_model.best_estimator_.predict_proba(X_test)

                # Accumulate predictions and true labels for global AUC calculation
                rf_predictions_all.extend(rf_predictions)
                Y_test_all.extend(Y_test_loo)

            # Compute global AUC after all LOO iterations
            rf_auc_global = roc_auc_score(Y_test_all, rf_predictions_all, multi_class="ovr")
            rf_acc_global = accuracy_score(Y_test_all, np.argmax(rf_predictions_all, axis=1))
            
            # Store global results
            results.append({
                "Technique": technique,
                "Parameters": str(params_new_format),
                "N": N,
                "Model": "RF",
                "AUC": rf_auc_global,
                "Accuracy": rf_acc_global,
            })
            
            # Write intermediate results to CSV after each model training
            pd.DataFrame(results).to_csv(csv_filename, index=False)
            
# Convert results to a DataFrame for easy analysis
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv("results/augmentation_experiment_combinations.csv", index=False)

plt.rcParams.update({'font.size': 16})  # Increase font size for clarity

# Plot results with baseline
for technique in augmentation_techniques.keys():
    for model in ["RF"]:
        plt.figure(figsize=(12, 6))
        subset = results_df[(results_df["Technique"] == technique) & (results_df["Model"] == model)]
        
        # Convert subset["Parameters"] to string to avoid issues with plotting
        subset["Parameters"] = subset["Parameters"].apply(str)

        for params in np.unique(subset["Parameters"]):
            param_subset = subset[subset["Parameters"] == params]
            plt.plot(param_subset["N"], param_subset["AUC"], label=f"{params}")

        # Plot baseline: AUC when N=60 from the CSV
        baseline_auc = results_df[
            (results_df["Technique"] == technique) & 
            (results_df["Model"] == model) & 
            (results_df["N"] == 60)
        ]["AUC"].values[0]
        
        plt.axhline(y=baseline_auc, color="red", linestyle="--", label="Baseline (No Augmentation)")

        plt.xlabel("Number of augmented samples")
        plt.ylabel("AUC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{technique}_{model}_combinatory_performance.png", dpi=300)
        plt.show()
        
        # Plot accuracy
        plt.figure(figsize=(12, 6))
        baseline_acc = baseline_rf_acc
        plt.axhline(y=baseline_acc, color="red", linestyle="--", label="Baseline (No Augmentation)")
        for params in np.unique(subset["Parameters"]):
            param_subset = subset[subset["Parameters"] == params]
            plt.plot(param_subset["N"], param_subset["Accuracy"], label=f"{params}")
        
        plt.title(f"{model} Accuracy with {technique}")
        plt.xlabel("Number of augmented samples")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{technique}_{model}_combinatory_accuracy.png", dpi=300)
        plt.show()
        
        # Plot best DA results
        read_results = pd.read_csv("results/best_da_repeated_results.csv")
        
        plt.figure(figsize=(12, 6))
        plt.axhline(y=0.880800925925926, color="red", linestyle="--", label="Baseline (No Augmentation)")
        auc_mean = read_results["AUC_Mean"].values
        auc_std = read_results["AUC_Std"].values
        x_axis = read_results["N"].values
        # Do a plot with error bars
        plt.errorbar(x_axis, auc_mean, yerr=auc_std, fmt='o-', color='blue', capsize=5, label="Best DA combination")

        plt.xlabel("Number of augmented samples")
        plt.ylabel("AUC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/auc_vs_n_with_baseline_errorbars.png", dpi=300)
        plt.show()
        
        
