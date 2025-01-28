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
baseline_id_label = []
test_media_label = []
test_semana_label = []
test_id_label = []

for medio in medios:
    for semana in semanas:
        for clase in clases:
            ruta = f'/export/data_ml4ds/bacteria_id/C_diff/Reproducibilidad/ClostiRepro/ClostriRepro/Reproducibilidad No extracción/{medio}/{semana}/{clase}' 
            
            if os.path.exists(ruta):
                for f in os.listdir(ruta):
                    ruta_f = os.path.join(ruta, f)
                    
                    if medio == 'Medio Ch'and semana == 'Semana 1':
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
        "params": list(itertools.product([0.10, 0.40, 0.60], [0.01, 0.05, 0.1, 0.5, 1])),  # Combinations of p1 and v1
    },
    "proportional_shifting": {
        "fn": lambda s, **kwargs: augmenter.proportional_shifting(s, **kwargs),
        "params": {"p0": [0.2, 0.4, 0.6, 0.8, 1]},  # Fraction of peaks to shift
    },
    "machine_variability": {
        "fn": lambda s, **kwargs: augmenter.machine_variability(s, **kwargs),
        "params": {"v3": [0.01, 0.05, 0.1, 0.2]},
    },
}

augmenter = DataAugmenter(random_state=42)

# Dataset sizes for augmentation
dataset_multiples = [1, 2, 3, 4, 5, 10, 20, 30, 100] # From N to 10*N


# Baseline AUC without any augmentation
baseline_rf_model = RandomForestClassifier(random_state=42)
grid_rf = {'n_estimators': [100, 200, 300]}
baseline_rf_model = GridSearchCV(baseline_rf_model, grid_rf, cv=3, n_jobs=-10, verbose=1)
baseline_lgbm_model = LGBMClassifier(random_state=42)
grid_lgbm = {'n_estimators': [100, 200, 300]}
baseline_lgbm_model = GridSearchCV(baseline_lgbm_model, grid_lgbm, cv=3, n_jobs=-10, verbose=1)

X_train_baseline = [s.intensity for s in baseline_samples]
X_test_baseline = [s.intensity for s in test_samples]

baseline_rf_model.fit(X_train_baseline, Y_train)
baseline_rf_auc = roc_auc_score(Y_test, baseline_rf_model.best_estimator_.predict_proba(X_test_baseline), multi_class='ovr')
baseline_rf_acc = baseline_rf_model.best_estimator_.score(X_test_baseline, Y_test)

baseline_lgbm_model.fit(X_train_baseline, Y_train)
baseline_lgbm_auc = roc_auc_score(Y_test, baseline_lgbm_model.best_estimator_.predict_proba(X_test_baseline), multi_class='ovr')
baseline_lgbm_acc = baseline_lgbm_model.best_estimator_.score(X_test_baseline, Y_test)

results = []

for technique, config in augmentation_techniques.items():
    print(f"Running experiments for {technique}...")
    params_list = config["params"]

    if isinstance(params_list, dict):  # Handle single-parameter cases
        params_list = [dict(zip(params_list.keys(), values)) for values in itertools.product(*params_list.values())]

    for params in tqdm(params_list, desc=f"Testing {technique} parameters"):
        for multiple in dataset_multiples:
            N = multiple * len(baseline_samples)
            # Initialize storage for global predictions and true labels
            rf_predictions_all = []
            lgbm_predictions_all = []
            Y_test_all = []

            for idx, left_out_id in tqdm(enumerate(baseline_id_label), 
                             total=len(baseline_id_label), 
                             desc="Leave-One-Out Epochs"):
                # Leave-One-Out Split
                baseline_samples_loo = [
                    SpectrumObject(mz=np.copy(s.mz), intensity=np.copy(s.intensity)) for i, s in enumerate(baseline_samples) if baseline_id_label[i] != left_out_id
                ]
                Y_train_loo = np.array([label for i, label in enumerate(Y_train) if baseline_id_label[i] != left_out_id])
                
                # Generate augmented dataset
                augmented_samples, augmented_labels = generate_augmented_dataset(
                    baseline_samples_loo, Y_train_loo, augmenter, N, config["fn"], **dict(params if isinstance(params, dict) else zip(["p1", "v1"], params))
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
                    cv=3,
                    scoring="roc_auc_ovr",
                    n_jobs=-10,
                )
                rf_model.fit(X_train_loo, Y_train_loo)
                rf_predictions = rf_model.best_estimator_.predict_proba(X_test)
                lgbm_model = GridSearchCV(
                    LGBMClassifier(random_state=42),
                    grid_lgbm,
                    cv=3,
                    scoring="roc_auc_ovr",
                    n_jobs=-10,
                )
                lgbm_model.fit(X_train_loo, Y_train_loo)
                # lgbm_predictions = LGBMClassifier(random_state=42).predict_proba(X_test)
                lgbm_predictions = lgbm_model.best_estimator_.predict_proba(X_test)
                # lgbm_predictions = np.zeros((len(X_test), len(CLASSES)))

                # Accumulate predictions and true labels for global AUC calculation
                rf_predictions_all.extend(rf_predictions)
                lgbm_predictions_all.extend(lgbm_predictions)
                Y_test_all.extend(Y_test_loo)

            # Compute global AUC after all LOO iterations
            rf_auc_global = roc_auc_score(Y_test_all, rf_predictions_all, multi_class="ovr")
            lgbm_auc_global = roc_auc_score(Y_test_all, lgbm_predictions_all, multi_class="ovr")
            rf_acc_global = accuracy_score(Y_test_all, np.argmax(rf_predictions_all, axis=1))
            lgbm_acc_global = accuracy_score(Y_test_all, np.argmax(lgbm_predictions_all, axis=1))
            

            # Format parameters for intensity_variability
            if technique == "intensity_variability":
                params_new_format = dict(zip(["p1", "v1"], params))
            else:
                params_new_format = params

            # Store global results
            results.append({
                "Technique": technique,
                "Parameters": params_new_format,
                "N": N,
                "Model": "RF",
                "AUC": rf_auc_global,
                "Accuracy": rf_acc_global,
            })
            results.append({
                "Technique": technique,
                "Parameters": params_new_format,
                "N": N,
                "Model": "LGBM",
                "AUC": lgbm_auc_global,
                "Accuracy": lgbm_acc_global,
            })


# Convert results to a DataFrame for easy analysis
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv("results/augmentation_experiment_combinations.csv", index=False)

# Plot results with baseline
for technique in augmentation_techniques.keys():
    for model in ["RF", "LGBM"]:
        plt.figure(figsize=(12, 6))
        subset = results_df[(results_df["Technique"] == technique) & (results_df["Model"] == model)]
        
        # Convert subset["Parametesr"] to string to avoid issues with plotting
        subset["Parameters"] = subset["Parameters"].apply(str)

        for params in np.unique(subset["Parameters"]):
            param_subset = subset[subset["Parameters"] == params]
            plt.plot(param_subset["N"], param_subset["AUC"], label=f"{params}")

        # Plot baseline
        baseline_auc = baseline_rf_auc if model == "RF" else baseline_lgbm_auc
        plt.axhline(y=baseline_auc, color="red", linestyle="--", label="Baseline (No Augmentation)")

        plt.title(f"{model} Performance with {technique}")
        plt.xlabel("N (Number of Samples)")
        plt.ylabel("AUC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{technique}_{model}_combinatory_performance.png")
        plt.show()
        
        # Plot accuracy
        plt.figure(figsize=(12, 6))
        baseline_acc = baseline_rf_acc if model == "RF" else baseline_lgbm_acc
        plt.axhline(y=baseline_acc, color="red", linestyle="--", label="Baseline (No Augmentation)")
        for params in np.unique(subset["Parameters"]):
            param_subset = subset[subset["Parameters"] == params]
            plt.plot(param_subset["N"], param_subset["Accuracy"], label=f"{params}")
        
        plt.title(f"{model} Accuracy with {technique}")
        plt.xlabel("N (Number of Samples)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{technique}_{model}_combinatory_accuracy.png")
        plt.show()