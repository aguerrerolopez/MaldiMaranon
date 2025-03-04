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

train=True

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
            ruta = f'../../data/ClostriRepro/Reproducibilidad No extracción/{medio}/{semana}/{clase}' 
            
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



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to perform repeated experiments and collect statistics
def run_experiments(repeats, dataset_multiples, augmenter, baseline_samples, Y_train, test_samples, Y_test):
    results = []

    # Baseline Random Forest (No Augmentation)
    print("Training baseline Random Forest (No Augmentation)...")
    X_train_baseline = [s.intensity for s in baseline_samples]
    X_test_baseline = [s.intensity for s in test_samples]

    baseline_rf_model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_grid,
        cv=3,
        scoring="roc_auc_ovr",
        n_jobs=-10,
        verbose=1,
    )
    baseline_rf_model.fit(X_train_baseline, Y_train)
    baseline_rf_best = baseline_rf_model.best_estimator_
    baseline_rf_auc = roc_auc_score(Y_test, baseline_rf_best.predict_proba(X_test_baseline), multi_class="ovr")
    baseline_rf_acc = baseline_rf_best.score(X_test_baseline, Y_test)

    # Store baseline results
    results.append({
        "Technique": "No Augmentation",
        "N": 0,
        "Model": "RF",
        "AUC_Mean": baseline_rf_auc,
        "AUC_Std": 0,
        "Accuracy_Mean": baseline_rf_acc,
        "Accuracy_Std": 0,
    })

    # Augmentation configurations
    proportional_shifting_p0 = 0.8
    machine_variability_v3 = 0.05

    # Loop over dataset multiples
    for multiple in tqdm(dataset_multiples, desc="Dataset Multiples"):
        N = multiple * len(baseline_samples)

        auc_scores = []
        acc_scores = []

        for repeat in range(repeats):
            # Generate augmented dataset
            augmented_samples, augmented_labels = generate_augmented_dataset(
                baseline_samples, Y_train, augmenter, N, augmenter.proportional_shifting, p0=proportional_shifting_p0
            )
            augmented_samples, augmented_labels = generate_augmented_dataset(
                augmented_samples, augmented_labels, augmenter, N, augmenter.machine_variability, v3=machine_variability_v3
            )

            # Prepare data
            X_train = [s.intensity for s in augmented_samples]
            Y_train_augmented = augmented_labels
            X_test = [s.intensity for s in test_samples]

            # Train Random Forest with Grid Search
            rf_model = GridSearchCV(
                RandomForestClassifier(random_state=42),
                rf_grid,
                cv=3,
                scoring="roc_auc_ovr",
                n_jobs=-10,
                verbose=1,
            )
            rf_model.fit(X_train, Y_train_augmented)
            rf_best = rf_model.best_estimator_

            # Predictions and metrics
            rf_predictions = rf_best.predict_proba(X_test)
            rf_auc = roc_auc_score(Y_test, rf_predictions, multi_class="ovr")
            rf_acc = accuracy_score(Y_test, np.argmax(rf_predictions, axis=1))

            # Collect scores
            auc_scores.append(rf_auc)
            acc_scores.append(rf_acc)

        # Compute mean and std for AUC and Accuracy
        results.append({
            "Technique": "Best DA",
            "N": N,
            "Model": "RF",
            "AUC_Mean": np.mean(auc_scores),
            "AUC_Std": np.std(auc_scores),
            "Accuracy_Mean": np.mean(acc_scores),
            "Accuracy_Std": np.std(acc_scores),
        })

    return pd.DataFrame(results)

# Define grid search parameters for Random Forest
rf_grid = {
    "n_estimators": [50, 100, 200, 300],
}

# Run experiments
repeats = 10
# dataset_multiples = [1] # Up to 10× dataset size
dataset_multiples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 45, 50, 100] 
augmenter = DataAugmenter(random_state=42)
results_df = run_experiments(repeats, dataset_multiples, augmenter, baseline_samples, Y_train, test_samples, Y_test)

# Save results to CSV
results_df.to_csv("results/best_da_repeated_results.csv", index=False)

# Plot AUC vs. N with error bars
plt.figure(figsize=(12, 6))
plt.plot(results_df["N"], results_df["AUC_Mean"], "o-", label="AUC")
plt.errorbar(results_df["N"], results_df["AUC_Mean"], yerr=results_df["AUC_Std"], fmt="o", capsize=5, label="AUC")
plt.title("AUC vs. N (Augmented Samples)")
plt.xlabel("N (Number of Samples)")
plt.ylabel("AUC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/auc_vs_n_with_baseline_errorbars.png")
plt.show()

# Plot Accuracy vs. N with error bars
plt.figure(figsize=(12, 6))
plt.plot(results_df["N"], results_df["Accuracy_Mean"], "o-", label="Accuracy")
plt.errorbar(results_df["N"], results_df["Accuracy_Mean"], yerr=results_df["Accuracy_Std"], fmt="o", capsize=5, label="Accuracy")
plt.title("Accuracy vs. N (Augmented Samples)")
plt.xlabel("N (Number of Samples)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy_vs_n_with_baseline_errorbars.png")
plt.show()
