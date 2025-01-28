#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Import custom modules/classes
from dataloader.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer
from dataloader.SpectrumObject import SpectrumObject
import pymzml

###########################################
# Global Constants and Configuration
###########################################
# Bacterial strain classes for label encoding
CLASSES = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']

# Define the dataset structure parameters.
semanas = ['Semana 1', 'Semana 2', 'Semana 3']
clases_list = CLASSES  # same order for iteration
medios = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU']

# Use a particular condition for training. For example, here training samples are selected when:
#   medio == 'Medio Ch' and semana == 'Semana 1'
training_media  = 'Medio Ch'
training_week   = 'Semana 1'
n_biomarkers = 10

# Base path for the data (adjust as needed)
base_path = '/export/data_ml4ds/bacteria_id/C_diff/Reproducibilidad/ClostiRepro/ClostriRepro/Reproducibilidad No extracción'

###########################################
# Data Loading
###########################################
baseline_samples = []          # Will hold SpectrumObject instances (training samples)
baseline_id_label = []         # IDs extracted from file names
Y_train = []                   # Class labels

print("Loading training data ...")
for medio in medios:
    for semana in semanas:
        for clase in clases_list:
            ruta = f"{base_path}/{medio}/{semana}/{clase}"
            if os.path.exists(ruta):
                for f in os.listdir(ruta):
                    ruta_f = os.path.join(ruta, f)
                    # Select training samples from specified condition.
                    if medio == training_media and semana == training_week:
                        baseline_id_label.append(f.split('_')[0])
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
                                # Look for 'fid' and 'acqu' files in subfolders.
                                fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
                                acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
                                if fid_files and acqu_files:
                                    ruta_fid = fid_files[0]
                                    ruta_acqu = acqu_files[0]
                                    s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                                    baseline_samples.append(s)
                                    Y_train.append(clase)

# Encode the string labels to integer indices.
label_mapping = {label: idx for idx, label in enumerate(CLASSES)}
Y_train = np.array([label_mapping[label] for label in Y_train])

###########################################
# Preprocessing Pipeline (Binned)
###########################################
binning_step = 3
pipeline_binned = SequentialPreprocessor(
    VarStabilizer(method="sqrt"),
    Smoother(halfwindow=10),
    BaselineCorrecter(method="SNIP", snip_n_iter=10),
    Normalizer(sum=1),
    Trimmer(),
    Binner(step=binning_step),
)

print("Preprocessing training data (binned)...")
baseline_samples_binned = [pipeline_binned(s) for s in tqdm(baseline_samples, desc="Binned Training Samples")]

###########################################
# Prepare Data Matrix for Training
###########################################
# Convert each SpectrumObject's intensity (binned) into a row of the X_train matrix.
X_train_binned = np.array([s.intensity for s in baseline_samples_binned])

###########################################
# One-vs-Rest RF Training and Biomarker Extraction
###########################################
# For each class, create a binary classifier (1 = class, 0 = rest) and extract the top 50
# feature importances from a RandomForestClassifier.
biomarkers = {}  # Dictionary to store for each class: (top_indices, importances)

# Use a default RandomForest configuration.
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("\nTraining One-vs-Rest Random Forest classifiers and extracting biomarkers ...")
for class_label, class_idx in label_mapping.items():
    # Create binary labels: 1 for samples of this class and 0 for all other samples.
    y_binary = (Y_train == class_idx).astype(int)
    
    # Fit the RandomForest classifier using the binned training data.
    rf.fit(X_train_binned, y_binary)
    
    # Extract feature importances.
    importances = rf.feature_importances_
    
    # Sort features by importance (descending order) and select the top 50.
    sorted_indices = np.argsort(importances)[::-1]
    top50_indices = sorted_indices[:n_biomarkers]
    top50_importances = importances[top50_indices]
    
    biomarkers[class_label] = (top50_indices, top50_importances)
    
    print(f"Extracted {len(top50_indices)} biomarkers for class {class_label}.")

###########################################
# Compute Mean Binned Spectrum for Each Class
###########################################
mean_spectra_binned = {}

for class_label, class_idx in label_mapping.items():
    # Find indices of training samples that belong to the current class.
    class_indices = np.where(Y_train == class_idx)[0]
    if len(class_indices) == 0:
        continue
    # Calculate the mean spectrum (mean intensity for each binned feature).
    mean_spectrum_binned = np.mean(X_train_binned[class_indices, :], axis=0)
    mean_spectra_binned[class_label] = mean_spectrum_binned


###########################################
# Plotting: Zoom In on Each Biomarker
###########################################
# Create a main output directory.
output_dir = "results/biomarkers_plots"
os.makedirs(output_dir, exist_ok=True)

print("\nCreating zoom-in plots for each biomarker per class ...")
for class_label, class_idx in label_mapping.items():
    if class_label not in mean_spectra_binned:
        continue
    mean_spec_bin = mean_spectra_binned[class_label]
    
    # Create a folder for the class
    class_folder = os.path.join(output_dir, class_label)
    os.makedirs(class_folder, exist_ok=True)
    
    # Retrieve the top biomarkers (indices and importances).
    top_indices, top_importances = biomarkers[class_label]
    
    for biomarker_no, (idx, imp) in enumerate(zip(top_indices, top_importances), start=1):
        # Define the zoom window (from idx - 200 to idx + 200)
        start_idx = max(idx - 200, 0)  # ensure start index is non-negative
        end_idx = min(idx + 200, len(mean_spec_bin))
        
        # Get the x-axis values and spectrum for the zoom window.
        x_zoom = np.arange(start_idx, end_idx)
        zoom_spec = mean_spec_bin[start_idx:end_idx]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_zoom, zoom_spec, label="Zoomed Mean Spectrum", color="blue")
        
        # Overlay a vertical bar at the biomarker position.
        # Here we draw a bar from the bottom to the top of the plot.
        bar_height = np.max(zoom_spec)
        # To center the bar at 'idx', we use a width of 13 (i.e. ±6 indices)
        plt.bar(x=idx, height=bar_height, width=13, color='red', alpha=0.2, align='center',
                label=f"Biomarker {biomarker_no}")
        
        plt.xlabel("Binned Feature Index")
        plt.ylabel("Intensity")
        plt.title(f"Class {class_label} - Biomarker {biomarker_no} (Index {idx})")
        plt.legend()
        plt.tight_layout()
        
        # Save plot; each biomarker gets its own image.
        plot_path = os.path.join(class_folder, f"{class_label}_biomarker_{biomarker_no}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved zoom-in plot for class {class_label} biomarker {biomarker_no} at: {plot_path}")

print("\nBiomarker extraction and zoomed plotting complete.")