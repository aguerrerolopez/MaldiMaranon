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
binning_step = 1
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

import numpy as np
import matplotlib.pyplot as plt

# Helper function to extract a zoomed m/z region from a spectrum.
def extract_zoom(spectrum, low, high):
    mask = (spectrum.mz >= low) & (spectrum.mz <= high)
    return spectrum.mz[mask], spectrum.intensity[mask]

# Compute the mean spectrum over the specified region for a list of spectra.
def mean_spectrum(spectra, low, high):
    if not spectra:
        return None, None
    # Use the first spectrum for the m/z grid.
    x, _ = extract_zoom(spectra[0], low, high)
    intensities = np.array([extract_zoom(s, low, high)[1] for s in spectra])
    return x, np.mean(intensities, axis=0)

# A small helper to filter spectra by a given condition on test indices.
def filter_condition(indices, condition, sample_list):
    return [sample_list[i] for i in indices if condition(i)]

# Process one ribotype: filter baseline and test spectra into several conditions,
# then compute the mean spectrum for each.
def process_ribotype(ribotype, baseline_samples, Y_train, test_samples, Y_test,
                     test_media_label, test_semana_label, test_pe_label, region0, region1):
    ribo_label = label_mapping[ribotype]  # Get numeric label from your mapping

    # Baseline: Always "Ch, week 1" (from Y_train and baseline_samples)
    baseline_idx = [i for i, lab in enumerate(Y_train) if lab == ribo_label]
    conditions = {"Ch, week 1": [baseline_samples[i] for i in baseline_idx]}
    
    # Test samples: filter by ribotype from Y_test.
    test_idx = [i for i, lab in enumerate(Y_test) if lab == ribo_label]
    conditions["Ch, week 2"] = filter_condition(
        test_idx,
        lambda i: test_media_label[i] == 'Medio Ch' and test_semana_label[i] == 'Semana 2',
        test_samples
    )
    conditions["Media variability Sc"] = filter_condition(
        test_idx,
        lambda i: test_media_label[i] == 'Medio Sc' and test_semana_label[i] == 'Semana 1',
        test_samples
    )
    conditions["Media variability Br"] = filter_condition(
        test_idx,
        lambda i: test_media_label[i] == 'Medio Br' and test_semana_label[i] == 'Semana 1',
        test_samples
    )
    conditions["PE variability"] = filter_condition(
        test_idx,
        lambda i: test_media_label[i] == 'Medio Ch' and test_semana_label[i] == 'Semana 1' and test_pe_label[i] == 1,
        test_samples
    )
    conditions["Machine variability"] = filter_condition(
        test_idx,
        lambda i: test_media_label[i] == 'GU' and test_semana_label[i] == 'Semana 1',
        test_samples
    )
    
    # Compute the mean spectra in REGION for all conditions.
    means = {}
    for key, spec_list in conditions.items():
        x, m = mean_spectrum(spec_list, region0, region1)
        means[key] = (x, m)
    return means

import matplotlib.ticker as ticker

# A helper to plot all the mean spectra along with any vertical lines.
def plot_means(means, title, vlines=[], y_lim=0.005):
    plt.figure(figsize=(10, 6))
    for label, (x, m) in means.items():
        if m is not None:
            plt.plot(x, m, label=label)
    # Draw vertical lines (each a tuple: (x_position, label))
    for x_val, vlabel in vlines:
        plt.axvline(x=x_val, color='r', lw=2, ls='--', label=vlabel, alpha=0.5)
    plt.xlabel("m/z")
    plt.ylabel("Mean Intensity")
    plt.ylim(0, y_lim)
    plt.title(title)
    
    # Set x-axis major ticks at each 1 unit increment
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.grid(which='major', axis='x')
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
############## this region to demonstrate shifting variability ##############

REGION = (4920, 4940)

# Process and plot for each ribotype:
# For RT027: a vertical line at 4933 m/z
means_RT027 = process_ribotype('RT027', baseline_samples, Y_train, test_samples, Y_test,
                               test_media_label, test_semana_label, test_pe_label, REGION[0], REGION[1])
plot_means(means_RT027, f"Mean RT027 Spectra ({REGION[0]}-{REGION[1]} m/z)",
            vlines=[(4933, 'Biomarker 4933 m/z')])


# For RT181: a vertical line at 4993 m/z
means_RT181 = process_ribotype('RT181', baseline_samples, Y_train, test_samples, Y_test,
                               test_media_label, test_semana_label, test_pe_label, REGION[0], REGION[1])
plot_means(means_RT181, f"Mean RT181 Spectra ({REGION[0]}-{REGION[1]} m/z)",
            vlines=[(4933, 'Biomarker 4933 m/z')])

# For RT078: vertical lines for both biomarkers
means_RT078 = process_ribotype('RT078', baseline_samples, Y_train, test_samples, Y_test,
                               test_media_label, test_semana_label, test_pe_label, REGION[0], REGION[1])
plot_means(means_RT078, f"Mean RT078 Spectra ({REGION[0]}-{REGION[1]} m/z)",
           vlines=[(4933, 'Biomarker RT181 4933 m/z')])



############   this region to demonstrate intensity variability   ############

# Global variable for the region of interest (adjust as needed)
REGION = (4980, 5010)

# Process and plot for each ribotype:

# For RT027: a vertical line at 4933 m/z
means_RT027 = process_ribotype('RT027', baseline_samples, Y_train, test_samples, Y_test,
                               test_media_label, test_semana_label, test_pe_label, REGION[0], REGION[1])
plot_means(means_RT027, f"Mean RT027 Spectra ({REGION[0]}-{REGION[1]} m/z)",
            vlines=[(4993, 'Biomarker RT181 4993 m/z')])


# For RT181: a vertical line at 4993 m/z
means_RT181 = process_ribotype('RT181', baseline_samples, Y_train, test_samples, Y_test,
                               test_media_label, test_semana_label, test_pe_label, REGION[0], REGION[1])
plot_means(means_RT181, f"Mean RT181 Spectra ({REGION[0]}-{REGION[1]} m/z)",
            vlines=[(4993, 'Biomarker RT181 4993 m/z')])

# For RT078: vertical lines for both biomarkers
means_RT078 = process_ribotype('RT078', baseline_samples, Y_train, test_samples, Y_test,
                               test_media_label, test_semana_label, test_pe_label, REGION[0], REGION[1])
plot_means(means_RT078, f"Mean RT078 Spectra ({REGION[0]}-{REGION[1]} m/z)",
           vlines=[(4993, 'Biomarker RT181 4993 m/z')])