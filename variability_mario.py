import os
import numpy as np
import pandas as pd
import pymzml
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataloader.preprocess import (
    SequentialPreprocessor,
    VarStabilizer,
    Smoother,
    BaselineCorrecter,
    Trimmer,
    Binner,
    Normalizer,
    IntensityThresholding,
)
from dataloader.SpectrumObject import SpectrumObject

# Global constants
WEEKS = ['Semana 1', 'Semana 2', 'Semana 3']
CLASSES = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']
MEDIA_NOPE = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU', 'Medio Fa']
MEDIA_PE   = ['Chx', 'Brx', 'Clx', 'Scx', 'GUx', 'Fax']
# For variability evaluation we consider five different media
EVAL_MEDIA = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'Medio Fa']
EVAL_TIME = ['Semana 1', 'Semana 2', 'Semana 3']
EVAL_HOSP = ['Medio Ch', 'GU']
EVAL_PE = ['Medio Ch', 'Chx']


# Preprocessing pipeline (as you use in your other code)
binning_step = 3
preprocess_pipeline = SequentialPreprocessor(
    VarStabilizer(method="sqrt"),
    Smoother(halfwindow=10),
    BaselineCorrecter(method="SNIP", snip_n_iter=10),
    Normalizer(sum=1),
    Trimmer(),
    Binner(step=binning_step),
    IntensityThresholding(method='zscore', threshold=1)
)

def _read_spectrum(path):
    """Return a SpectrumObject from path (mzML or Bruker), or None if unfeasible."""
    if os.path.isfile(path) and path.lower().endswith('.mzml'):
        run = pymzml.run.Reader(path)
        scans = [s for s in run]
        if not scans:
            return None
        return SpectrumObject(mz=scans[0].mz, intensity=scans[0].i)
    if os.path.isdir(path):
        subdirs = os.listdir(path)
        if not subdirs:
            return None
        subpath = os.path.join(path, subdirs[0])
        fid = glob(os.path.join(subpath, '*', '1SLin', 'fid'))
        acqu = glob(os.path.join(subpath, '*', '1SLin', 'acqu'))
        if fid and acqu:
            return SpectrumObject().from_bruker(acqu[0], fid[0])
    return None

def _read_folder(base_dir, media_list, ext_type):
    """Traverse base_dir to find (week, class) folders and read valid spectra."""
    data_list = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path): 
            continue
        # Special case if folder is 'Chx_24h'
        media_label = 'Chx_24h' if folder_name == 'Chx_24h' else folder_name
        if media_label not in media_list and media_label != 'Chx_24h': 
            continue

        for week_name in os.listdir(folder_path):
            if week_name not in WEEKS: 
                continue
            week_path = os.path.join(folder_path, week_name)
            if not os.path.isdir(week_path): 
                continue

            for class_name in os.listdir(week_path):
                if class_name not in CLASSES: 
                    continue
                class_path = os.path.join(week_path, class_name)
                if not os.path.isdir(class_path): 
                    continue

                for fname in os.listdir(class_path):
                    path_f = os.path.join(class_path, fname)
                    spec = _read_spectrum(path_f)
                    if spec:
                        data_list.append({
                            'id_label': fname.split('_')[0],
                            'media': media_label,
                            'week': week_name,
                            'class': class_name,
                            'extraction_type': ext_type,
                            'spectrum': spec
                        })
    return data_list

def read_all_data(base_path='/export/data_ml4ds/bacteria_id/C_diff/Reproducibilidad/ClostiRepro/ClostriRepro'):
    """Combine data from NoPE and PE (including 'Chx_24h' if present)."""
    all_data = []
    no_ext = os.path.join(base_path, 'Reproducibilidad No extracción')
    pe_ext = os.path.join(base_path, 'Reproducibilidad Extracción')
    if os.path.isdir(no_ext):
        all_data += _read_folder(no_ext, MEDIA_NOPE, 'NoPE')
    if os.path.isdir(pe_ext):
        all_data += _read_folder(pe_ext, MEDIA_PE, 'PE')
    return all_data

def split_data(data, weeks=None, media=None, classes=None, ext_types=None):
    """Split data into train/test by filters (anything matching filters -> train, else test)."""
    if weeks is None:     weeks = WEEKS
    if media is None:     media = list({d['media'] for d in data})
    if classes is None:   classes = CLASSES
    if ext_types is None: ext_types = ['NoPE', 'PE']

    train, test = [], []
    for dct in data:
        if (dct['week'] in weeks and dct['media'] in media 
            and dct['class'] in classes and dct['extraction_type'] in ext_types):
            train.append(dct)
        else:
            test.append(dct)
    return train, test

# --- Vectorization using your preprocessing pipeline ---
def vectorize_data(samples):
    """
    Convert a list of sample dictionaries into numerical feature vectors by applying
    the preprocessing pipeline on each SpectrumObject.
    """
    X = []
    y = []
    # For each sample, get the spectrum adn the label.
    for sample in samples:
        X.append(sample['spectrum'].intensity)
        y.append(sample['class'])
    X = np.array(X)
    y = np.array(y)
    return X, y

# --- Baseline: One-vs-Rest RF per ribotype using real preprocessed data ---
def baseline_one_vs_rest_rf(train_samples, K=10, step_bin=3, offset=2000):
    """
    Filter to week 1, media 'Medio Sc', NoPE samples and then for each ribotype
    train a one-vs-rest RandomForest. Extract the top K features (biomarkers)
    based on feature importance. Convert feature index to real Dalton:
         real_Da = (bin_index * step_bin) + offset.
    Returns a dict mapping ribotype to a list of tuples:
         (feature index, real Dalton value, average intensity for that feature).
    """
    X, y_all = vectorize_data(train_samples)
    baseline_biomarkers = {}
    for ribo in CLASSES:
        # Create binary labels: 1 if sample belongs to the current ribotype, else 0.
        y_binary = (y_all == ribo).astype(int)
        if np.sum(y_binary) < 2:
            print(f"Not enough samples for {ribo}, skipping baseline training.")
            continue
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y_binary)
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:K]
        biomarkers = []
        for i in top_idx:
            real_Da = i * step_bin + offset
            avg_intensity = np.mean(X[:, i])
            biomarkers.append((i, real_Da, avg_intensity))
        baseline_biomarkers[ribo] = biomarkers
    return baseline_biomarkers


import matplotlib.ticker as ticker

def evaluate_variability(all_samples, baseline_biomarkers, eval = "eval_media", window=50, step_bin=3, offset=2000):
    """
    Evaluate variability on what's asked on EVAL.
    For each ribotype and for each baseline biomarker, generate a MALDI plot with a zoom of ±window bins.
    The x-axis is in Dalton: (bin index * step_bin + offset).
    
    Modifications:
      1. Only the mean of samples positive for the ribotype (i.e. samples where sample['class'] == ribo) are averaged.
      2. A vertical band is drawn at the baseline biomarker position, corresponding to (feat_idx * step_bin + offset)
         with a width of 3 Dalton (i.e. ±1.5 Da).
      3. Grid lines are added every 1 Dalton along the x-axis.
    """
    ############## modify this to do the followign, sometimes the user will ask eval= [semana 1 vs semana 2 vs semana 3], other eval = [medio A vs medio B]
    if eval == "eval_media":
        EVAL = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'Medio Fa']
        eval_samples = [d for d in all_samples if d['week'] == 'Semana 1' 
                    and d['extraction_type'] == 'NoPE' 
                    and d['media'] in EVAL]
    if eval == "eval_time":
        EVAL = ['Semana 1', 'Semana 2', 'Semana 3']
        eval_samples = [d for d in all_samples if d['media'] == 'Medio Ch' 
                    and d['extraction_type'] == 'NoPE' 
                    and d['week'] in EVAL]
    if eval == "eval_hosp":
        EVAL = ['Medio Ch', 'GU']
        eval_samples = [d for d in all_samples if d['week'] == 'Semana 1' 
                    and d['extraction_type'] == 'NoPE' 
                    and d['media'] in EVAL]
    if eval == "eval_pe":
        EVAL = ['Medio Ch', 'Chx']
        eval_samples = [d for d in all_samples if d['week'] == 'Semana 1'
                    and d['media'] in EVAL]
    
    # Filter week 1, NoPE samples from evaluation media
    
    # Organize samples by ribotype and media; only positive samples for that ribotype are grouped.
    data_by_media = {ribo: {m: [] for m in EVAL} for ribo in CLASSES}
    for sample in eval_samples:
        ribo = sample['class']
        media = sample['media']
        if ribo in CLASSES and media in EVAL:
            data_by_media[ribo][media].append(sample)
    
    for ribo in CLASSES:
        if ribo not in baseline_biomarkers:
            continue
        # Loop through each baseline biomarker for the ribotype.
        for feat_idx, real_Da_center, _ in baseline_biomarkers[ribo]:
            plt.figure(figsize=(8, 5))
            
            # Optionally, determine x-axis limits based on one available media group.
            x_min = max(0, feat_idx - window) * step_bin + offset
            x_max = None
            for media in EVAL_MEDIA:
                if data_by_media[ribo][media]:
                    X_eval, _ = vectorize_data(data_by_media[ribo][media])
                    n_features = X_eval.shape[1]
                    x_max = min(n_features, feat_idx + window) * step_bin + offset
                    break
            if x_max is None:
                continue  # No samples found for this ribotype
            
            # For each media, compute the mean intensity from samples positive for the ribotype.
            for media in EVAL_MEDIA:
                samples = data_by_media[ribo][media]
                if not samples:
                    continue
                X_eval, _ = vectorize_data(samples)
                n_samples, n_features = X_eval.shape
                start = max(0, feat_idx - window)
                end = min(n_features, feat_idx + window)
                bins_range = np.arange(start, end)
                x_axis = bins_range * step_bin + offset
                segment = X_eval[:, start:end]
                avg_segment = np.mean(segment, axis=0)
                plt.plot(x_axis, avg_segment, label=media)
            
            # Draw a vertical band at the baseline biomarker position.
            baseline_line = feat_idx * step_bin + offset
            # Draw a band spanning ±1.5 Da (i.e. a total width of 3 Da)
            plt.axvspan(baseline_line - 1.5, baseline_line + 1.5, color='gray', alpha=0.5, label='Baseline biomarker')
            
            plt.title(f"Variability for {ribo} biomarker at {real_Da_center:.1f} Da (bin {feat_idx})")
            plt.xlabel("Dalton (Da)")
            plt.ylabel("Intensity (a.u.)")
            
            # Set x-axis grid lines at every 1 Dalton.
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            # Draw grid lines for the minor ticks (every 1 Da).
            ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45)
            
            plt.legend()
            plt.tight_layout()
            # Check folder exists, if not create it
            folder_path = f"results/mario/{eval}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            filename = f"results/mario/{eval}/variability_{ribo}_biomarker_{feat_idx}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved variability plot: {filename}")
            
import pandas as pd
from scipy.stats import ttest_ind

# --- Helper Functions ---

def compute_peak_centroid_window(vector, feat_idx, tolerance=1, step_bin=3, offset=2000):
    """
    Compute the weighted centroid of the peak in a narrow window.
    The window covers bins from (feat_idx - tolerance) to (feat_idx + tolerance) inclusive.
    Dalton conversion: Dalton = (bin_index * step_bin) + offset.
    """
    n_features = len(vector)
    start = max(0, feat_idx - tolerance)
    end = min(n_features, feat_idx + tolerance + 1)  # +1 because slicing is exclusive
    bins_range = np.arange(start, end)
    intensities = vector[start:end]
    max_int = np.argmax(intensities)
    n_feature_centroid = bins_range[max_int]
    centroid = (n_feature_centroid * step_bin) + offset
    return centroid

def compute_peak_intensity_window(vector, feat_idx, tolerance=1):
    """
    Return the maximum intensity within the narrow window from 
    (feat_idx - tolerance) to (feat_idx + tolerance) inclusive.
    """
    n_features = len(vector)
    start = max(0, feat_idx - tolerance)
    end = min(n_features, feat_idx + tolerance + 1)
    intensities = vector[start:end]
    return np.max(intensities)

# --- Main Function for T-tests Across All Ribotypes ---

def run_t_tests_for_all_biomarkers(all_samples, baseline_biomarkers,
                                   tolerance=1, step_bin=3, offset=2000,
                                   output_csv='t_test_results.csv'):
    """
    For each ribotype (using its baseline biomarkers from training on week 1, Medio Sc),
    compute, for each sample positive for that ribotype (week 1, NoPE), the peak centroid and
    the peak intensity in a narrow window (±1 bin, i.e. 3 Da tolerance) around the biomarker.
    
    Then, for each ribotype and for each baseline biomarker (i.e. a given bin index),
    compare the metric (centroid and intensity) in baseline media ("Medio Sc") with that in every
    other media (as defined in EVAL_MEDIA) via a two-sample t-test.
    
    The function stores one row per ribotype, per biomarker, per compared media in a CSV file.
    """
    results = []
    
    # Loop over each ribotype for which baseline biomarkers were computed.
    for ribo in baseline_biomarkers:
        # For each baseline biomarker (each tuple: (feat_idx, real_Da, avg_intensity))
        for biomarker in baseline_biomarkers[ribo]:
            feat_idx, real_Da, avg_intensity_biomarker = biomarker
            # Dictionaries to store the computed centroids and intensities per media.
            centroids_by_media = {media: [] for media in EVAL_MEDIA}
            intensities_by_media = {media: [] for media in EVAL_MEDIA}
            
            # Consider only week 1, NoPE samples for the current ribotype.
            for sample in all_samples:
                if (sample['week'] == 'Semana 1' and 
                    sample['extraction_type'] == 'NoPE' and 
                    sample['media'] in EVAL_MEDIA and 
                    sample['class'] == ribo):

                    vector = sample['spectrum'].intensity
                    centroid = compute_peak_centroid_window(vector, feat_idx,
                                                            tolerance=tolerance,
                                                            step_bin=step_bin,
                                                            offset=offset)
                    peak_intensity = compute_peak_intensity_window(vector, feat_idx,
                                                                    tolerance=tolerance)
                    if not np.isnan(centroid):
                        centroids_by_media[sample['media']].append(centroid)
                        intensities_by_media[sample['media']].append(peak_intensity)
            
            # Use "Medio Sc" as the baseline.
            baseline_media = "Medio Sc"
            baseline_centroids = np.array(centroids_by_media[baseline_media])
            baseline_intensities = np.array(intensities_by_media[baseline_media])
            if len(baseline_centroids) == 0:
                print(f"No baseline samples for ribotype {ribo} biomarker at bin {feat_idx}. Skipping.")
                continue
            
            # Compare baseline with each other media.
            for media in EVAL_MEDIA:
                if media == baseline_media:
                    continue
                media_centroids = np.array(centroids_by_media[media])
                media_intensities = np.array(intensities_by_media[media])
                if len(media_centroids) == 0:
                    continue  # Skip if no samples in this media.
                
                # T-test for the centroid (peak position shift) comparison.
                t_stat_cent, p_val_cent = ttest_ind(baseline_centroids, media_centroids, nan_policy='raise')
                # T-test for the peak intensity (height) comparison.
                t_stat_int, p_val_int = ttest_ind(baseline_intensities, media_intensities, nan_policy='raise')
                              
                # Mark as statistically different if either p-value is below 0.05.
                stat_diff = (p_val_cent < 0.05) or (p_val_int < 0.05)
                
                # In case of a significant difference, print the results, add an extra column saying "displaced peak" or "different intensity"
                if stat_diff:
                    # if both thinks are different say the biggest
                    if p_val_cent < 0.05 and p_val_int < 0.05:
                        if p_val_cent < p_val_int:
                            text = "shifted peak"
                        else:
                            text = "different intensity"
                    elif p_val_cent < 0.05 and p_val_int > 0.05:
                        text = "shifted peak"
                    else:
                        text = "different intensity"
                else:
                    text = "no significant difference"
                
                # Append the results.
                results.append({
                    'ribotype': ribo,
                    'feat_idx': feat_idx,
                    'real_Da': real_Da,
                    'baseline_media': baseline_media,
                    'compared_media': media,
                    'n_baseline_samples': len(baseline_centroids),
                    'n_compared_samples': len(media_centroids),
                    'centroid_t_stat': t_stat_cent,
                    'centroid_p_value': p_val_cent,
                    'intensity_t_stat': t_stat_int,
                    'intensity_p_value': p_val_int,
                    'Statistically_different_biomarker?': stat_diff,
                    'description': text
                })
    
    # Save results to a CSV file.
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved t-test results to {output_csv}")

# --- Main execution ---
if __name__ == '__main__':
    print("Reading all data...")
    all_data = read_all_data()
    
    # Preprocess all data using the pipeline
    for d in all_data:
        if d['spectrum'].intensity.sum() == 0:
            print(f"Warning: zero intensity spectrum found for {d['id_label']}. Skipping.")
            # Remove it from the list
            all_data.remove(d)
        d['spectrum'] = preprocess_pipeline(d['spectrum'])
    
    # Parameters
    number_biomarkers = 20
    window = 10 # how much bins +/- from the biomarker to plot
    
    
    # STEP 1: Baseline training using week 1, media 'Medio Sc', extraction 'NoPE'
    baseline_train = [d for d in all_data if d['week'] == 'Semana 1' 
                      and d['media'] == 'Medio Sc' 
                      and d['extraction_type'] == 'NoPE']
    print(f"Baseline training samples: {len(baseline_train)}")
    baseline_biomarkers = baseline_one_vs_rest_rf(baseline_train, K=number_biomarkers, step_bin=binning_step, offset=2000)
    
    run_t_tests_for_all_biomarkers(all_data, baseline_biomarkers,
                               tolerance=1, step_bin=3, offset=2000,
                               output_csv='t_test_results.csv')
    
    # # Step 3: Evaluate variability across different weeks (Medio Sc, NoPE)
    # evaluate_variability(all_data, baseline_biomarkers, eval="eval_time", window=window, step_bin=binning_step, offset=2000)

    # STEP 2: Evaluate variability across different media (week 1, NoPE)
    evaluate_variability(all_data, baseline_biomarkers, eval="eval_media", window=window, step_bin=binning_step, offset=2000)
    
    # Step 3: Evaluate variability across different extraction methods (week 1, Medio Ch)
    evaluate_variability(all_data, baseline_biomarkers, eval="eval_pe", window=window, step_bin=binning_step, offset=2000)
    
    # Step 4: Evaluate variability across different extraction methods (week 1, Medio Ch, con PE, Ch24h vs Ch48h). 
    evaluate_variability(all_data, baseline_biomarkers, eval="eval_extract", window=window, step_bin=binning_step, offset=2000)
    
    # Step 5: Evaluate variability across different hospitals (week 1 (new), Medio Ch, con PE, GU vs HGUGM)
    evaluate_variability(all_data, baseline_biomarkers, eval="eval_hosp", window=window, step_bin=binning_step, offset=2000)
    
    

