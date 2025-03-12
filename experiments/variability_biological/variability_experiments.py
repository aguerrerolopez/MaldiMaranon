import os
import numpy as np
import pandas as pd
import pymzml
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import matplotlib.ticker as ticker
from pyopenms import MSSpectrum, SpectrumAlignment


from dataloader.preprocess import (
    SequentialPreprocessor,
    VarStabilizer,
    Smoother,
    BaselineCorrecter,
    Trimmer,
    Binner,
    Normalizer,
    IntensityThresholding,
    detect_peaks,
    Aligner
)
from dataloader.SpectrumObject import SpectrumObject
CLASSES = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']

# --- Helper Functions ---


def align_spectrum_to_reference(target_spec_obj, ref_spec_obj, tolerance=0.002):
    """
    Align a target spectrum to a reference spectrum using pyOpenMS's SpectrumAlignment.
    
    Parameters
    ----------
    target_spec_obj : SpectrumObject
        The spectrum to be aligned.
    ref_spec_obj : SpectrumObject
        The reference spectrum (e.g. the mean baseline spectrum).
    tolerance : float, optional
        Tolerance in Dalton for matching peaks, by default 0.002.
    
    Returns
    -------
    SpectrumObject
        A new SpectrumObject with the m/z axis warped to align with the reference.
    """
    # Convert spectra to pyOpenMS MSSpectrum objects.
    target_ms = spectrumobject_to_msspectrum(target_spec_obj)
    ref_ms = spectrumobject_to_msspectrum(ref_spec_obj)
    
    aligner = SpectrumAlignment()
    alignment = []
    aligner.getSpectrumAlignment(alignment, target_ms, ref_ms)
    
    if len(alignment) == 0:
        # If no matches, return the original spectrum.
        return target_spec_obj

    # Extract matched m/z values.
    # get_peaks() returns a tuple (mz_array, intensity_array)
    target_peaks = target_ms.get_peaks()[0]
    ref_peaks = ref_ms.get_peaks()[0]
    
    matched_target_mz = np.array([target_peaks[i] for i, _ in alignment])
    matched_ref_mz = np.array([ref_peaks[j] for _, j in alignment])
    
    # Compute a simple linear mapping: f(x) = a * x + b via linear regression.
    a, b = np.polyfit(matched_target_mz, matched_ref_mz, 1)
    
    # Apply the transformation to the target spectrum's m/z values.
    new_mz = a * target_spec_obj.mz + b
    
    # Create a new SpectrumObject with the warped m/z axis.
    aligned_spec = SpectrumObject(mz=new_mz, intensity=target_spec_obj.intensity)
    return aligned_spec


def spectrumobject_to_msspectrum(spec_obj):
    """
    Convert a SpectrumObject (with .mz and .intensity as numpy arrays)
    into a pyOpenMS MSSpectrum.
    """
    spec = MSSpectrum()
    # pyOpenMS expects a tuple (mz_array, intensity_array)
    # Make sure arrays are of type float
    spec.set_peaks((spec_obj.mz.astype(float), spec_obj.intensity.astype(float)))
    return spec

def msspectrum_to_spectrumobject(ms_spec, original_intensity=None):
    """
    Convert a pyOpenMS MSSpectrum to a SpectrumObject.
    If original_intensity is provided, use it instead of ms_spec's intensity.
    """
    # Get peaks: a tuple (mz_array, intensity_array)
    mz_arr, intensity_arr = ms_spec.get_peaks()
    # Here we assume SpectrumObject is a constructor that takes mz and intensity.
    # If you want to keep original intensity values (e.g., not warped), pass original_intensity.
    intensity = original_intensity if original_intensity is not None else intensity_arr
    return SpectrumObject(mz=np.array(mz_arr), intensity=np.array(intensity))



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
    WEEKS = ['Semana 1', 'Semana 2', 'Semana 3']
    
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
    MEDIA_NOPE = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU', 'Medio Fa']
    MEDIA_PE   = ['Chx', 'Brx', 'Clx', 'Scx', 'GUx', 'Fax']
    if os.path.isdir(no_ext):
        all_data += _read_folder(no_ext, MEDIA_NOPE, 'NoPE')
    if os.path.isdir(pe_ext):
        all_data += _read_folder(pe_ext, MEDIA_PE, 'PE')
    return  all_data

def get_samples(all_samples, media, week, extraction_type):
    """Filter samples by media, week, and extraction_type."""
    return [d for d in all_samples if d['media'] == media and d['week'] == week and d['extraction_type'] == extraction_type]

def vectorize_data(samples):
    """
    Convert a list of sample dictionaries into a data matrix X and label vector y.
    Each sample is assumed to be a dict with keys 'intensity' (a numpy array) and 'class' (the ribotype).
    """
    X = np.vstack([sample['spectrum'].intensity for sample in samples])
    y = np.array([sample['class'] for sample in samples])
    return X, y

def compute_centroid(segment):
    """
    Compute the centroid (weighted average bin index) for each sample in the segment.
    A small epsilon is added to the denominator to avoid division by zero.
    """
    bins = np.arange(segment.shape[1])
    return np.sum(segment * bins, axis=1) / (np.sum(segment, axis=1) + 1e-9)

# --- Baseline Model Training and Biomarker Extraction ---

def train_baseline_model(all_samples, n_biomarkers=20, bin_size=3, media="Medio Sc", week="Semana 1", extraction="NoPE"):
    """
    Train a separate Random Forest classifier for each ribotype in a one-vs-rest manner.
    Baseline samples are those from media 'Medio Sc', week 'Semana 1', and extraction 'NoPE'.
    
    For each ribotype, the function extracts the top n_biomarkers features based on feature importance.
    
    Parameters
    ----------
    all_samples : list
        List of sample dictionaries.
    n_biomarkers : int, optional
        Number of biomarkers (features) to extract for each ribotype, by default 20.
    bin_size : int, optional
        The step factor for converting a feature index to Dalton, by default 3.
    
    Returns
    -------
    rf_models : dict
        A dictionary mapping ribotype to its trained Random Forest classifier.
    biomarkers_info : dict
        A dictionary mapping ribotype to a list of tuples (feature_index, peak_Da)
        representing the top biomarkers for that ribotype.
    """
    # Filter baseline samples: media 'Medio Sc', week 'Semana 1', extraction 'NoPE'
    baseline_samples = get_samples(all_samples, media=media, week=week, extraction_type=extraction)
    X, y = vectorize_data(baseline_samples)
    
    # Determine the unique ribotypes in the baseline data.
    ribotypes = np.unique(y)
    
    rf_models = {}
    biomarkers_info = {}
    
    step_bin = bin_size
    offset = 2000
    
    # Train a one-vs-rest RF for each ribotype.
    for ribo in ribotypes:
        # Create binary labels: 1 if the sample's class equals the ribotype, else 0.
        y_binary = (y == ribo).astype(int)
        
        # Train the RF model for the current ribotype.
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y_binary)
        rf_models[ribo] = rf
        
        # Get feature importances and extract top n_biomarkers feature indices.
        importances = rf.feature_importances_
        top_features_idx = np.argsort(importances)[-n_biomarkers:]
        
        # Store biomarker info (feature index and its corresponding Dalton value) for this ribotype.
        biomarkers_info[ribo] = []
        for idx in top_features_idx:
            peak_Da = idx * step_bin + offset
            biomarkers_info[ribo].append((idx, peak_Da))
    
    return rf_models, biomarkers_info

# --- Variability t-Test Experiments ---

def variability_t_tests(all_samples, biomarkers_info, window=10, significance=0.05, baseline_media="Medio Sc", baseline_week="Semana 1", baseline_extraction="NoPE"):
    """
    For each ribotype (RTX) and for each biomarker (given by its feature bin and corresponding Da),
    run t-tests comparing the baseline condition (Medio Sc, Semana 1, NoPE) to all variability conditions.
    
    Variability conditions:
      1. Six media without protein extraction: ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU', 'Medio Fa']
      2. Six media with protein extraction: ['Chx', 'Brx', 'Clx', 'Scx', 'GUx', 'Fax']
      3. Three timepoints: ['Semana 1', 'Semana 2', 'Semana 3']
      
    For each condition, the t-test is performed on two aspects:
      - Intensity: mean intensity over the window.
      - Shift: computed as the centroid (weighted average index) of the window.
      
    The output CSV contains columns:
      RTX | Feature bin | Peak Da | Media | Time | p-value int | p-value shift | CODE
    """
    # Variability conditions.
    media_no_pe = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU', 'Medio Fa']
    media_pe = ['Chx', 'Brx', 'Clx', 'Scx', 'GUx', 'Fax']
    weeks = ['Semana 1', 'Semana 2', 'Semana 3']
    
    # Baseline: media 'Medio Sc', week 'Semana 1', extraction 'NoPE'
    baseline_samples = get_samples(all_samples, media=baseline_media, week=baseline_week, extraction_type=baseline_extraction)
    X_baseline, y_baseline = vectorize_data(baseline_samples)
    
    results = []
    
    # Iterate over each ribotype for which we have biomarkers.
    for ribo in biomarkers_info.keys():
        # Filter baseline samples for the current ribotype.
        baseline_idx = [i for i, label in enumerate(y_baseline) if label == ribo]
        if len(baseline_idx) == 0:
            continue
        X_base_ribo = X_baseline[baseline_idx]
        
        # For each biomarker defined for this ribotype.
        for feat_idx, peak_Da in biomarkers_info[ribo]:
            start = max(0, feat_idx - window)
            end = min(X_base_ribo.shape[1], feat_idx + window)
            baseline_segment = X_base_ribo[:, start:end]
            baseline_intensities = np.mean(baseline_segment, axis=1)
            baseline_centroids = compute_centroid(baseline_segment)
            
            # Loop over both media sets and weeks.
            for media_group in (media_no_pe + media_pe):
                for week in weeks:
                    # Determine extraction type: if media name ends with 'x', use protein extraction (PE), else NoPE.
                    extraction_type = 'PE' if media_group.endswith('x') else 'NoPE'
                    condition_samples = get_samples(all_samples, media=media_group, week=week, extraction_type=extraction_type)
                    if len(condition_samples) == 0:
                        continue
                    X_condition, y_condition = vectorize_data(condition_samples)
                    # Filter by current ribotype.
                    condition_idx = [i for i, label in enumerate(y_condition) if label == ribo]
                    if len(condition_idx) == 0:
                        continue
                    X_cond_ribo = X_condition[condition_idx]
                    # Ensure the window is valid in the condition samples.
                    if X_cond_ribo.shape[1] < end:
                        continue
                    condition_segment = X_cond_ribo[:, start:end]
                    condition_intensities = np.mean(condition_segment, axis=1)
                    condition_centroids = compute_centroid(condition_segment)
                    
                    # Perform t-tests.
                    t_stat_int, p_val_int = ttest_ind(baseline_intensities, condition_intensities, nan_policy='omit')
                    t_stat_cent, p_val_cent = ttest_ind(baseline_centroids, condition_centroids, nan_policy='omit')
                    
                    # Determine CODE based on significance and direction.
                    # Default code is 0 (no significant difference).
                    code = 0
                    intensity_significant = p_val_int < significance
                    shift_significant = p_val_cent < significance
                    
                    # Compute mean differences.
                    intensity_diff = np.mean(condition_intensities) - np.mean(baseline_intensities)
                    centroid_diff = np.mean(condition_centroids) - np.mean(baseline_centroids)
                    
                    # Code for intensity: 1 if condition intensity is higher, 1 if lower.
                    code_intensity = 1 if (intensity_significant and intensity_diff > 0) else -1 if intensity_significant else 0
                    # Code for shift: -1 if shifted to the right (centroid higher), -2 if shifted to the left.
                    code_shift = -1 if (shift_significant and centroid_diff > 0) else 1 if shift_significant else 0
                    
                    if intensity_significant and shift_significant:
                        # If both tests are significant, choose the one with lower p-value.
                        if p_val_int < p_val_cent:
                            code = code_intensity
                        else:
                            code = code_shift
                    elif intensity_significant:
                        code = code_intensity
                    elif shift_significant:
                        code = code_shift
                    else:
                        code = 0
                    
                    results.append({
                        'RTX': ribo,
                        'Feature bin': feat_idx,
                        'Peak Da': peak_Da,
                        'Media': media_group,
                        'Time': week,
                        'p-value int': p_val_int,
                        'p-value shift': p_val_cent,
                        'CODE I': code_intensity,
                        'CODE S': code_shift
                    })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("t_test_variability_results.csv", index=False)
    print("Saved t-test variability results to t_test_variability_results.csv")
    return df_results

def evaluate_variability(all_samples, baseline_biomarkers, eval="eval_media", window=50, step_bin=3, offset=2000, baseline_media="Medio Sc", baseline_week="Semana 1", baseline_extraction="NoPE"):
    """
    Evaluate variability based on the evaluation condition.
    For each ribotype and for each baseline biomarker, generate a MALDI plot with a zoom of ±window bins.
    The x-axis is in Dalton: (bin index * step_bin + offset).

    The function creates separate folders (inside results/mario/) for each evaluation condition:
      - "eval_media": For week="Semana 1", extraction type "NoPE", and media in ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'Medio Fa', 'GU']
      - "eval_pe": For week="Semana 1", extraction type "PE", and media in ['Chx', 'Brx', 'Clx', 'Scx', 'Fax', 'GUx']
      - "eval_time": For media="Medio Ch", extraction type "NoPE", and week in ['Semana 1', 'Semana 2', 'Semana 3']
      - "eval_hosp": For week="Semana 1", extraction type "NoPE", and media in ['Medio Ch', 'GU']
    """
    # Define evaluation conditions.
    if eval == "eval_media":
        EVAL = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'Medio Fa', 'GU']
        eval_samples = [d for d in all_samples if d['week'] == 'Semana 1'
                        and d['extraction_type'] == 'NoPE'
                        and d['media'] in EVAL]
    elif eval == "eval_time":
        EVAL = ['Semana 1', 'Semana 2', 'Semana 3']
        # For eval_time, fix media to 'Medio Ch' and extraction type to 'NoPE'.
        eval_samples = [d for d in all_samples if d['media'] == 'Medio Ch'
                        and d['extraction_type'] == 'NoPE'
                        and d['week'] in EVAL]
    elif eval == "eval_hosp":
        EVAL = ['Medio Ch', 'GU']
        eval_samples = [d for d in all_samples if d['week'] == 'Semana 1'
                        and d['extraction_type'] == 'NoPE'
                        and d['media'] in EVAL]
    elif eval == "eval_pe":
        EVAL = ['Chx', 'Brx', 'Clx', 'Scx', 'Fax', 'GUx']
        eval_samples = [d for d in all_samples if d['week'] == 'Semana 1'
                        and d['extraction_type'] == 'PE'
                        and d['media'] in EVAL]
    else:
        raise ValueError("Invalid eval condition. Choose one of: eval_media, eval_time, eval_hosp, eval_pe.")
    
    # Group samples by ribotype and evaluation variable.
    # For eval_time, the grouping key is week; for others, it is media.
    grouping_key = 'media' if eval != "eval_time" else 'week'
    data_by_group = {ribo: {grp: [] for grp in EVAL} for ribo in CLASSES}
    
    for sample in eval_samples:
        ribo = sample['class']
        grp_val = sample[grouping_key]
        if ribo in CLASSES and grp_val in EVAL:
            data_by_group[ribo][grp_val].append(sample)
    
    # Compute the baseline mean spectrum from media "Medio Sc", Semana 1, NoPE.
    baseline_samples = get_samples(all_samples, media=baseline_media, week=baseline_week, extraction_type=baseline_extraction)
    if baseline_samples:
        X_baseline, _ = vectorize_data(baseline_samples)
        baseline_mean = np.mean(X_baseline, axis=0)
        # Assume that all baseline spectra share the same m/z axis.
        baseline_mz = baseline_samples[0]['spectrum'].mz
    else:
        baseline_mean = None
    
    # Create folder for saving plots.
    folder_path = os.path.join("results", "mario", eval)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Generate a plot for each ribotype and for each baseline biomarker.
    for ribo in CLASSES:
        # Each entry in baseline_biomarkers for a ribotype is a tuple: (feat_idx, real_Da_center)
        for feat_idx, real_Da_center in baseline_biomarkers[ribo]:
            plt.figure(figsize=(8, 5))
            
            # Determine x-axis limits based on one available group.
            x_min = max(0, feat_idx - window) * step_bin + offset
            x_max = None
            for grp in EVAL:
                if data_by_group[ribo][grp]:
                    X_eval, _ = vectorize_data(data_by_group[ribo][grp])
                    n_features = X_eval.shape[1]
                    x_max = min(n_features, feat_idx + window) * step_bin + offset
                    break
            if x_max is None:
                plt.close()
                continue  # No samples found for this ribotype
            
            # For each group in the evaluation set, plot the mean intensity over the window with std shading.
            for grp in EVAL:
                samples = data_by_group[ribo][grp]
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
                std_segment = np.std(segment, axis=0)
                
                # Plot mean line and standard deviation as a shaded area.
                plt.plot(x_axis, avg_segment, label=str(grp))
                plt.fill_between(x_axis, avg_segment - std_segment, avg_segment + std_segment, alpha=0.3)
            
            # Force plot the baseline mean spectrum (if available) in blue, dotted line.
            if baseline_mean is not None:
                # Extract the corresponding window from the baseline spectrum.
                if len(baseline_mean) >= end:
                    baseline_segment = baseline_mean[start:end]
                    plt.plot(x_axis, baseline_segment, color='blue', linestyle='dotted', linewidth=2, label='baseline spectrum')
            
            # Draw a vertical band at the baseline biomarker position.
            baseline_line = feat_idx * step_bin + offset
            plt.axvspan(baseline_line - 1.5, baseline_line + 1.5, color='gray', alpha=0.5, label='Baseline biomarker')
            
            plt.title(f"Variability for {ribo} biomarker at {real_Da_center:.1f} Da (bin {feat_idx})")
            plt.xlabel("Dalton (Da)")
            plt.ylabel("Intensity (a.u.)")
            
            # Set x-axis grid lines: major every 3 Da and minor every 1 Da.
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45)
            
            plt.legend()
            plt.tight_layout()
            
            # Save the plot with the naming pattern "RT_{ribo}_biomarker_{feat_idx}.png"
            filename = os.path.join(folder_path, f"RT_{ribo}_biomarker_{feat_idx}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Saved variability plot: {filename}")
            


# --- Main Script Execution ---

if __name__ == '__main__':
    # Assume that `all_samples` is already loaded from your data source.
    all_samples = read_all_data()
    
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
    
    # Preprocess all data using the pipeline
    for d in all_samples[:]:
        if d['spectrum'].intensity.sum() == 0:
            print(f"Warning: zero intensity spectrum found for {d['id_label']}. Skipping.")
            all_samples.remove(d)
        else:
            d['spectrum'] = preprocess_pipeline(d['spectrum'])
            # After preprocessing, assert that the intensity size is 6000.
            assert d['spectrum'].intensity.size == 6000, f"Preprocessing failed for {d['id_label']}."

    # Parameters
    number_biomarkers = 20
    window = 10 # how much bins +/- from the biomarker to plot
    
    # Parameters for baseline selection.
    baseline_media = "Scx"
    baseline_week = "Semana 1"
    baseline_extraction = "PE"
    
    # Extract baseline samples.
    baseline_samples = get_samples(all_samples, media=baseline_media, week=baseline_week, extraction_type=baseline_extraction)
    all_ref_peaks = []
    mean_spectrum = np.mean(np.vstack([d['spectrum'].intensity for d in baseline_samples]), axis=0)
    mean_mz = np.mean(np.vstack([d['spectrum'].mz for d in baseline_samples]), axis=0)
    mean_baseline = SpectrumObject(mz=mean_mz, intensity=mean_spectrum)
   
    for d in all_samples:
        d['spectrum'] = align_spectrum_to_reference(d['spectrum'], mean_baseline, tolerance=0.002)
    
    
    # Train the baseline model (using SC media, Semana 1, NoPE) and extract biomarkers.
    rf_model, biomarkers_info = train_baseline_model(all_samples, n_biomarkers=number_biomarkers, bin_size=binning_step, media=baseline_media, week=baseline_week, extraction=baseline_extraction)
    
    
    # Run the variability t-test experiments and export results to CSV.
    results_df = variability_t_tests(all_samples, biomarkers_info, window=window, baseline_media=baseline_media, baseline_week=baseline_week, baseline_extraction=baseline_extraction)
    
    # evaluate_variability(all_samples, biomarkers_info, eval="eval_media", window=10, step_bin=binning_step, offset=2000)
    # evaluate_variability(all_samples, biomarkers_info, eval="eval_time", window=10, step_bin=binning_step, offset=2000)
    # evaluate_variability(all_samples, biomarkers_info, eval="eval_hosp", window=10, step_bin=binning_step, offset=2000)
    # evaluate_variability(all_samples, biomarkers_info, eval="eval_pe", window=10, step_bin=binning_step, offset=2000)

