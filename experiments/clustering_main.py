from dataloader.MaldiDataset import MaldiDataset
import numpy as np
from dataloader.preprocess import PIKEScaleNormalizer, NoiseRemoval, SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer
import matplotlib.pyplot as plt
from dataloader.e_pike import KernelPipeline
import h5py
from sklearn.cluster import SpectralClustering, KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score



data_path = "/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/MaldiMaranonDB"


# Preprocessing pipeline
binning_step = 3
preprocess_pipeline = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                Smoother(halfwindow=10),
                                BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                Trimmer(),
                                Binner(step=binning_step),
                                Normalizer(sum=1), 
                                NoiseRemoval(), # New technique by Rafa
                                PIKEScaleNormalizer() # all peaks >= 1
                                )


# Check fi complete_kernel.h5 exists
try:
    with h5py.File("complete_kernel.h5", "r") as h5f:
        complete_kernel = h5f["complete_kernel"][:]
        print("Kernel already computed")
except:
    print("Kernel not computed yet")
    complete_kernel = None

if complete_kernel is None:
    # Create dataset
    dataset = MaldiDataset(data_path, preprocess_pipeline, genus="Escherichia", species="Coli")

    dataset.parse_dataset()

    # Get spectral data
    data = dataset.get_data()

    # get data per year
    data_per_year = {}
    for d in data:
        year = d['year_label']
        if year not in data_per_year:
            data_per_year[year] = []
        data_per_year[year].append(d)
        
    # Select only data from 2018 and 2019
    data = data_per_year['2018'] + data_per_year['2019']

    # Remove from data all samples that have lesss than 6000 spectrum_intensities
    data = [d for d in data if len(d['spectrum_intensity']) == 6000]


    # Prepare data for PIKE
    kp = KernelPipeline()

    # Compute the complete kernel
    complete_kernel = kp._no_peak_removal(data, th=1e-4)

    # Save kernel to file in h5 format
    with h5py.File("complete_kernel.h5", "w") as h5f:
        h5f.create_dataset("complete_kernel", data=complete_kernel)
        
    # Compute the masked peaks (before kernel computaiton) kernel
    masked_peaks_kernel = kp._masked_peaks(data, th=1e-4)
    # Save kernel to file in h5 format
    with h5py.File("masked_peaks_kernel.h5", "w") as h5f:
        h5f.create_dataset("masked_peaks_kernel", data=masked_peaks_kernel)
        
    # Compute the masked peaks on kernel space
    spr_kernel = kp._spr(data, th=1e-4)
    # Save kernel to file in h5 format
    with h5py.File("spr_kernel.h5", "w") as h5f:
        h5f.create_dataset("spr_kernel", data=spr_kernel)
    
    
        

def cluster_data(kernel, n_clusters, plot_samples_per_cluster=True, key='', model='SC'):
    # Step 1: Perform Spectral Clustering or KMeans
    if model == 'SC':
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=10)
    elif model == 'SCRBF':
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=10)
    elif model == 'kMeans':
        spectral = KMeans(n_clusters=n_clusters, n_init=5, random_state=10)
    
    spectral.fit(kernel)
    labels_sc = spectral.labels_

    # Step 2: Reorder cluster ids from more to less common
    unique, counts = np.unique(labels_sc, return_counts=True)
    cluster_counts = np.array(list(zip(unique, counts)))
    sorted_clusters = cluster_counts[cluster_counts[:, 1].argsort()[::-1], 0]
    new_cluster_mapping = {old: new for new, old in enumerate(sorted_clusters)}
    labels_sc = np.array([new_cluster_mapping[cluster] for cluster in labels_sc])

    # Step 3: Use KMeans to compute centroids
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    kmeans.fit(kernel)  # Using the precomputed kernel for KMeans
    centroids = kmeans.cluster_centers_

    # Step 4: Calculate distances and sort IDs within each cluster
    sorted_ids_per_cluster = {}

    for cluster_idx in range(n_clusters):
        # Get the points belonging to this cluster
        cluster_mask = labels_sc == cluster_idx  # Create a boolean mask for current cluster
        cluster_points = kernel[cluster_mask]
        # cluster_ids = np.array(ids)[cluster_mask]  # Get corresponding IDs

        # Compute the Euclidean distance to the centroid
        centroid = centroids[cluster_idx].reshape(1, -1)
        distances = cdist(cluster_points, centroid, metric='euclidean').flatten()

        # Get the sorted indices based on distance
        sorted_indices = np.argsort(distances)

        # Store the sorted IDs for this cluster
        # sorted_ids_per_cluster[f"Cluster {cluster_idx}"] = list(cluster_ids[sorted_indices])

    # Step 5: Samples per cluster
    if plot_samples_per_cluster:
        unique_labels, counts = np.unique(labels_sc, return_counts=True)
        plt.figure(figsize=(15, 6))
        bars = plt.bar(unique_labels, counts)

        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), count, ha='center', va='bottom')

        plt.xticks(range(n_clusters))
        plt.yscale('log')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        plt.title(f'Number of Samples per Cluster: {key}')
        plt.show()
        
    return labels_sc # ,sorted_ids_per_cluster


# results for complete kernel

# First, plot the kernel
plt.figure(figsize=(10, 10))
plt.imshow(complete_kernel, cmap='viridis')
plt.title('Complete Kernel')
plt.colorbar()
plt.savefig('results/surveillance/complete_kernel.png')
plt.show()

n_clusters_list = np.arange(2, 20)

complete_cluster_sh = {key: [] for key in ['complete']}
for n_clusters in n_clusters_list:
    key = 'complete'
    kernel = complete_kernel

    # Cluster data
    labels_sc = cluster_data(kernel, n_clusters, plot_samples_per_cluster=False)

    # Convert kernel to distance
    diag_elements = np.diag(kernel)
    distance_kernel = np.sqrt(diag_elements[:, np.newaxis] + diag_elements[np.newaxis, :] - 2 * kernel)

    # Compute silhouette score
    sh = silhouette_score(distance_kernel, labels_sc, metric='precomputed')
    complete_cluster_sh[key].append(sh)

fig, ax = plt.subplots(figsize=(10, 6))
ax.grid()
ax.set_xlabel('Number of Clusters', fontsize=13)
for key, value in complete_cluster_sh.items():
    ax.plot(n_clusters_list, value, label='PIKE complete', marker='o')
    ax.set_xticks(n_clusters_list)
    ax.legend(loc='best')
