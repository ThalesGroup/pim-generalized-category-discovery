from tqdm import tqdm
import numpy as np
from models.clustering_methods import torch_semi_supervised_kmeans as ssKM


def ssKM_protos_init(feature_map_preds, assumed_k, lab_subset_labels, mask_lab, 
                      mask_old_new, centroids_init_nbr, device_name, 
                      protos_file_name):
    
    all_input_init_centroids = []
    for init_centr_it in tqdm(range(0, centroids_init_nbr)):
        all_input_init_centroids.append(ssKM.eucl_ssKM_plspls_init(feature_map_preds, assumed_k, lab_subset_labels, mask_lab))
    all_input_init_centroids = np.asarray(all_input_init_centroids)

    best_inertia = None
    print("ssKM clustering...")
    for centr_it in tqdm(range(0, centroids_init_nbr)):
        input_init_centroids = all_input_init_centroids[centr_it]
        inertia, clustering_labels, _, _, all_mus = ssKM.clustering(feature_map_preds,
                                                                    lab_subset_labels,
                                                                    mask_lab,
                                                                    n_clusters=assumed_k,
                                                                    input_init_centroids=input_init_centroids,
                                                                    init_strategy='use_input_centroids',
                                                                    device_name=device_name)
        if best_inertia is None or inertia < best_inertia:
            best_clustering_labels = clustering_labels.copy()
            best_inertia = inertia
            best_all_mus = all_mus.copy()
    
    with open(protos_file_name, 'wb') as f:
        np.save(f, best_all_mus)
        
    print("ssKM clustering completed. \n")
    return best_all_mus