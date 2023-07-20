import numpy as np
import torch
import os
# import pyflann
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans as sk_KMeans
# from math import sqrt, floor


def from_numpy_to_torch(np_array, torch_device):
    return torch.from_numpy(np_array).to(torch_device)

def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()


def eucl_KM_plspls_init(distributions, k):
    float_epsilon = 2.220446049250313e-16

    random_id = np.random.choice(len(distributions))
    centers = [distributions[random_id]]

    _distance = np.array(euclidean_norm(distributions, np.asarray(centers[0])))

    # Tackle infinity distance
    infidx = np.isinf(_distance)
    idx = np.logical_not(infidx)
    _distance[infidx] = _distance[idx].max()

    while len(centers) < k:
        p = _distance ** 2
        p /= p.sum() + float_epsilon

        random_id_wrt_p = np.random.choice(len(distributions), p=p)
        centers.append(distributions[random_id_wrt_p])

        _distance = np.minimum(_distance,
                               euclidean_norm(distributions, np.asarray(centers[-1])))

    return np.asarray(centers)


# Enables to not assume that old class labels are assigned with the first int labels
def unique(lbls):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in lbls:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return np.sort(unique_list)


def eucl_ssKM_plspls_init(distributions, k, lab_lbls, msk_lb):
    float_epsilon = 2.220446049250313e-16
    sup_dist = distributions[msk_lb]
    unsup_dist = distributions[~msk_lb]
    unique_lbls = unique(lab_lbls)
    centers = []
    # 1. Init old class centroids using labelled set
    for k_id in range(len(unique_lbls)):
        current_class_lbl = unique_lbls[k_id]
        mean_ctr = np.asarray(sup_dist[lab_lbls == current_class_lbl]).mean(0)
        centers.append(mean_ctr)
        if k_id == 0:
            _distance = np.array(euclidean_norm(unsup_dist, np.asarray(centers[0])))
        else:
            _distance = np.minimum(_distance,
                                   euclidean_norm(unsup_dist, np.asarray(centers[-1])))

    # 2. Init new class centroids using kmeans++
    while len(centers) < k:
        # km++
        p = _distance ** 2
        p /= p.sum() + float_epsilon
        random_id_wrt_p = np.random.choice(len(unsup_dist), p=p)
        centers.append(unsup_dist[random_id_wrt_p])

        _distance = np.minimum(_distance,
                               euclidean_norm(unsup_dist, np.asarray(centers[-1])))
    return np.asarray(centers)


def euclidean_norm(X, mu):
    return np.linalg.norm(X - mu.T, axis=1)

def torch_euclidean_norm(X, mu):
    return torch.linalg.norm(X - mu.T, axis=1)

def torch_cosine_sim(X, mu):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(X, mu.T)


def clustering(full_set,
               lab_pts_lbls,
               mask_lbls,
               iters=100,
               n_clusters=10,
               is_unbiased=False,
               input_init_centroids=None,
               init_strategy="semi_sup_kmeans_plusplus_init",
               lambda_param=0.,
               distortion_metric="eucl",
               device_name='cuda:0',
               do_semi_sup_clust = True):
    
    torch_device = torch.device(device_name)
    points_dim = len(full_set[0])
    float_epsilon = 2.220446049250313e-16
    best_inertia = None
    if distortion_metric == "eucl":
        torch_distortion_metric_function = torch_euclidean_norm
        arg_opt = torch.argmin
    elif distortion_metric == "cosine_sim":
        torch_distortion_metric_function = torch_cosine_sim
        arg_opt = torch.argmax

    # Conversions npy to pytorch
    torch_x_pred = from_numpy_to_torch(full_set, torch_device)
    torch_lab_pts_lbls = from_numpy_to_torch(lab_pts_lbls, torch_device)
    torch_mask_lbls = from_numpy_to_torch(mask_lbls, torch_device)

    # Clustering parameters
    torch_all_mus = torch.zeros((n_clusters, points_dim), dtype=torch.float, device=torch_device)

    # Balancing weights (only used when is_unbiased=True)
    torch_estim_weights = torch.ones(n_clusters, device=torch_device) / n_clusters

    prev_assign = torch.zeros(len(torch_x_pred), dtype=torch.long, device=torch_device).type_as(torch_lab_pts_lbls)
    #
    torch_all_dist_estims = torch.zeros((len(torch_x_pred), n_clusters), dtype=torch.float, device=torch_device)
    for it in range(0, iters):
        # print("it: ", it)

        # Parameters estimation
        all_mus = None
        if it == 0:
            if init_strategy == "random_init":
                all_mus = np.array([full_set[i] for i in np.random.randint(len(full_set), size=n_clusters)])
            elif init_strategy == "kmeans_plusplus_init":
                all_mus = eucl_KM_plspls_init(full_set, n_clusters)
            elif init_strategy == "semi_sup_kmeans_plusplus_init":
                all_mus = eucl_ssKM_plspls_init(full_set, n_clusters, lab_pts_lbls, mask_lbls)
            elif init_strategy == "use_input_centroids" and (input_init_centroids is not None):
                all_mus = input_init_centroids
            else:
                print("init_strategy: ", init_strategy, " does not exist.")

            all_mus = np.asarray(all_mus)
            torch_all_mus = from_numpy_to_torch(all_mus, torch_device)

        else:
            if do_semi_sup_clust == True:
                torch_labels[torch_mask_lbls] = torch_lab_pts_lbls
            for cl_id in range(0, n_clusters):
                mask = torch_labels.eq(cl_id)
                sum_mask = torch.sum(mask)
                torch_all_mus[cl_id] = torch.sum((mask * torch_x_pred.t()), dim=1) / sum_mask

        # Points assignment with clusters
        for cl_id in range(0, n_clusters):
            torch_cl_dists = torch_distortion_metric_function(torch_x_pred, torch_all_mus[cl_id])
            torch_all_dist_estims[:, cl_id] = torch_cl_dists
        torch_all_dist_estims = torch_all_dist_estims + float_epsilon
        if is_unbiased == False:
            dists = torch_all_dist_estims
        else:
            dists = torch_all_dist_estims - lambda_param * torch_estim_weights * torch.log(
                torch_estim_weights + float_epsilon)
        torch_labels = arg_opt(dists, axis=1).type_as(torch_lab_pts_lbls)
        if distortion_metric == "cosine_sim":
            u_inertia = -torch.max(dists[~torch_mask_lbls], 1)[0].sum()
            l_inertia = -torch.max(dists[torch_mask_lbls], 1)[0].sum()
            inertia = u_inertia + l_inertia
        elif distortion_metric == "eucl":
            u_inertia = torch.min(dists[~torch_mask_lbls], 1)[0].sum()
            l_inertia = torch.min(dists[torch_mask_lbls], 1)[0].sum()
            inertia = u_inertia + l_inertia

        # Balancing weights estimation
        for cluster_id in range(0, n_clusters):
            torch_estim_weights[cluster_id] = (torch_labels == cluster_id).sum()
        torch_estim_weights = torch_estim_weights / torch.sum(torch_estim_weights)

        # check convergence
        if best_inertia is None or inertia < best_inertia:
            best_torch_labels = torch_labels.clone()
            best_inertia = inertia
        if torch.allclose(torch_labels, prev_assign) and it >= 1:
            #print('k-means converged in %d iterations' % (it + 1))
            break
        prev_assign = torch_labels.clone()
        
    return from_torch_to_numpy(best_inertia), from_torch_to_numpy(best_torch_labels), from_torch_to_numpy(
        dists), from_torch_to_numpy(torch_estim_weights), from_torch_to_numpy(torch_all_mus)
