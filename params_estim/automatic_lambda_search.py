import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from metrics import partitioning_metrics as p_metrics
from sklearn.cluster import KMeans 
from models.PIM import PIM_partitioner


def from_numpy_to_torch(np_array, torch_device):
    return torch.from_numpy(np_array).to(torch_device)

def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()

def lambda_search(path_auto_lambda, feature_map_preds, epochs, assumed_k, 
                  mask_lab, device_name, dataset, torch_device, 
                  lab_subset_labels):
    
    # Apply unsupervised kmeans on the entire feature map set
    # in order to initialize our unsupervised PIM
    kmeans = KMeans(n_clusters=assumed_k, random_state=0).fit(feature_map_preds)
    km_centroids = from_numpy_to_torch(np.asarray(kmeans.cluster_centers_), torch_device)
    
    # lambda search using unsupervised PIM (i.e. PIM without CE term)
    lambda_vals_list = np.arange(0.05, 1.+0.05, 0.05)
    all_lab_Accs = []
    mb_size = len(feature_map_preds)
    for curr_lambda_val in lambda_vals_list:
        print("Current lambda value:", np.round(curr_lambda_val,2))
        pim = PIM_partitioner(num_features=len(feature_map_preds[0]), num_classes=assumed_k).to(device_name)
        for name, param in pim.named_parameters():
            if name == "partitioner.weight":
                pim.partitioner.weight.data = km_centroids.type_as(param)
        optimizer = optim.Adam(pim.parameters(), lr=0.001, weight_decay=1e-2)
        for epoch in range(epochs):
            running_loss = 0.0
            for mb_id in range(0, int(len(feature_map_preds) / mb_size)):
                mb_inputs = from_numpy_to_torch(feature_map_preds[mb_id * mb_size:(mb_id + 1) * mb_size], torch_device).float()
                mb_lab_mask = from_numpy_to_torch(mask_lab[mb_id * mb_size:(mb_id + 1) * mb_size], torch_device)
                optimizer.zero_grad()
                mb_logits_outputs = pim(mb_inputs)
                soft_mb_logits_outputs = F.softmax(mb_logits_outputs, dim=1)
                loss = - ((soft_mb_logits_outputs[~mb_lab_mask] + 2.220446049250313e-16) * torch.log(soft_mb_logits_outputs[~mb_lab_mask] + 2.220446049250313e-16)).sum(
                           1).mean() * curr_lambda_val
                loss += (soft_mb_logits_outputs.mean(0) * torch.log(soft_mb_logits_outputs.mean(0) + 1e-12)).sum()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
        ## Eval
        with torch.no_grad():
            outputs = pim(from_numpy_to_torch(feature_map_preds, torch_device).float())
            _, predicted = torch.max(outputs.data, 1)
            lab_Acc = p_metrics.cluster_acc_old_only(np.asarray(lab_subset_labels, dtype=int), np.asarray(from_torch_to_numpy(predicted)[mask_lab], dtype=int))
        all_lab_Accs.append(np.round(100. * lab_Acc, 2))
        ##
    lambda_vals_list = np.asarray(lambda_vals_list)
    all_lab_Accs = np.asarray(all_lab_Accs)
    
    if not os.path.exists(path_auto_lambda):
        os.makedirs(path_auto_lambda)
    with open(path_auto_lambda + '/' + "lambda_vals_list_"+ str(assumed_k) +".npy", 'wb') as f:
        np.save(f, lambda_vals_list)
    with open(path_auto_lambda + '/' + "all_lab_Accs_"+ str(assumed_k) +".npy", 'wb') as f:
        np.save(f, all_lab_Accs)
    
    auto_lambda_value = lambda_vals_list[1:][np.argmax(all_lab_Accs[1:])]
    
    return auto_lambda_value