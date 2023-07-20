import os
import yaml
import argparse
from tqdm import tqdm

import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from feature_maps_manip import feature_maps_manip as fm
from metrics import partitioning_metrics as p_metrics
from sklearn.cluster import KMeans 
from models.clustering_methods import torch_semi_supervised_kmeans as ssKM
from configs import config_K
from params_estim.ssKM_protos_initialization import ssKM_protos_init
from params_estim.automatic_lambda_search import lambda_search
from models.PIM import PIM_partitioner


def from_numpy_to_torch(np_array, torch_device):
    return torch.from_numpy(np_array).to(torch_device)


def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()


def partitioning_eval(unlab_gt_labs, unlab_preds, seen_mask, dset_name, path_k_strat):

    # GCD metrics (used in GCD arxiv v1)
    accs_v1 = p_metrics.cluster_acc_v1(unlab_gt_labs, unlab_preds, seen_mask)
    
    # GCD metrics (used in GCD CVPR-22 and GCD arxiv v2)
    accs_v2 = p_metrics.cluster_acc_v2(unlab_gt_labs, unlab_preds, seen_mask)
    
    # ORCA metrics (used in ORCA ICLR-22)
    orca_accs = p_metrics.orca_all_old_new_ACCs(unlab_preds, unlab_gt_labs, seen_mask)
    
    
    print("Classes:                      (All) & (Old) & (New)")
    
    print("PIM ACC (v1):                 ", np.round(100. * accs_v1[0], 1),
                                      "& ", np.round(100. * accs_v1[1], 1), 
                                      "& ", np.round(100. * accs_v1[2], 1))
    
    print("PIM ACC (ORCA metric):        ", np.round(100. * orca_accs[0], 1),
                                      "& ", np.round(100. * orca_accs[1], 1), 
                                      "& ", np.round(100. * orca_accs[2], 1))
    
    print("PIM ACC (Official GCD metric):", np.round(100. * accs_v2[0], 1),
                                      "& ", np.round(100. * accs_v2[1], 1), 
                                      "& ", np.round(100. * accs_v2[2], 1))
    
    path_accs_v2 = 'params_estim/' + path_k_strat + '/' + dset_name + '/scores'
    if not os.path.exists(path_accs_v2):
        os.makedirs(path_accs_v2)
    
    accs_v2_file_name = path_accs_v2 + '/ACCs_v2.npy'
    with open(accs_v2_file_name, 'wb') as f:
        np.save(f, accs_v2)
    return 1


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for GCD challenge")
    
    parser.add_argument('--dataset', type=str, default='cub',
                        choices=['cifar10', 'cifar100', 'imagenet_100', 
                                 'cub', 'scars', 'herbarium'])
    
    parser.add_argument('--device_name', type=str, default='cuda:0',
                        choices=['cpu', 'cuda:0', 'cuda:1'])
    
    parser.add_argument('--centroids_init_nbr', type=int, default=100,
                        help="Number of separate centroids initialization.")
    
    parser.add_argument('--cfg', type=str, default="./configs/config_fm_paths.yml",
                        help="feature maps roots config file")

    parser.add_argument('--epochs', type=int, default=1000,
                        help="Total number of training epoch iterations for PIM partitioner.")
    
    parser.add_argument('--perform_init_protos', type=bool, default=False,
                        help="Set True if you want to (re)perform prototypes initiliaztion using ssKM.")
    
    parser.add_argument('--perform_lambda_search', type=bool, default=False,
                        help="Set True if you want to (re)perform automatic lambda search.")
    
    parser.add_argument('--seed', type=int, default=2022,
                        help="Seed for RandomState.")
    
    parser.add_argument('--k_strat', type=str, default='ground_truth_K',
                        choices=['ground_truth_K',
                                 'Max_ACC_PIM_Brent'],
                        help="Select a strategy to set the assumed number of classes " +
                             "(ground truth or precomputed using Max-ACC).")
    
    return parser.parse_args()


def main():
    
    args = get_arguments()
    with open(args.cfg, "r") as stream:
        try:
            cfg_paths = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    torch_device = torch.device(args.device_name)
    np.random.seed(args.seed)
    
    ### Get feature map set, ground truth labels, labeled set mask, old-new mask
    print("Dataset:", args.dataset)
    gcd_feature_map_datasets_path = cfg_paths["gcd_feature_map_datasets_path"]
    feature_maps_folder_name = cfg_paths[args.dataset]
    set_path = os.path.join(gcd_feature_map_datasets_path, feature_maps_folder_name, 'train')
    feature_map_preds, gt_labels, nbr_of_classes, mask_old_new, mask_lab = fm.get_fm_preds_and_gt_labels(set_path)
    ###
    
    ### Get assumed K
    assumed_k, path_k_strat = config_K.get_assumed_nbr_of_classes(args.k_strat, args.dataset)
    print('K strategy:', args.k_strat, '\nAssumed K:', assumed_k, '\n')
    path_params = 'params_estim/' + path_k_strat + '/' + args.dataset 
    if not os.path.exists(path_params):
        os.makedirs(path_params)
    ###
    
    ### Get ssKM centroids to initialize PIM prototypes
    protos_file_name = path_params + '/' + 'protos.npy'
    if ((not os.path.exists(protos_file_name)) or (args.perform_init_protos == True)) and (args.k_strat != 'ground_truth_K'):
        # Estimate ssKM centroids
        print("PIM init prototypes estimation using ssKM...")
        prototypes = ssKM_protos_init(feature_map_preds, 
                                      assumed_k, 
                                      gt_labels[mask_lab], 
                                      mask_lab, 
                                      mask_old_new, 
                                      args.centroids_init_nbr, 
                                      args.device_name,
                                      protos_file_name)
    else:
        print("PIM init prototypes already generated \n")
        with open(protos_file_name, 'rb') as f:
            prototypes = np.load(f)
    prototypes = from_numpy_to_torch(np.asarray(prototypes), torch_device)
    ###
    
    ### Get lambda (automatically estimated)
    path_auto_lambda = path_params + '/auto_lambda_search'
    lambda_search_lab_acc_file_name = path_auto_lambda  + '/' + "all_lab_Accs_"+ str(assumed_k) +".npy"
    if (not os.path.exists(lambda_search_lab_acc_file_name)) or (args.perform_lambda_search == True):
        # Auto lambda search
        print("Start automatic lambda search...")
        auto_lambda_val = lambda_search(path_auto_lambda, 
                                        feature_map_preds, 
                                        args.epochs, 
                                        assumed_k, 
                                        mask_lab, 
                                        args.device_name, 
                                        args.dataset, 
                                        torch_device, 
                                        gt_labels[mask_lab])
    else:
        print("lambda already estimated")
        with open(path_auto_lambda + '/' + 'lambda_vals_list_'+ str(assumed_k) +'.npy', 'rb') as f:
            lambda_vals_list = np.load(f)
        with open(path_auto_lambda + '/' + 'all_lab_Accs_'+ str(assumed_k) +'.npy', 'rb') as f:
            all_lab_Accs = np.load(f)
        auto_lambda_val = lambda_vals_list[1:][np.argmax(all_lab_Accs[1:])]
    print("Obtained lambda:", auto_lambda_val, "\n")
    ###
    
    ### PIM partitioner
    # Init model and prototypes
    pim = PIM_partitioner(num_features=len(feature_map_preds[0]), num_classes=assumed_k).to(args.device_name)
    for name, param in pim.named_parameters():
        if name=="partitioner.weight":
            pim.partitioner.weight.data=prototypes.type_as(param)
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pim.parameters(), lr=0.001, weight_decay=1e-2)
    
    # PIM training
    print('PIM training...')
    mb_size = len(feature_map_preds)
    for epoch in range(args.epochs):  # loop over the feature map set multiple times
        running_loss = 0.0
        
        for mb_id in range(0, int(len(feature_map_preds)/mb_size)):
            
            # get the batch
            mb_inputs = from_numpy_to_torch(feature_map_preds[mb_id*mb_size:(mb_id+1)*mb_size], torch_device).float()
            mb_gt_labels = from_numpy_to_torch(gt_labels[mb_id*mb_size:(mb_id+1)*mb_size], torch_device)
            mb_lab_mask = from_numpy_to_torch(mask_lab[mb_id*mb_size:(mb_id+1)*mb_size], torch_device)
            mb_lab_points = mb_gt_labels[mb_lab_mask] # We only use labels information for Z_L subset
            optimizer.zero_grad()
            
            mb_logits_outputs = pim(mb_inputs)
            soft_mb_logits_outputs = F.softmax(mb_logits_outputs, dim=1)
            
            loss = - ((soft_mb_logits_outputs[~mb_lab_mask]+2.220446049250313e-16) * torch.log(soft_mb_logits_outputs[~mb_lab_mask]+2.220446049250313e-16)).sum(1).mean() * auto_lambda_val
            loss += (soft_mb_logits_outputs.mean(0) * torch.log(soft_mb_logits_outputs.mean(0) + 1e-12)).sum()
            loss += criterion(mb_logits_outputs[mb_lab_mask], mb_lab_points.to(torch.int64))
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if epoch % 100 == 0:
            print("  epoch:", epoch, ", loss:", running_loss)
            
    print('PIM training is completed! \n')
    
    # PIM partitioning evaluation
    with torch.no_grad():
        outputs = pim(from_numpy_to_torch(feature_map_preds, torch_device).float())
        _, predicted = torch.max(outputs.data, 1)
        unlab_preds = np.asarray(from_torch_to_numpy(predicted)[~mask_lab], dtype=int)
    unlab_gt_labs = np.asarray(gt_labels[~mask_lab], dtype=int)
    seen_mask = mask_old_new[~mask_lab].astype(bool)
    partitioning_eval(unlab_gt_labs, unlab_preds, seen_mask, args.dataset, path_k_strat)
    ###


if __name__ == '__main__':
    main()
