# feature_maps_manip.py module provides the function to get the saved feature maps

import os
import numpy as np

def get_fm_preds_and_gt_labels(set_path):
    gt_labels = []
    feature_map_preds = []
    print("Loading feature map dataset...")
    with open(os.path.join(set_path, "all_feats.npy"), 'rb') as f_all_feats:
        all_feats = np.load(f_all_feats)
    with open(os.path.join(set_path, "mask_lab.npy"), 'rb') as f_mask_lab:
         # True for a labeled point, False for an unlabeled point
        mask_lab = np.load(f_mask_lab)
    with open(os.path.join(set_path, "mask_cls.npy"), 'rb') as f_mask_cls:
        # True if the point is from an old class,
        # False if the point is from a novel class
        mask_cls = np.load(f_mask_cls) 
    with open(os.path.join(set_path, "targets.npy"), 'rb') as f_targets:
        targets = np.load(f_targets)
    l_feats = all_feats[mask_lab]                        # Get labeled set Z_L
    u_feats = all_feats[~mask_lab]                       # Get unlabeled set Z_U
    l_targets = np.asarray(targets[mask_lab],dtype=int)  # Get labeled targets
    u_targets = np.asarray(targets[~mask_lab],dtype=int) # Get unlabeled targets
    
    # Get portion of mask_cls which corresponds to the unlabeled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)
    
    nbr_of_classes = int(np.max(targets) + 1)
    
    print("Loaded.")
    print(" ")
    return np.asarray(all_feats), np.asarray(targets,dtype=int), nbr_of_classes, mask_cls.astype(bool), mask_lab.astype(bool)