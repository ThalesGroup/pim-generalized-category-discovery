import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred, return_ind=False):
    
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_acc_old_only(y_true, y_pred):

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_acc = cluster_acc(y_true, y_pred)

    return old_acc


# GCD metric (arxiv v1)
def cluster_acc_v1(y_true, y_pred, mask):

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc


# Official GCD metric (CVPR22, arxiv v2)
def cluster_acc_v2(y_true, y_pred, mask):
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])
    
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    
    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


# ORCA metrics
def orca_accuracy(output, target):
    
    num_correct = np.sum(output == target)
    res = num_correct / len(target)
    
    return res

def orca_cluster_acc(y_pred, y_true):
    
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    return w[row_ind, col_ind].sum() / y_pred.size

def orca_all_old_new_ACCs(unlab_preds, unlab_gt_labs, seen_mask):
    
    orca_all_acc = orca_cluster_acc(unlab_preds, unlab_gt_labs)
    orca_old_acc = orca_accuracy(unlab_preds[seen_mask], unlab_gt_labs[seen_mask])
    orca_new_acc = orca_cluster_acc(unlab_preds[~seen_mask], unlab_gt_labs[~seen_mask])
    
    return orca_all_acc, orca_old_acc, orca_new_acc
