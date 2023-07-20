

def get_assumed_nbr_of_classes(startegy_for_K, dataset_name):
    
    if startegy_for_K == 'ground_truth_K':
        path_k_strat = 'using_ground_truth_K'
    else:
        path_k_strat = 'using_hat_K/' + startegy_for_K
    
    if startegy_for_K == 'ground_truth_K':
        if dataset_name == 'cifar10':
            k_value = 10
        elif dataset_name == 'cifar100':
            k_value = 100
        elif dataset_name == 'imagenet_100':
            k_value = 100
        elif dataset_name == 'cub':
            k_value = 200
        elif dataset_name == 'scars':
            k_value = 196
        elif dataset_name == 'herbarium':
            k_value = 683
    
    elif startegy_for_K == 'Max_ACC_PIM_Brent':
        if dataset_name == 'cifar10':
            k_value = 10
        elif dataset_name == 'cifar100':
            k_value = 95
        elif dataset_name == 'imagenet_100':
            k_value = 102
        elif dataset_name == 'cub':
            k_value = 227
        elif dataset_name == 'scars':
            k_value = 169
        elif dataset_name == 'herbarium':
            k_value = 563
    
    return k_value, path_k_strat