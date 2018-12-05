import numpy as np


def L1_norm(weight,prune_num):
    sort_by_l1 = np.argsort(np.sum(np.abs(eval_weight), axis=(1, 2, 3)), axis=0)
    layer_index = sort_by_l1[:prune_num]
    return layer_index