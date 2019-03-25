import numpy as np
import itertools


def L1_norm(eval_weight,prune_num):
    sort_by_l1 = np.argsort(np.sum(np.abs(eval_weight), axis=(1, 2, 3)), axis=0)
    layer_index = sort_by_l1[:prune_num]
    return layer_index

def L2_norm(eval_weight,prune_num):
    sort_by_l1 = np.argsort(np.sum(np.abs(eval_weight), axis=(1, 2, 3)), axis=0)
    layer_index = sort_by_l1[:prune_num]
    return layer_index

def feature_grim_matrix(eval_feature):
    B,C,H,W = eval_feature.shape

    eval_feature = eval_feature.reshape(B,C,H*W)
    T_feature = eval_feature.transpose((0,2,1))
    grim_matrix_batch = np.einsum('ijk,ikl->ijl',eval_feature,T_feature)
    grim_matrix_batch = grim_matrix_batch/(H*W)
    grim_matrix_avg = np.average(grim_matrix_batch, axis=0)


    return grim_matrix_avg

