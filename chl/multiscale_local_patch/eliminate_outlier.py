import math

import torch
import time
import operator
import numpy as np
from torch import nn
import scipy.spatial as spatial
from torch.nn.utils.rnn import pad_sequence
from chl.multiscale_local_patch.test_correspond import test_corr


@torch.no_grad()
def multi_scale_local_patches(point_c, p_index):
    unique_indices = torch.unique(p_index).detach().cpu().numpy()
    min_indices = unique_indices.shape[0] - 1
    point_index = point_c.detach().cpu().numpy()[unique_indices]
    point_index_tree = spatial.KDTree(point_index)
    mscale_local_patches = {}
    scales = [min(3, min_indices), min(5, min_indices), min(7, min_indices)]
    max_distance = 0
    for s in scales:
        local_feat = {}
        for i in range(point_index.shape[0]):
            distance, indices = point_index_tree.query(point_index[i], k=s + 1)
            selected_indices = unique_indices[indices]
            local_feat[unique_indices[i]] = np.array([selected_indices, distance])
        mscale_local_patches[s] = local_feat
    return mscale_local_patches, scales, max_distance


@torch.no_grad()
def local_grape_eliminate_outlier(ref_point, src_point, ref_index, src_index, gt_indices):
    ref_mscale_local, ref_local_scales, ref_max_distance = multi_scale_local_patches(ref_point, ref_index)
    src_mscale_local, src_local_scales, src_max_distance = multi_scale_local_patches(src_point, src_index)
    corr_index = torch.stack([ref_index, src_index], dim=0)
    unique_c = torch.unique(corr_index, dim=1).T.detach().cpu().numpy()
    len_unique = unique_c.shape[0]
    len_ls = len(ref_local_scales)
    multi_corr = np.zeros((len_ls + 1, len_unique))
    multi_distance = np.zeros((len_ls + 1, len_unique))
    for r, s, i in zip(ref_local_scales, src_local_scales, range(len_ls)):
        for j in range(len_unique):
            corr_distance = 0
            cont = 0
            ref_corr_point = unique_c[j][0]
            src_corr_point = unique_c[j][1]
            ref_local_point = ref_mscale_local[r][ref_corr_point]
            src_local_point = src_mscale_local[s][src_corr_point]
            for k in range(len_unique):
                if unique_c[k, 0] in ref_local_point[0]:
                    if unique_c[k, 1] in src_local_point[0]:
                        cont = cont + 1
                        ref_indices = np.where(ref_local_point[0] == unique_c[k][0])[0]
                        ref_distance = ref_local_point[1][ref_indices]
                        src_indices = np.where(src_local_point[0] == unique_c[k][1])[0]
                        src_distance = src_local_point[1][src_indices]
                        current_distance = abs(ref_distance - src_distance) / (max(ref_distance, src_distance) + 1e-8)
                        current_score = np.exp(-current_distance)
                        corr_distance += current_score
            multi_corr[i][j] = cont
            multi_distance[i][j] = corr_distance/cont
        sel_count = np.where(multi_corr[i] > (min(r, s) * 2))
        sel_distance = np.where(multi_distance[i] > (np.sum(multi_distance[i]) / len_unique))
        multi_corr[len_ls][sel_count] += 1
        multi_distance[len_ls][sel_distance] += 1
    sel_index = np.where((multi_corr[len_ls] > 0) & (multi_distance[len_ls] > 0))
    local_score = np.squeeze(multi_distance[:len_ls, sel_index])
    scores = np.mean(local_score, axis=0)
    sel = unique_c[sel_index]
    ref = torch.from_numpy(sel.T[0]).cuda()
    src = torch.from_numpy(sel.T[1])

    # test_corr(ref, src, gt_indices)
    return ref, src, scores


@torch.no_grad()
def multi_scale_local_patches_torch(point_c, p_index, co_scale):
    unique_indices = torch.unique(p_index)
    min_indices = unique_indices.shape[0] - 1
    point_index = point_c[unique_indices]
    dists = torch.cdist(point_index, point_index)
    mscale_local_patches = {}
    scales = [min(co_scale[0], min_indices), min(co_scale[1], min_indices), min(co_scale[2], min_indices)]  # , min(co_scale[3], min_indices)
    for s in scales:
        top_k_corr = torch.topk(dists, k=s + 1, dim=1, largest=False)
        top_k_indices = unique_indices[top_k_corr.indices]
        mscale_local_patches[s] = torch.stack([top_k_indices, top_k_corr.values], dim=1)
    return mscale_local_patches, scales, unique_indices


class MultiScaleLocalCon(nn.Module):
    def __init__(self, co_scale):
        super(MultiScaleLocalCon, self).__init__()
        self.co_scale = co_scale
        pass

    def forward(self, ref_point, src_point, ref_index, src_index, gt_indices):
        ref_mscale_local, ref_local_scales, ref_mscale_indices = multi_scale_local_patches_torch(ref_point,
                                                                                                 ref_index,
                                                                                                 self.co_scale)
        src_mscale_local, src_local_scales, src_mscale_indices = multi_scale_local_patches_torch(src_point,
                                                                                                 src_index,
                                                                                                 self.co_scale)
        corr_index = torch.stack([ref_index, src_index], dim=0)
        unique_c = torch.unique(corr_index, dim=1).T
        len_unique = unique_c.shape[0]
        len_ls = len(ref_local_scales)
        multi_corr = torch.zeros(len_ls + 1, len_unique).cuda()
        multi_distance = torch.zeros(len_ls + 1, len_unique).cuda()
        for r, s, i in zip(ref_local_scales, src_local_scales, range(len_ls)):
            ref_corr_point = torch.where(ref_mscale_indices == unique_c[:, 0].unsqueeze(1))[1]
            src_corr_point = torch.where(src_mscale_indices == unique_c[:, 1].unsqueeze(1))[1]
            ref_local_point = ref_mscale_local[r][ref_corr_point]
            src_local_point = src_mscale_local[s][src_corr_point]
            ref_local_eq = ref_local_point[:, 0, :].unsqueeze(2) == unique_c[:, 0].unsqueeze(0)
            src_local_eq = src_local_point[:, 0, :].unsqueeze(2) == unique_c[:, 1].unsqueeze(0)
            corr_tu = torch.any(ref_local_eq, dim=1) & torch.any(src_local_eq, dim=1)
            co_select_u = unique_c[torch.nonzero(corr_tu).T[1]].view(-1, 2)
            co_select_count = torch.sum(corr_tu == 1, dim=1)
            co_select_list = torch.split(co_select_u, co_select_count.tolist())
            ########################################################################################
            # a = torch.split(ref_local_point[:, 0, :], (1,) * len_unique)
            ########################################################################################
            for j in range(len_unique):
                ref_indices = torch.where(ref_local_point[j, 0, :] == co_select_list[j][:, 0].unsqueeze(1))[1]
                src_indices = torch.where(src_local_point[j, 0, :] == co_select_list[j][:, 1].unsqueeze(1))[1]
                ref_distance = ref_local_point[j, 1, :][ref_indices.squeeze()]
                src_distance = src_local_point[j, 1, :][src_indices.squeeze()]
                current_distance = torch.abs(ref_distance - src_distance) / (
                        torch.max(ref_distance, src_distance) + 1e-8)
                corr_distance = torch.sum(torch.exp(-current_distance))
                multi_distance[i][j] = corr_distance / co_select_list[j].shape[0]
            multi_corr[i] = co_select_count
            sel_count = torch.where(multi_corr[i] > (min(r, s) * 2))
            sel_distance = torch.where(multi_distance[i] > (multi_distance[i].sum() / len_unique))
            multi_corr[len_ls][sel_count] += 1
            multi_distance[len_ls][sel_distance] += 1
        sel_index = torch.where((multi_corr[len_ls] > 0) & (multi_distance[len_ls] > 0))[0]
        local_score = multi_distance[:len_ls, sel_index]
        scores = torch.mean(local_score, dim=0)
        sel = unique_c[sel_index]
        ref = sel.T[0]
        src = sel.T[1]
        return ref, src, scores


def comment():
    # local_feat = {}
    # for i in range(point_index.shape[0]):
    #     indices = torch.topk(dists[i].flatten(), k=s + 1, largest=False)
    #     selected_indices = unique_indices[indices.indices]
    #     ind = unique_indices[i].item()
    #     local_feat[ind] = torch.stack([selected_indices, indices.values])
    pass
