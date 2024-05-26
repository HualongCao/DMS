import torch
import torch.nn as nn


from geotransformer.modules.ops import pairwise_distance
from chl.multiscale_local_patch.test_correspond import test_corr, find_con_low


class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, c_matrix, gt_node_corr_indices, ref_masks=None, src_masks=None):
        r"""Extract super_point correspondences.

        Args:
            ref_feats (Tensor): features of the super_points in reference point cloud.
            src_feats (Tensor): features of the super_points in source point cloud.
            c_matrix (Tensor): 相关性矩阵
            gt_node_corr_indices: 真实对
            ref_masks (BoolTensor=None): masks of the super_points in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the super_points in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]

        # select top-k proposals
        corr_scores = []
        corr_indices = []

        # for i in range(1):
        for i in range(len(c_matrix)):
            if self.dual_normalization:
                ref_matching_scores = c_matrix[i] / c_matrix[i].sum(dim=1, keepdim=True)
                src_matching_scores = c_matrix[i] / c_matrix[i].sum(dim=0, keepdim=True)
                c_matrix[i] = ref_matching_scores * src_matching_scores
            num_correspondences = min(self.num_correspondences[i], c_matrix[i].numel())
            c_score, c_indices = c_matrix[i].view(-1).topk(k=num_correspondences, largest=True)
            corr_scores.extend(list(c_score))
            corr_indices.extend(list(c_indices))

        corr_indices = torch.tensor(corr_indices)
        corr_scores = torch.tensor(corr_scores)

        ref_sel_indices = corr_indices // c_matrix[0].shape[1]
        src_sel_indices = corr_indices % c_matrix[0].shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        # test_corr(ref_corr_indices, src_corr_indices, gt_node_corr_indices)
        # find_con_low(ref_corr_indices, src_corr_indices, corr_scores, gt_node_corr_indices)
        # find_con_low(ref_corr_indices, src_corr_indices, corr_scores, gt_node_corr_indices)

        return ref_corr_indices, src_corr_indices, corr_scores
