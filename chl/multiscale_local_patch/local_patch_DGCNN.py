import torch
import numpy as np
import torch.nn as nn
import scipy.spatial as spatial


# @torch.no_grad()
# def multi_scale_local_patches(feats_c_norm, point_c_norm, f_masks):
#     r"""利用KD树的最近邻算法获取点云中的多尺度特征
#
#
#     Args:
#         feats_c_norm: 点云的特征
#         point_c_norm: 对应的点云的每个点的坐标信息(x, y, z)
#         f_masks: (BoolTensor=None): masks of the super_points in point cloud (False if empty).
#
#     Returns:
#         mscale_local_patches:点云的多尺度特征
#         scales：最近邻的尺度大小
#
#     """
#
#     scale_index = 1
#     mscale_local_patches = {}
#     if f_masks is None:
#         f_masks = torch.ones(size=(feats_c_norm.shape[0],), dtype=torch.bool).cuda()
#     # remove empty patch
#     f_indices = torch.nonzero(f_masks, as_tuple=True)[0]
#     feats_c_norm = feats_c_norm[f_indices]
#     point_c_norm = point_c_norm[f_indices]
#     # 定义了几个尺度大小 scales，它们代表了不同尺度下的局部块大小
#     scales = [2, 4, 6, 8]
#     mscale_local_patches[0] = feats_c_norm
#     dists = torch.cdist(point_c_norm, point_c_norm)
#     for s in scales:
#         local_feat = torch.zeros(size=(feats_c_norm.shape[0], (s * feats_c_norm.shape[1])), dtype=torch.float32).cuda()
#         for i in range(feats_c_norm.shape[0]):
#             indices = torch.topk(dists[i].flatten(), k=s + 1, largest=False).indices
#             neighbor_feats = feats_c_norm[indices[1:], :]
#             local_feat[i] = neighbor_feats.flatten()
#         mscale_local_patches[scale_index] = local_feat
#         scale_index += 1
#     return mscale_local_patches
#
#
# @torch.no_grad()
# def multi_scale_local_patches1(feats_c_norm, point_c_norm, f_masks):
#     r"""利用KD树的最近邻算法获取点云中的多尺度特征
#
#
#     Args:
#         feats_c_norm: 点云的特征
#         point_c_norm: 对应的点云的每个点的坐标信息(x, y, z)
#         f_masks: (BoolTensor=None): masks of the super_points in point cloud (False if empty).
#
#     Returns:
#         mscale_local_patches:点云的多尺度特征
#         scales：最近邻的尺度大小
#
#     """
#
#     scale_index = 1
#     mscale_local_patches = {}
#
#     if f_masks is None:
#         f_masks = torch.ones(size=(feats_c_norm.shape[0],), dtype=torch.bool).cuda()
#     # remove empty patch
#     f_indices = torch.nonzero(f_masks, as_tuple=True)[0]
#     feats_c_norm = feats_c_norm[f_indices]
#     point_c_norm = point_c_norm[f_indices]
#
#     # 定义了几个尺度大小 scales，它们代表了不同尺度下的局部块大小
#     scales = [2, 4, 6, 8]
#     pre_scales = [0, 0, 0, 0]
#     # pre_scales1 = [0] * len(scales)
#
#     # 计算每个点之间的欧式距离
#     # point_distances = torch.cdist(point_c_norm, point_c_norm)
#
#     # 计算每个数据点和查询点的欧氏距离
#     # dists = torch.cdist(point_c_norm, point_c_norm)
#
#     # 将torch的张量转换为numpy数组
#     # point_distances_np = point_distances.detach().cpu().numpy()
#     feats_c_norm_np = feats_c_norm.detach().cpu().numpy()
#     point_np = point_c_norm.detach().cpu().numpy()
#
#     # 构建KD树
#     # feats_c_tree = spatial.KDTree(feats_c_norm_np)
#     point_tree = spatial.KDTree(point_np)
#
#     feats_c_norm_dict = {i: row for i, row in enumerate(feats_c_norm_np)}
#     mscale_local_patches[0] = feats_c_norm_dict
#
#     # 遍历每个尺度
#     for s, pro in zip(scales, pre_scales):
#         local_feat = {}
#         # 遍历每个点
#         for i in range(feats_c_norm_np.shape[0]):
#
#             # 找到最近的k个邻居点
#             # indices 是一个N×(k+1)的Numpy数组，其中每一行是与对应点最近的k个点的索引。
#             # 注意：由于第一个索引是查询点自己，所以k一定是 k=s+1
#             # _, indices = feats_c_tree.query(feats_c_norm_np[i], k=s + 1)  # 基于特征进行距离计算的，不太准确
#
#             _, indices = point_tree.query(point_np[i], k=s + 1)
#
#             # 切片，不重复选择
#             selected_indices = indices[pro: s + 1]
#
#             # 下面的验证是验证一下否是取到是最小的k个距离
#             # min_distances, min_indices = torch.topk(point_distances[i], s+1, largest=False)
#             # min_indices_np = min_indices.detach().cpu().numpy()
#             # ad = np.array([np.in1d(row, indices[ig]) for ig, row in enumerate(min_indices_np)]).T
#             # ad_th = torch.from_numpy(ad)
#
#             # 获取周围最近的 k 个点的局部特征，将它们合并成一个包含 k 个点的局部特征向量 ,保存在neighbor_feats中
#             neighbor_feats = feats_c_norm_np[selected_indices[1:], :]
#             # 由于第一个索引是查询点自己，所以 indices[1:] 表示除去查询点自己之外的最近 15 个点的索引
#
#             # 将ref_feats_c_norm和neighbor_feats连接起来，形成一个维度为(1, N * k)的张量
#             local_feat[i] = np.concatenate([neighbor_feats.reshape(-1)])
#             # 相当于把自己这个点的N个特征和选出的k个最近点的特征放在一个一维的数组中
#
#         mscale_local_patches[scale_index] = local_feat
#         scale_index = scale_index + 1
#
#     return mscale_local_patches


class MultiScaleLocalPatchesTorch(nn.Module):
    def __init__(self, scale_index, scale):
        super(MultiScaleLocalPatchesTorch, self).__init__()
        self.scale_index = scale_index
        self.scale = scale

    def forward(self, feats_c_norm, point_c_norm, f_masks):
        r"""利用最近邻算法获取点云中的多尺度特征

        Args:
            feats_c_norm: 点云的特征
            point_c_norm: 对应的点云的每个点的坐标信息(x, y, z)
            f_masks: (BoolTensor=None): masks of the super_points in point cloud (False if empty).

        Returns:
            mscale_local_patches:点云的多尺度特征
        """
        scale_i = self.scale_index
        mscale_local_patches = {}
        if f_masks is None:
            f_masks = torch.ones(size=(feats_c_norm.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        f_indices = torch.nonzero(f_masks, as_tuple=True)[0]
        feats_c_norm = feats_c_norm[f_indices]
        point_c_norm = point_c_norm[f_indices]
        # 定义了几个尺度大小 scales，它们代表了不同尺度下的局部块大小
        mscale_local_patches[0] = feats_c_norm
        dists = torch.cdist(point_c_norm, point_c_norm)
        for s in self.scale:
            s = min(s, dists.shape[1]-1)
            top_k_indices = torch.topk(dists, k=s + 1, dim=1, largest=False).indices
            neighbor_feats = feats_c_norm[top_k_indices[:, 1:], :]
            local_feat = torch.flatten(neighbor_feats, start_dim=1)
            mscale_local_patches[scale_i] = local_feat
            scale_i += 1
        return mscale_local_patches


# class MultiScaleLocalPatchesNumpy(nn.Module):
#     def __init__(self, scale_index, scale, pre_scales):
#         super(MultiScaleLocalPatchesNumpy, self).__init__()
#         self.scale_index = scale_index
#         self.scale = scale
#         self.pre_scales = pre_scales
#
#     def forward(self, feats_c_norm, point_c_norm, f_masks):
#         r"""利用KD树的最近邻算法获取点云中的多尺度特征
#
#         Args:
#             feats_c_norm: 点云的特征
#             point_c_norm: 对应的点云的每个点的坐标信息(x, y, z)
#             f_masks: (BoolTensor=None): masks of the super_points in point cloud (False if empty).
#
#         Returns:
#             mscale_local_patches:点云的多尺度特征
#         """
#         scale_i = self.scale_index
#         mscale_local_patches = {}
#
#         if f_masks is None:
#             f_masks = torch.ones(size=(feats_c_norm.shape[0],), dtype=torch.bool).cuda()
#         # remove empty patch
#         f_indices = torch.nonzero(f_masks, as_tuple=True)[0]
#         feats_c_norm = feats_c_norm[f_indices]
#         point_c_norm = point_c_norm[f_indices]
#
#         # 计算每个点之间的欧式距离
#         # point_distances = torch.cdist(point_c_norm, point_c_norm)
#
#         # 计算每个数据点和查询点的欧氏距离
#         # dists = torch.cdist(point_c_norm, point_c_norm)
#
#         # 将torch的张量转换为numpy数组
#         # point_distances_np = point_distances.detach().cpu().numpy()
#         feats_c_norm_np = feats_c_norm.detach().cpu().numpy()
#         point_np = point_c_norm.detach().cpu().numpy()
#
#         # 构建KD树
#         # feats_c_tree = spatial.KDTree(feats_c_norm_np)
#         point_tree = spatial.KDTree(point_np)
#
#         feats_c_norm_dict = {i: row for i, row in enumerate(feats_c_norm_np)}
#         mscale_local_patches[0] = feats_c_norm_dict
#
#         # 遍历每个尺度
#         for s, pro in zip(self.scale, self.pre_scales):
#             local_feat = {}
#             # 遍历每个点
#             for i in range(feats_c_norm_np.shape[0]):
#                 # 找到最近的k个邻居点
#                 # indices 是一个N×(k+1)的Numpy数组，其中每一行是与对应点最近的k个点的索引。
#                 # 注意：由于第一个索引是查询点自己，所以k一定是 k=s+1
#                 # _, indices = feats_c_tree.query(feats_c_norm_np[i], k=s + 1)  # 基于特征进行距离计算的，不太准确
#
#                 _, indices = point_tree.query(point_np[i], k=s + 1)
#
#                 # 切片，不重复选择
#                 selected_indices = indices[pro: s + 1]
#
#                 # 下面的验证是验证一下否是取到是最小的k个距离
#                 # min_distances, min_indices = torch.topk(point_distances[i], s+1, largest=False)
#                 # min_indices_np = min_indices.detach().cpu().numpy()
#                 # ad = np.array([np.in1d(row, indices[ig]) for ig, row in enumerate(min_indices_np)]).T
#                 # ad_th = torch.from_numpy(ad)
#
#                 # 获取周围最近的 k 个点的局部特征，将它们合并成一个包含 k 个点的局部特征向量 ,保存在neighbor_feats中
#                 neighbor_feats = feats_c_norm_np[selected_indices[1:], :]
#                 # 由于第一个索引是查询点自己，所以 indices[1:] 表示除去查询点自己之外的最近 15 个点的索引
#
#                 # 将ref_feats_c_norm和neighbor_feats连接起来，形成一个维度为(1, N * k)的张量
#                 local_feat[i] = np.concatenate([neighbor_feats.reshape(-1)])
#                 # 相当于把自己这个点的N个特征和选出的k个最近点的特征放在一个一维的数组中
#
#             mscale_local_patches[scale_i] = local_feat
#             scale_i += 1
#
#         return mscale_local_patches
