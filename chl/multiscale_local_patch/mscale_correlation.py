import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCorrelation(nn.Module):
    def __init__(self, scale):
        super(MultiScaleCorrelation, self).__init__()
        self.scale = scale
    
    def forward(self, ref_multi_scale_local_patches, src_multi_scale_local_patches,):
        r"""计算每个尺度下的相关性矩阵
        Args:

            ref_multi_scale_local_patches:torch类型，目标点云中的每个尺度下的关于局部特征

            src_multi_scale_local_patches:torch类型，源点云中的每个尺度下的关于局部特征

        Returns:
            corr_matrix: 每个尺度下的相关性矩阵

        """
        corr_matrix = {}
        # 循环每个尺度
        for i in range(len(self.scale) + 1):
            # # # 分离对应的尺度
            # ref_mscale_patch = ref_multi_scale_local_patches[i]  # ref_mscale_patch是字典类型, 表示该尺度下每个点的局部特征
            # src_mscale_patch = src_multi_scale_local_patches[i]  # src_mscale_patch是字典类型，表示该尺度下每个点的局部特征
            #
            # # 转换为PyTorch的张量，使用PyTorch内置的函数计算余弦相似度，这样可以利用GPU加速计算，提高效率。
            # ref = torch.tensor(list(ref_mscale_patch.values())).cuda()  # 字典变成tensor型
            # src = torch.tensor(list(src_mscale_patch.values()))  # 字典变成tensor型
            #
            # # 将维度提升一维，这样就可以将二维扩到三维，在三维的张量中，就可以进行广播计算
            # ref = ref.unsqueeze(1)  # 将其增加一个维度，变为(M, 1, D)的三维张量
            # src = src.unsqueeze(0)  # 将其增加一个维度，变为(1, N, D)的三维张量，这样两个三维张量可以进行广播计算

            ref = ref_multi_scale_local_patches[i].unsqueeze(1)  # 将其增加一个维度，变为(M, 1, D)的三维张量
            src = src_multi_scale_local_patches[i].unsqueeze(0)  # 将其增加一个维度，变为(1, N, D)的三维张量，这样两个三维张量可以进行广播计算

            # 求相关性矩阵
            # 最后得到的相关性矩阵大小是(M, N)，表示ref的每一行与src的每一行之间的余弦相似度。
            # similarity_matrix = F.cosine_similarity(ref.cpu(), src.cpu(), dim=-1).cuda()
            similarity_matrix = F.cosine_similarity(ref, src, dim=-1)
            # dim=-1参数表示在最后一个维度上进行计算，即在D维上进行余弦相似度的计算

            # 将相关性矩阵保存到到字典中
            corr_matrix[i] = similarity_matrix

        # for i in range(5):
        #     corr_matrix_np = corr_matrix[i].detach().cpu().numpy()

        return corr_matrix


# @torch.no_grad()
# def multi_scale_correlation(mscale,
#                             ref_multi_scale_local_patches,
#                             src_multi_scale_local_patches,
#                             ):
#     r"""计算每个尺度下的相关性矩阵
#     Args:
#         mscale: 一共多少个尺度
#         ref_multi_scale_local_patches:torch类型，目标点云中的每个尺度下的关于局部特征
#
#         src_multi_scale_local_patches:torch类型，源点云中的每个尺度下的关于局部特征
#
#     Returns:
#         corr_matrix: 每个尺度下的相关性矩阵
#
#     """
#
#     corr_matrix = {}
#
#     # 循环每个尺度
#     for i in range(len(mscale)+1):
#
#         # # 分离对应的尺度
#         ref_mscale_patch = ref_multi_scale_local_patches[i]  # ref_mscale_patch是字典类型, 表示该尺度下每个点的局部特征
#         src_mscale_patch = src_multi_scale_local_patches[i]  # src_mscale_patch是字典类型，表示该尺度下每个点的局部特征
#
#         # 转换为PyTorch的张量，使用PyTorch内置的函数计算余弦相似度，这样可以利用GPU加速计算，提高效率。
#         ref = torch.tensor(list(ref_mscale_patch.values()))  # 字典变成tensor型
#         src = torch.tensor(list(src_mscale_patch.values()))  # 字典变成tensor型
#
#         # 将维度提升一维，这样就可以将二维扩到三维，在三维的张量中，就可以进行广播计算
#         ref = ref.unsqueeze(1)  # 将其增加一个维度，变为(M, 1, D)的三维张量
#         src = src.unsqueeze(0)  # 将其增加一个维度，变为(1, N, D)的三维张量，这样两个三维张量可以进行广播计算
#
#         # ref = ref_multi_scale_local_patches[i].unsqueeze(1)  # 将其增加一个维度，变为(M, 1, D)的三维张量
#         # src = src_multi_scale_local_patches[i].unsqueeze(0)  # 将其增加一个维度，变为(1, N, D)的三维张量，这样两个三维张量可以进行广播计算
#
#         # 求相关性矩阵
#         # 最后得到的相关性矩阵大小是(M, N)，表示ref的每一行与src的每一行之间的余弦相似度。
#         similarity_matrix = F.cosine_similarity(ref, src, dim=-1)
#         # dim=-1参数表示在最后一个维度上进行计算，即在D维上进行余弦相似度的计算
#
#         # 将相关性矩阵保存到到字典中
#         corr_matrix[i] = similarity_matrix
#
#     # for i in range(5):
#     #     corr_matrix_np = corr_matrix[i].detach().cpu().numpy()
#
#     return corr_matrix

