import numpy as np
import torch


def find_identical_rows(matrix1, matrix2):
    identical_rows = []
    for row1 in matrix1:
        for row2 in matrix2:
            if np.array_equal(row1, row2):
                identical_rows.append(row1)
    return identical_rows


# 测试真实对的
def test_corr(ref_indices, src_indices, gt_indices):
    ref = ref_indices.detach().cpu().numpy()
    src = src_indices.detach().cpu().numpy()
    gt = gt_indices.detach().cpu().numpy()
    co = np.array([ref, src]).T
    ture_co = find_identical_rows(co, gt)
    a = list(set(map(tuple, ture_co)))
    print("{:.2f}%".format(len(a)/co.shape[0] * 100))

    pass


# 找出相同的列
def find_con_low(ref, src, scores, gt_indices):
    ref = ref.detach().cpu().numpy()
    src = src.detach().cpu().numpy()
    gt = gt_indices.detach().cpu().numpy()
    co_mat = np.array([ref, src])
    unique_columns, counts = np.unique(co_mat, axis=1, return_counts=True)
    duplicated_columns = unique_columns[:, counts > 2]
    ture_co = find_identical_rows(duplicated_columns.T, gt)
    a = list(set(map(tuple, ture_co)))

    # co_mat = torch.LongTensor([ref, src])
    # unique_columns, counts = torch.unique(co_mat, dim=1, return_counts=True)
    # duplicated_columns = unique_columns[:, counts > 1]
    print(duplicated_columns)
    pass
