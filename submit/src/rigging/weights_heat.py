import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# 构建 cotangent Laplacian L 和 质量矩阵 M

def cotangent_laplacian(V, F):
    # 省略：计算每条边对角的 cot(alpha)+cot(beta)，装配稀疏 C，再得 L = M^{-1} C
    pass

# 对每根骨骼 k，设定 Dirichlet 边界条件（骨骼附近顶点值=1，其它骨骼=0），解 L w_k = 0

def heat_weights(V, F, Jpos, bones, anchor_radius=0.02):
    # 1) 选"锚点"顶点集合 S_k：距离骨骼段 < r 的顶点
    # 2) 对每根骨骼解一次稀疏线性方程（或一次分解多次回代）
 # 3) 负值截断为 0，最后按顶点归一化使 \sum_k w_k = 1
    pass