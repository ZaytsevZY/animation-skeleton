# rigging/lbs.py
import numpy as np

def apply_lbs(V, W, bones, G_current, G_bind_inv):
    """
 V: (N,3)  绑定姿态顶点
    W: (N,B)  顶点-骨骼权重（行和=1）
    bones: list[(j_parent, j_child)]
    G_current: (J,4,4)  当前全局矩阵（FK 后）
    G_bind_inv: (J,4,4)  绑定全局矩阵的逆
  return: (N,3)  变形后顶点
    """
    N = V.shape[0]
    Vh = np.hstack([V, np.ones((N,1), dtype=np.float32)])  # (N,4)
    out = np.zeros((N,4), dtype=np.float32)
    for k,(jp,jc) in enumerate(bones):
        T = G_current[jc] @ G_bind_inv[jc]
        out += (W[:,[k]] * (Vh @ T.T))
    return out[:,:3]