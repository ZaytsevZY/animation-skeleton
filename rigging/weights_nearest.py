# rigging/weights_nearest.py
import numpy as np

def _point_segment_distance(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def hard_nearest_bone_weights(V, Jpos, bones):
    """每个顶点只受最近一根骨骼影响（权重=1）"""
    nV = V.shape[0]
    nB = len(bones)
    W = np.zeros((nV, nB), dtype=np.float32)
    for i in range(nV):
        dmin, kmin = 1e9, -1
        for k,(jp,jc) in enumerate(bones):
            d = _point_segment_distance(V[i], Jpos[jp], Jpos[jc])
            if d < dmin:
                dmin, kmin = d, k
        W[i, kmin] = 1.0
    return W

def idw_two_bones(V, Jpos, bones):
    """反距离加权，选最近两根骨骼并归一化（更平滑）。"""
    nV = V.shape[0]
    nB = len(bones)
    W = np.zeros((nV, nB), dtype=np.float32)
    for i in range(nV):
        dlist = []
        for k,(jp,jc) in enumerate(bones):
            d = _point_segment_distance(V[i], Jpos[jp], Jpos[jc])
            dlist.append((d,k))
        dlist.sort(key=lambda x: x[0])
        (d0,k0),(d1,k1) = dlist[0], dlist[1]
        w0 = 1.0/(d0+1e-4); w1 = 1.0/(d1+1e-4)
        s = w0+w1
        W[i,k0] = w0/s; W[i,k1] = w1/s
    return W
