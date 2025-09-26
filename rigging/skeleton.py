# rigging/skeleton.py
import numpy as np
from dataclasses import dataclass

@dataclass
class Joint:
    name: str
    parent: int  # -1 表根
    pos: np.ndarray  # (3,) 绑定位姿下的关节位置

class Skeleton:
    def __init__(self, joints):
        self.joints = joints  # list[Joint]
        self.n = len(joints)

    def parents(self):
        return np.array([j.parent for j in self.joints], dtype=np.int32)

    def bind_positions(self):
        return np.stack([j.pos for j in self.joints], axis=0)  # (J,3)

    def global_from_local(self, local_T):
        """FK：局部变换按父子层级传播为全局矩阵。"""
        J = self.n
        parents = self.parents()
        global_T = np.zeros((J,4,4), dtype=np.float32)
        for i in range(J):
            p = parents[i]
            if p < 0:
                global_T[i] = local_T[i]
            else:
                global_T[i] = global_T[p] @ local_T[i]
        return global_T

def quadruped_auto_place(bbox_min, bbox_max):
    """四足模板的自动放置（基于包围盒比例），后续可手动微调。"""
    c = (bbox_min + bbox_max) * 0.5
    L = bbox_max - bbox_min
    x0, y0, z0 = c
    hips  = np.array([x0 - 0.25*L[0], y0 + 0.45*L[1], z0])
    chest = np.array([x0 + 0.15*L[0], y0 + 0.5 *L[1], z0])
    neck  = np.array([x0 + 0.35*L[0], y0 + 0.6 *L[1], z0])
    head  = np.array([x0 + 0.5 *L[0], y0 + 0.7 *L[1], z0])
    root  = np.array([x0 - 0.35*L[0], y0 + 0.4 *L[1], z0])
    zoff = 0.25*L[2]
    shoulder_L = chest + np.array([0.0, 0.0, +zoff])
    shoulder_R = chest + np.array([0.0, 0.0, -zoff])
    elbow_L    = shoulder_L + np.array([0.0, -0.25*L[1], 0.0])
    elbow_R    = shoulder_R + np.array([0.0, -0.25*L[1], 0.0])
    wrist_L    = elbow_L    + np.array([0.0, -0.25*L[1], 0.0])
    wrist_R    = elbow_R    + np.array([0.0, -0.25*L[1], 0.0])
    hip_L  = hips + np.array([0.0, 0.0, +zoff])
    hip_R  = hips + np.array([0.0, 0.0, -zoff])
    knee_L = hip_L + np.array([0.0, -0.3*L[1], 0.0])
    knee_R = hip_R + np.array([0.0, -0.3*L[1], 0.0])
    ankle_L= knee_L + np.array([0.0, -0.25*L[1], 0.0])
    ankle_R= knee_R + np.array([0.0, -0.25*L[1], 0.0])

    P = [
        ("root",   -1, root),
        ("spine1",  0, hips),
        ("spine2",  1, chest),
        ("neck",    2, neck),
        ("head",    3, head),
        ("L_shoulder", 2, shoulder_L),
        ("L_elbow",    5, elbow_L),
        ("L_wrist",    6, wrist_L),
        ("R_shoulder", 2, shoulder_R),
        ("R_elbow",    8, elbow_R),
        ("R_wrist",    9, wrist_R),
        ("L_hip",   1, hip_L),
        ("L_knee", 11, knee_L),
        ("L_ankle",12, ankle_L),
        ("R_hip",   1, hip_R),
        ("R_knee", 14, knee_R),
        ("R_ankle",15, ankle_R),
    ]
    joints = [Joint(n,p, np.asarray(x, dtype=np.float32)) for (n,p,x) in P]
    return Skeleton(joints)
