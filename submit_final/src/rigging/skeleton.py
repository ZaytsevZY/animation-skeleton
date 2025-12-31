# rigging/skeleton.py
import numpy as np
from dataclasses import dataclass

@dataclass
class Joint:
    name: str
    parent: int  # -1 表根
    pos: np.ndarray  # (3,) 绑定位姿下的关节位置

class Skeleton:
    def __init__(self, joints=None):
        self.joints = joints if joints is not None else []
        self.n = len(self.joints)

    def add_joint(self, joint):
        """添加关节到骨架"""
        self.joints.append(joint)
        self.n = len(self.joints)

    def parents(self):
        return np.array([j.parent for j in self.joints], dtype=np.int32)

    def bind_positions(self):
        return np.stack([j.pos for j in self.joints], axis=0)  # (J,3)

    def global_from_local(self, local_T):
        """FK：局部变换按父子层级传播为全局矩阵。
        修复版本：正确计算全局位置。
        """
        J = self.n
        parents = self.parents()
        bind_pos = self.bind_positions()  # 绑定姿态的关节位置

        # 初始化全局变换矩阵
        global_T = np.zeros((J, 4, 4), dtype=np.float32)

        for i in range(J):
            p = parents[i]

            if p < 0:
                # 根节点：从绑定位置开始，应用局部变换
                bind_matrix = np.eye(4, dtype=np.float32)
                bind_matrix[:3, 3] = bind_pos[i]  # 绑定位置
                global_T[i] = bind_matrix @ local_T[i]
            else:
                # 子节点：从父节点变换后的位置开始
                # 1. 先计算相对于父节点的偏移
                relative_pos = bind_pos[i] - bind_pos[p]
                offset_matrix = np.eye(4, dtype=np.float32)
                offset_matrix[:3, 3] = relative_pos

                # 2. 应用父节点的全局变换，然后是偏移，最后是局部变换
                global_T[i] = global_T[p] @ offset_matrix @ local_T[i]

        return global_T

def quadruped_auto_place(bbox_min, bbox_max):
    """四足模板的自动放置（基于包围盒比例），Y轴向上的坐标系"""
    c = (bbox_min + bbox_max) * 0.5
    L = bbox_max - bbox_min
    x0, y0, z0 = c

    # Y轴向上的坐标系：X前后，Y上下，Z左右
    hips = np.array([x0 - 0.25*L[0], y0 - 0.05*L[1], z0])  # 髋部稍微下降
    chest = np.array([x0 + 0.15*L[0], y0 + 0.0*L[1], z0])  # 胸部在中间高度
    neck = np.array([x0 + 0.35*L[0], y0 + 0.1*L[1], z0])  # 颈部稍微上升
    head = np.array([x0 + 0.5*L[0], y0 + 0.2*L[1], z0])  # 头部更高
    root = np.array([x0 - 0.35*L[0], y0 - 0.1*L[1], z0])  # 根部稍微下降

    # Z轴分量用于左右腿的分离
    zoff = 0.25*L[2]

    # 前肩和前腿 - Y坐标向下延伸
    shoulder_L = chest + np.array([0.0, 0.0, +zoff])
    shoulder_R = chest + np.array([0.0, 0.0, -zoff])
    elbow_L = shoulder_L + np.array([0.0, -0.25*L[1], 0.0])  # Y向下
    elbow_R = shoulder_R + np.array([0.0, -0.25*L[1], 0.0])  # Y向下
    wrist_L = elbow_L + np.array([0.0, -0.25*L[1], 0.0])  # Y向下
    wrist_R = elbow_R + np.array([0.0, -0.25*L[1], 0.0])  # Y向下

    # 后臀和后腿 - Y坐标向下延伸
    hip_L = hips + np.array([0.0, 0.0, +zoff])
    hip_R = hips + np.array([0.0, 0.0, -zoff])
    knee_L = hip_L + np.array([0.0, -0.3*L[1], 0.0])  # Y向下
    knee_R = hip_R + np.array([0.0, -0.3*L[1], 0.0])  # Y向下
    ankle_L = knee_L + np.array([0.0, -0.25*L[1], 0.0])  # Y向下
    ankle_R = knee_R + np.array([0.0, -0.25*L[1], 0.0])  # Y向下

    P = [
        ("root", -1, root),
        ("spine1", 0, hips),
        ("spine2", 1, chest),
        ("neck", 2, neck),
        ("head", 3, head),
        ("L_shoulder", 2, shoulder_L),
        ("L_elbow", 5, elbow_L),
        ("L_wrist", 6, wrist_L),
        ("R_shoulder", 2, shoulder_R),
        ("R_elbow", 8, elbow_R),
        ("R_wrist", 9, wrist_R),
        ("L_hip", 1, hip_L),
        ("L_knee", 11, knee_L),
        ("L_ankle", 12, ankle_L),
        ("R_hip", 1, hip_R),
        ("R_knee", 14, knee_R),
        ("R_ankle", 15, ankle_R),
    ]
    joints = [Joint(n, p, np.asarray(x, dtype=np.float32)) for (n, p, x) in P]
    return Skeleton(joints)