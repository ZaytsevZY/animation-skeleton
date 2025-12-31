# rigging/mesh_io.py
import trimesh
import numpy as np

class Mesh:
    def __init__(self, path=None):
        """
        初始化Mesh对象

        Parameters:
        -----------
        path : str, optional
            模型文件路径。如果为None,创建空的Mesh对象
        """
        self.v = None  # 顶点数组
        self.f = None  # 面片数组
        self.mesh = None  # trimesh对象

        if path is not None:
            self.load_from_file(path)

    def load_from_file(self, path: str):
        """从文件加载网格"""
        m = trimesh.load(path, force='mesh')
        if not isinstance(m, trimesh.Trimesh):
            m = m.dump().sum()
        m.remove_duplicate_faces()
        m.remove_unreferenced_vertices()
        m.merge_vertices()
        self.v = m.vertices.astype(np.float32)  # (N,3)
        self.f = m.faces.astype(np.int32)       # (M,3)
        self.mesh = m
        return self

    def set_vertices_faces(self, vertices, faces):
        """直接设置顶点和面片"""
        self.v = vertices.astype(np.float32)
        self.f = faces.astype(np.int32)
        # 创建trimesh对象
        self.mesh = trimesh.Trimesh(vertices=self.v, faces=self.f)
        return self