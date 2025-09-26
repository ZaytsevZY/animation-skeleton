# rigging/mesh_io.py
import trimesh
import numpy as np

class Mesh:
    def __init__(self, path: str):
        m = trimesh.load(path, force='mesh')
        if not isinstance(m, trimesh.Trimesh):
            m = m.dump().sum()
        m.remove_duplicate_faces()
        m.remove_unreferenced_vertices()
        m.merge_vertices()
        self.v = m.vertices.astype(np.float32)  # (N,3)
        self.f = m.faces.astype(np.int32)       # (M,3)
        self.mesh = m
