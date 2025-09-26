# tools/preview_obj.py
import sys
from rigging.mesh_io import Mesh


if __name__ == '__main__':
    path = sys.argv[1]
    M = Mesh(path)
    print(M.v.shape, M.f.shape)
    M.mesh.show() # trimesh 自带快速预览