# test_joints_basic.py
"""
最小测试程序 - 验证关节球是否可见
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

# 导入模型加载模块
from rigging.skeleton_loader import load_skeleton_from_glb, load_mesh_from_glb
from rigging.mesh_io import Mesh


class SimpleJointViewer(QMainWindow):
    """简单的关节查看器"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("关节显示测试")
        self.setGeometry(100, 100, 1000, 800)
        
        # 创建 3D 视图
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter.interactor)
        
        # 加载和显示
        self.load_and_render()
    
    def load_and_render(self):
        """加载并渲染"""
        print("\n" + "="*60)
        print("🧪 开始测试...")
        
        # 1. 加载骨架
        glb_path = "data/cow/cow.glb"
        print(f"\n📂 加载文件: {glb_path}")
        
        try:
            # 加载网格
            vertices, faces = load_mesh_from_glb(glb_path, scale=1.0)
            mesh = Mesh()
            mesh.set_vertices_faces(vertices, faces)
            print(f"✅ 网格加载成功: {vertices.shape[0]} 顶点")
            
            # 加载骨架
            from rigging.skeleton_loader import load_skeleton_from_glb
            skeleton, bones = load_skeleton_from_glb(glb_path, scale=1.0, verbose=False)
            joint_positions = skeleton.bind_positions()
            
            print(f"✅ 骨架加载成功: {skeleton.n} 个关节")
            
            # 2. 打印关键信息
            print(f"\n📊 数据分析:")
            print(f"   网格中心: {vertices.mean(axis=0)}")
            print(f"   网格范围: X[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
            print(f"              Y[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
            print(f"              Z[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")
            
            print(f"\n   关节中心: {joint_positions.mean(axis=0)}")
            print(f"   关节范围: X[{joint_positions[:,0].min():.3f}, {joint_positions[:,0].max():.3f}]")
            print(f"              Y[{joint_positions[:,1].min():.3f}, {joint_positions[:,1].max():.3f}]")
            print(f"              Z[{joint_positions[:,2].min():.3f}, {joint_positions[:,2].max():.3f}]")
            
            # 3. 计算合适的球体半径
            mesh_size = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
            sphere_radius = mesh_size * 0.02  # 网格尺寸的2%
            print(f"\n🎯 网格总尺寸: {mesh_size:.3f}")
            print(f"   球体半径: {sphere_radius:.3f}")
            
            # 4. 渲染测试（只显示前5个关节，确保可见）
            print(f"\n🎨 开始渲染...")
            
            # 测试1: 绘制网格（半透明）
            print("   1️⃣ 绘制网格...")
            faces_with_count = np.hstack([np.full((len(faces), 1), 3), faces])
            mesh_pv = pv.PolyData(vertices, faces_with_count)
            self.plotter.add_mesh(
                mesh_pv,
                color='lightblue',
                opacity=0.3,  # 非常透明
                show_edges=False
            )
            
            # 测试2: 绘制骨架
            print("   2️⃣ 绘制骨架...")
            for jp, jc in bones:
                p1 = joint_positions[jp]
                p2 = joint_positions[jc]
                line = pv.Line(p1, p2)
                self.plotter.add_mesh(line, color='darkred', line_width=8)
            
            # 测试3: 绘制所有关节（使用点云）
            print("   3️⃣ 绘制关节点云...")
            points = pv.PolyData(joint_positions)
            self.plotter.add_points(
                points,
                color='red',
                point_size=30,
                render_points_as_spheres=True
            )
            
            # 测试4: 绘制前5个关节的大球体（确保可见）
            print("   4️⃣ 绘制前5个关节的球体标记...")
            colors = ['red', 'green', 'blue', 'yellow', 'magenta']
            for i in range(min(5, skeleton.n)):
                sphere = pv.Sphere(
                    radius=sphere_radius * 2,  # 双倍大小
                    center=joint_positions[i]
                )
                self.plotter.add_mesh(
                    sphere,
                    color=colors[i],
                    opacity=0.9
                )
                
                # 添加文本标签
                self.plotter.add_point_labels(
                    [joint_positions[i]],
                    [f"Joint {i}"],
                    font_size=12,
                    point_color='white',
                    point_size=5,
                    text_color=colors[i],
                    bold=True
                )
            
            # 5. 添加坐标轴
            self.plotter.add_axes()
            
            # 6. 设置相机
            self.plotter.reset_camera()
            self.plotter.camera.zoom(1.0)
            
            print("\n✅ 渲染完成!")
            print("="*60)
            print("\n💡 说明:")
            print("   - 前5个关节用彩色大球标记（红绿蓝黄紫）")
            print("   - 所有关节用红色点云显示")
            print("   - 骨架用深红色线条显示")
            print("   - 网格半透明显示")
            print("\n🎮 操作:")
            print("   - 鼠标左键: 旋转")
            print("   - 滚轮: 缩放")
            print("   - 鼠标中键: 平移")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    app = QApplication(sys.argv)
    window = SimpleJointViewer()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()