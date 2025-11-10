"""
简化版骨架绑定 UI - 使用 PyVista 内置 picking
功能：
- 3D渲染网格、骨架、关节
- 点击选择关节（使用PyVista的picking，无需手动计算投影）
- 拖拽关节移动（按住鼠标左键拖动）
- 实时蒙皮变形
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QEvent
import pyvista as pv
from pyvistaqt import QtInteractor

from rigging.mesh_io import Mesh
from rigging.skeleton_loader import load_skeleton_from_glb, load_mesh_from_glb
from rigging.weights_nearest import idw_two_bones
from rigging.lbs import apply_lbs


class SimpleDragUI(QMainWindow):
    """简化版骨架绑定UI - 使用PyVista内置picking"""
    
    def __init__(self):
        super().__init__()
        
        # 数据存储
        self.mesh = None
        self.skeleton = None
        self.bones = []
        self.weights = None
        self.G_bind_inv = None
        self.joint_transforms = None  # 关节的增量变换（4x4矩阵）
        
        # 选中的关节
        self.selected_joint = None
        self.joint_sphere_actors = {}  # {actor: joint_index} 映射
        
        # 坐标轴箭头（Gizmo）
        self.axis_arrows = {}  # {actor: ('x'|'y'|'z', direction_vector)}
        self.dragging_axis = None  # 当前拖拽的轴
        
        # 拖拽状态
        self.is_dragging = False
        self.last_mouse_pos = None  # 上一帧的鼠标位置
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("简化版骨架绑定 - 拖拽关节")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建3D渲染器
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        layout.addWidget(self.plotter.interactor)
        
        # ✨ 安装事件过滤器
        self.plotter.interactor.installEventFilter(self)
        
        # 创建 picker 用于点选
        import vtk
        self.picker = vtk.vtkPropPicker()
        
        # 状态栏
        self.statusBar().showMessage("💡 点击红色球体选择关节，选中后出现彩色箭头可沿轴拖拽")
    
    def eventFilter(self, obj, event):
        """事件过滤器 - 捕获鼠标事件用于拖拽"""
        if obj == self.plotter.interactor:
            if event.type() == QEvent.MouseButtonPress:
                self.handle_mouse_press(event)
                return False
            elif event.type() == QEvent.MouseMove:
                self.handle_mouse_move(event)
                return self.is_dragging  # 拖拽时拦截事件
            elif event.type() == QEvent.MouseButtonRelease:
                self.handle_mouse_release(event)
                return False
        
        return super().eventFilter(obj, event)
    
    def handle_mouse_press(self, event):
        """处理鼠标按下 - 选择关节或开始拖拽"""
        if event.button() == Qt.LeftButton:
            mouse_x = event.x()
            mouse_y = event.y()
            
            # 获取窗口大小和设备像素比
            window_size = self.plotter.window_size
            device_pixel_ratio = self.plotter.interactor.devicePixelRatio()
            
            print(f"📐 窗口大小: {window_size}, 设备像素比: {device_pixel_ratio}")
            print(f"🖱️ 原始鼠标位置: ({mouse_x}, {mouse_y})")
            
            # 考虑设备像素比（Retina屏幕）
            mouse_x_scaled = mouse_x * device_pixel_ratio
            mouse_y_scaled = mouse_y * device_pixel_ratio
            window_height = window_size[1]
            
            print(f"🖱️ 缩放后位置: ({mouse_x_scaled}, {mouse_y_scaled}), 窗口高度: {window_height}")
            
            # 使用 VTK 的 picker 进行拾取
            # VTK 坐标系从底部开始，需要翻转 Y
            self.picker.Pick(mouse_x_scaled, window_height - mouse_y_scaled, 0, self.plotter.renderer)
            
            # 获取被点击的 actor
            picked_actor = self.picker.GetActor()
            
            print(f"🎯 picked_actor: {type(picked_actor).__name__ if picked_actor else 'None'}")
            
            if picked_actor is not None:
                # 首先检查是否点击了坐标轴箭头
                if picked_actor in self.axis_arrows:
                    axis_name, axis_vector = self.axis_arrows[picked_actor]
                    self.is_dragging = True
                    self.dragging_axis = (axis_name, axis_vector)
                    self.last_mouse_pos = (mouse_x, mouse_y)
                    self.plotter.disable()
                    print(f"🎯 开始拖拽 {axis_name.upper()} 轴")
                    return
                
                # 检查是否点击了关节球体
                found_joint = False
                for sphere_actor, joint_idx in self.joint_sphere_actors.items():
                    if sphere_actor == picked_actor:
                        # 如果已经选中该关节，开始拖拽
                        if self.selected_joint == joint_idx:
                            self.is_dragging = True
                            self.last_mouse_pos = (mouse_x, mouse_y)
                            self.plotter.disable()
                            print(f"🖱️ 开始拖拽关节 [{joint_idx}]")
                        else:
                            # 选中新关节
                            self.selected_joint = joint_idx
                            self.render_scene()
                            joint_name = self.skeleton.joints[joint_idx].name
                            self.statusBar().showMessage(
                                f"✅ 选中关节 [{joint_idx}] {joint_name} - 拖拽箭头沿轴移动，或拖拽球体自由移动"
                            )
                            print(f"✅ 选中关节 [{joint_idx}] {joint_name}")
                        found_joint = True
                        break
                
                if not found_joint:
                    # 点击了其他物体，取消选中
                    print(f"  点击了其他物体（非关节球体）")
                    if self.selected_joint is not None:
                        self.selected_joint = None
                        self.render_scene()
                        self.statusBar().showMessage("💡 点击红色球体选择关节")
            else:
                # 点击空白处，取消选中
                print(f"  点击空白处（没有拾取到任何物体）")
                if self.selected_joint is not None:
                    self.selected_joint = None
                    self.render_scene()
                    self.statusBar().showMessage("💡 点击红色球体选择关节")
    
    def handle_mouse_move(self, event):
        """处理鼠标移动 - 拖拽关节"""
        if self.is_dragging and event.buttons() & Qt.LeftButton and self.selected_joint is not None:
            x, y = event.x(), event.y()
            
            if self.last_mouse_pos is None:
                self.last_mouse_pos = (x, y)
                return
            
            # 计算鼠标移动量
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            
            if abs(dx) < 1 and abs(dy) < 1:
                return
            
            # 获取相机参数
            camera = self.plotter.camera
            camera_pos = np.array(camera.GetPosition())
            
            # 获取当前关节位置
            bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            current_local = np.zeros_like(bind_local)
            for i in range(self.skeleton.n):
                current_local[i] = bind_local[i] @ self.joint_transforms[i]
            G_current = self.skeleton.global_from_local(current_local)
            joint_pos = G_current[self.selected_joint, :3, 3]
            
            # 计算距离缩放因子
            distance = np.linalg.norm(camera_pos - joint_pos)
            scale = distance * 0.001
            
            # 根据是否在拖拽轴来决定移动方向
            if self.dragging_axis is not None:
                # 拖拽坐标轴箭头 - 只沿该轴移动
                axis_name, axis_vector = self.dragging_axis
                
                # 计算相机坐标系
                view_up = np.array(camera.GetViewUp())
                view_dir = camera_pos - joint_pos
                view_dir = view_dir / np.linalg.norm(view_dir)
                
                right = np.cross(view_up, view_dir)
                right = right / np.linalg.norm(right)
                up = np.cross(view_dir, right)
                up = up / np.linalg.norm(up)
                
                # 计算屏幕空间的移动向量
                screen_delta = right * dx * scale + up * dy * scale
                
                # 投影到目标轴上（只保留沿轴的分量）
                delta_along_axis = np.dot(screen_delta, axis_vector) * axis_vector
                
                print(f"  沿 {axis_name.upper()} 轴移动: {delta_along_axis}")
                
                delta = delta_along_axis
            else:
                # 自由拖拽 - 在视角平面上移动
                view_up = np.array(camera.GetViewUp())
                view_dir = camera_pos - joint_pos
                view_dir = view_dir / np.linalg.norm(view_dir)
                
                right = np.cross(view_up, view_dir)
                right = right / np.linalg.norm(right)
                up = np.cross(view_dir, right)
                up = up / np.linalg.norm(up)
                
                delta = right * dx * scale + up * dy * scale
            
            # 更新关节位置
            self.joint_transforms[self.selected_joint][:3, 3] += delta
            self.update_children_cascade(self.selected_joint, delta)
            
            self.last_mouse_pos = (x, y)
            self.render_scene()
    
    def handle_mouse_release(self, event):
        """处理鼠标释放 - 结束拖拽"""
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.dragging_axis = None
            self.last_mouse_pos = None
            self.plotter.enable()
            
            if self.selected_joint is not None:
                joint_name = self.skeleton.joints[self.selected_joint].name
                self.statusBar().showMessage(
                    f"✅ 关节 [{self.selected_joint}] {joint_name} 移动完成"
                )
                print(f"✅ 拖拽完成")
    
    def load_model(self):
        """加载模型"""
        try:
            glb_path = "data/cow/cow.glb"
            
            # 加载网格
            vertices, faces = load_mesh_from_glb(glb_path, scale=1.0)
            self.mesh = Mesh()
            self.mesh.set_vertices_faces(vertices, faces)
            
            # 加载骨架
            self.skeleton, self.bones = load_skeleton_from_glb(glb_path, scale=1.0)
            
            # 计算蒙皮权重
            joint_positions = self.skeleton.bind_positions()
            self.weights = idw_two_bones(self.mesh.v, joint_positions, self.bones)
            
            # 计算绑定姿态逆矩阵
            bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            G_bind = self.skeleton.global_from_local(bind_local)
            self.G_bind_inv = np.linalg.inv(G_bind)
            
            # 初始化关节变换
            self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            
            # 渲染场景
            self.render_scene()
            
            self.statusBar().showMessage(
                f"✅ 加载成功：{self.skeleton.n} 个关节 | 点击关节显示XYZ箭头，拖拽箭头沿轴移动"
            )
            
        except Exception as e:
            print(f"加载失败：{e}")
            import traceback
            traceback.print_exc()
    
    def get_joint_children(self, joint_idx):
        """获取子关节"""
        children = []
        for i, joint in enumerate(self.skeleton.joints):
            if joint.parent == joint_idx:
                children.append(i)
        return children
    
    def update_children_cascade(self, parent_idx, delta):
        """递归更新子关节位置"""
        children = self.get_joint_children(parent_idx)
        for child_idx in children:
            self.joint_transforms[child_idx][:3, 3] += delta
            self.update_children_cascade(child_idx, delta)
    
    def compute_deformed_vertices(self):
        """计算变形后的顶点"""
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
        
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        
        G_current = self.skeleton.global_from_local(current_local)
        
        deformed_vertices = apply_lbs(
            self.mesh.v, self.weights, self.bones, G_current, self.G_bind_inv
        )
        
        return deformed_vertices
    
    def render_scene(self):
        """渲染场景"""
        self.plotter.clear()
        self.joint_sphere_actors = {}
        self.axis_arrows = {}  # 重置箭头映射
        
        # 计算当前关节位置
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        
        G_current = self.skeleton.global_from_local(current_local)
        current_joint_positions = G_current[:, :3, 3]
        
        # 计算变形后的网格
        deformed_vertices = self.compute_deformed_vertices()
        
        # 计算关节球体大小
        mesh_size = np.linalg.norm(deformed_vertices.max(axis=0) - deformed_vertices.min(axis=0))
        sphere_radius = mesh_size * 0.015  # 稍微大一点，更容易点击
        arrow_length = mesh_size * 0.1  # 箭头长度
        arrow_radius = sphere_radius * 0.3  # 箭头粗细
        
        # 1. 渲染网格
        faces_with_count = np.hstack([np.full((len(self.mesh.f), 1), 3), self.mesh.f])
        mesh_pv = pv.PolyData(deformed_vertices, faces_with_count)
        self.plotter.add_mesh(
            mesh_pv,
            color='lightblue',
            opacity=0.5,
            show_edges=True,
            edge_color='navy',
            line_width=0.3,
            smooth_shading=True,
            pickable=False
        )
        
        # 2. 渲染骨骼
        for jp, jc in self.bones:
            p1 = current_joint_positions[jp]
            p2 = current_joint_positions[jc]
            line = pv.Line(p1, p2)
            self.plotter.add_mesh(
                line,
                color='darkred',
                line_width=8,
                opacity=0.8,
                pickable=False
            )
        
        # 3. 渲染关节球体（可点击）
        for i, pos in enumerate(current_joint_positions):
            sphere = pv.Sphere(
                radius=sphere_radius,
                center=pos.tolist(),
                theta_resolution=16,
                phi_resolution=16
            )
            
            # 选中的关节用黄色
            color = 'yellow' if i == self.selected_joint else 'red'
            
            actor = self.plotter.add_mesh(
                sphere,
                color=color,
                opacity=0.9,
                pickable=True,  # 关键：可点击
                lighting=True
            )
            
            # 保存映射
            self.joint_sphere_actors[actor] = i
        
        # 4. 如果有选中的关节，渲染坐标轴箭头（Gizmo）
        if self.selected_joint is not None:
            pos = current_joint_positions[self.selected_joint]
            
            # 定义三个轴：X(红)、Y(绿)、Z(蓝)
            axes = [
                ('x', np.array([1.0, 0.0, 0.0]), 'red'),
                ('y', np.array([0.0, 1.0, 0.0]), 'green'),
                ('z', np.array([0.0, 0.0, 1.0]), 'blue')
            ]
            
            for axis_name, direction, color in axes:
                # 创建箭头
                start_point = pos.tolist()
                end_point = (pos + direction * arrow_length).tolist()
                
                arrow = pv.Arrow(
                    start=start_point,
                    direction=direction.tolist(),
                    tip_length=0.25,
                    tip_radius=0.1,
                    shaft_radius=0.03,
                    scale=float(arrow_length)  # 确保是 Python float
                )
                
                actor = self.plotter.add_mesh(
                    arrow,
                    color=color,
                    opacity=0.8,
                    pickable=True,  # 可点击
                    lighting=True
                )
                
                # 保存箭头映射
                self.axis_arrows[actor] = (axis_name, direction)
            
            # 显示标签
            joint_name = self.skeleton.joints[self.selected_joint].name
            label_pos = pos + np.array([0, sphere_radius * 3, 0])
            
            self.plotter.add_point_labels(
                [label_pos],
                [f"[{self.selected_joint}] {joint_name}"],
                font_size=14,
                bold=True,
                text_color='black',
                point_color='yellow',
                point_size=20,
                shape_opacity=0.8
            )
        
        # 5. 设置相机
        if not hasattr(self, '_camera_set'):
            self.plotter.reset_camera()
            self.plotter.camera.elevation = 15
            self.plotter.camera.azimuth = -60
            self.plotter.camera.zoom(1.2)
            self._camera_set = True
        
        self.plotter.update()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SimpleDragUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()