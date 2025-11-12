"""
优化版骨架绑定 UI - 带工具栏
功能：
- 左侧工具栏：重置按钮、蒙皮模式切换
- 性能优化：Actor缓存、延迟更新
- 修复：UI颜色优化
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QComboBox,
                              QGroupBox, QSplitter)
from PyQt5.QtCore import Qt, QEvent, QTimer
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from rigging.mesh_io import Mesh
from rigging.skeleton_loader import load_skeleton_from_glb, load_mesh_from_glb
from rigging.weights_nearest import idw_two_bones
from rigging.lbs import apply_lbs


class OptimizedDragUI(QMainWindow):
    """优化版骨架绑定UI - 带工具栏"""
    
    def __init__(self):
        super().__init__()
        
        # 数据存储
        self.mesh = None
        self.skeleton = None
        self.bones = []
        self.weights = None  # 完整权重（多关节加权）
        self.simple_weights = None  # 简化权重（单关节最近邻）
        self.G_bind_inv = None
        self.joint_transforms = None
        self.initial_joint_transforms = None
        
        # 选中的关节
        self.selected_joint = None
        self.joint_sphere_actors = {}
        
        # 坐标轴箭头
        self.axis_arrows = {}
        self.dragging_axis = None
        
        # 拖拽状态
        self.is_dragging = False
        self.last_mouse_pos = None
        
        # 缓存Actor
        self.mesh_actor = None
        self.bone_actors = []
        self.joint_actors = []
        self.gizmo_actors = []
        self.label_actor = None
        
        # 延迟更新
        self.pending_update = False
        self.update_timer = QTimer()
        self.update_timer.setInterval(16)  # ~60 FPS
        self.update_timer.timeout.connect(self._deferred_update)
        
        # 蒙皮模式：'full' 或 'simple'
        self.skinning_mode = 'full'
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("骨架绑定工具 - 优化版")
        self.setGeometry(100, 100, 1400, 800)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # ===== 左侧工具栏 =====
        toolbar_widget = self.create_toolbar()
        
        # ===== 右侧3D视图 =====
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        
        # 使用 QSplitter 分割工具栏和3D视图
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(toolbar_widget)
        splitter.addWidget(self.plotter.interactor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 1150])
        
        main_layout.addWidget(splitter)
        
        # 安装事件过滤器
        self.plotter.interactor.installEventFilter(self)
        
        # 创建 picker
        self.picker = vtk.vtkPropPicker()
        
        # 状态栏
        self.statusBar().showMessage("💡 点击红色球体选择关节，拖拽箭头沿轴移动")
    
    def create_toolbar(self):
        """✅ 创建左侧工具栏"""
        toolbar = QWidget()
        toolbar.setFixedWidth(250)
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #000000;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #000000;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#resetButton {
                background-color: #ff9800;
            }
            QPushButton#resetButton:hover {
                background-color: #e68900;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
                color: #333;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #666;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #333;
                selection-background-color: #4CAF50;
                selection-color: white;
                border: 1px solid #cccccc;
            }
        """)
        
        layout = QVBoxLayout(toolbar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # ===== 标题 =====
        title = QLabel("骨架绑定工具")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(title)
        
        # ===== 控制组 =====
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout()
        
        # 重置按钮
        self.reset_button = QPushButton("🔄 重置到初始状态")
        self.reset_button.setObjectName("resetButton")
        self.reset_button.clicked.connect(self.reset_to_initial)
        control_layout.addWidget(self.reset_button)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # ===== 蒙皮设置组 =====
        skinning_group = QGroupBox("蒙皮设置")
        skinning_layout = QVBoxLayout()
        
        # 蒙皮模式标签
        mode_label = QLabel("蒙皮模式:")
        mode_label.setStyleSheet("font-weight: normal; color: #555;")
        skinning_layout.addWidget(mode_label)
        
        # 蒙皮模式下拉框
        self.skinning_combo = QComboBox()
        self.skinning_combo.addItem("完整蒙皮（多关节加权）", "full")
        self.skinning_combo.addItem("简化蒙皮（最近关节）", "simple")
        self.skinning_combo.currentIndexChanged.connect(self.on_skinning_mode_changed)
        skinning_layout.addWidget(self.skinning_combo)
        
        # 模式说明
        mode_info = QLabel(
            "• 完整蒙皮：高质量，计算较慢\n"
            "• 简化蒙皮：快速，适合预览"
        )
        mode_info.setStyleSheet(
            "font-size: 11px; color: #555; "
            "background-color: #fff; padding: 8px; "
            "border-radius: 3px; border: 1px solid #ddd;"
        )
        mode_info.setWordWrap(True)
        skinning_layout.addWidget(mode_info)
        
        skinning_group.setLayout(skinning_layout)
        layout.addWidget(skinning_group)
        
        # ===== 占位符 =====
        layout.addStretch()
        
        # 底部信息
        info_label = QLabel("版本 1.0")
        info_label.setStyleSheet("font-size: 10px; color: #999;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        return toolbar
    
    def reset_to_initial(self):
        """重置到初始状态"""
        if self.initial_joint_transforms is None:
            self.statusBar().showMessage("⚠️ 没有可重置的初始状态")
            return
        
        self.joint_transforms = self.initial_joint_transforms.copy()
        self.selected_joint = None
        self.update_deformed_mesh_only()
        
        self.statusBar().showMessage("✅ 已重置到初始状态")
        print("🔄 重置到初始状态")
    
    def on_skinning_mode_changed(self, index):
        """蒙皮模式切换"""
        self.skinning_mode = self.skinning_combo.itemData(index)
        self.update_deformed_mesh_only()
        
        mode_name = self.skinning_combo.currentText()
        self.statusBar().showMessage(f"✅ 切换到：{mode_name}")
        print(f"🎨 蒙皮模式切换为：{self.skinning_mode}")
    
    def eventFilter(self, obj, event):
        """事件过滤器"""
        if obj == self.plotter.interactor:
            if event.type() == QEvent.MouseButtonPress:
                self.handle_mouse_press(event)
                return False
            elif event.type() == QEvent.MouseMove:
                self.handle_mouse_move(event)
                return self.is_dragging
            elif event.type() == QEvent.MouseButtonRelease:
                self.handle_mouse_release(event)
                return False
        
        return super().eventFilter(obj, event)
    
    def handle_mouse_press(self, event):
        """处理鼠标按下"""
        if event.button() == Qt.LeftButton:
            mouse_x = event.x()
            mouse_y = event.y()
            
            window_size = self.plotter.window_size
            device_pixel_ratio = self.plotter.interactor.devicePixelRatio()
            
            mouse_x_scaled = mouse_x * device_pixel_ratio
            mouse_y_scaled = mouse_y * device_pixel_ratio
            window_height = window_size[1]
            
            self.picker.Pick(mouse_x_scaled, window_height - mouse_y_scaled, 0, self.plotter.renderer)
            picked_actor = self.picker.GetActor()
            
            if picked_actor is not None:
                if picked_actor in self.axis_arrows:
                    axis_name, axis_vector = self.axis_arrows[picked_actor]
                    self.is_dragging = True
                    self.dragging_axis = (axis_name, axis_vector)
                    self.last_mouse_pos = (mouse_x, mouse_y)
                    self.plotter.disable()
                    print(f"🎯 开始拖拽 {axis_name.upper()} 轴")
                    return
                
                for sphere_actor, joint_idx in self.joint_sphere_actors.items():
                    if sphere_actor == picked_actor:
                        if self.selected_joint == joint_idx:
                            self.is_dragging = True
                            self.last_mouse_pos = (mouse_x, mouse_y)
                            self.plotter.disable()
                            print(f"🖱️ 开始拖拽关节 [{joint_idx}]")
                        else:
                            self.selected_joint = joint_idx
                            self.update_gizmo_only()
                            joint_name = self.skeleton.joints[joint_idx].name
                            self.statusBar().showMessage(
                                f"✅ 选中关节 [{joint_idx}] {joint_name}"
                            )
                            print(f"✅ 选中关节 [{joint_idx}] {joint_name}")
                        return
                
                if self.selected_joint is not None:
                    self.selected_joint = None
                    self.update_gizmo_only()
                    self.statusBar().showMessage("💡 点击红色球体选择关节")
            else:
                if self.selected_joint is not None:
                    self.selected_joint = None
                    self.update_gizmo_only()
                    self.statusBar().showMessage("💡 点击红色球体选择关节")
    
    def handle_mouse_move(self, event):
        """处理鼠标移动"""
        if self.is_dragging and event.buttons() & Qt.LeftButton and self.selected_joint is not None:
            x, y = event.x(), event.y()
            
            if self.last_mouse_pos is None:
                self.last_mouse_pos = (x, y)
                return
            
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            
            if abs(dx) < 1 and abs(dy) < 1:
                return
            
            camera = self.plotter.camera
            camera_pos = np.array(camera.GetPosition())
            
            bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            current_local = np.zeros_like(bind_local)
            for i in range(self.skeleton.n):
                current_local[i] = bind_local[i] @ self.joint_transforms[i]
            G_current = self.skeleton.global_from_local(current_local)
            joint_pos = G_current[self.selected_joint, :3, 3]
            
            distance = np.linalg.norm(camera_pos - joint_pos)
            scale = distance * 0.001
            
            if self.dragging_axis is not None:
                axis_name, axis_vector = self.dragging_axis
                view_up = np.array(camera.GetViewUp())
                view_dir = camera_pos - joint_pos
                view_dir = view_dir / np.linalg.norm(view_dir)
                
                right = np.cross(view_up, view_dir)
                right = right / np.linalg.norm(right)
                up = np.cross(view_dir, right)
                up = up / np.linalg.norm(up)
                
                screen_delta = right * dx * scale + up * dy * scale
                delta = np.dot(screen_delta, axis_vector) * axis_vector
            else:
                view_up = np.array(camera.GetViewUp())
                view_dir = camera_pos - joint_pos
                view_dir = view_dir / np.linalg.norm(view_dir)
                
                right = np.cross(view_up, view_dir)
                right = right / np.linalg.norm(right)
                up = np.cross(view_dir, right)
                up = up / np.linalg.norm(up)
                
                delta = right * dx * scale + up * dy * scale
            
            self.joint_transforms[self.selected_joint][:3, 3] += delta
            self.update_children_cascade(self.selected_joint, delta)
            
            self.last_mouse_pos = (x, y)
            
            self.pending_update = True
            if not self.update_timer.isActive():
                self.update_timer.start()
    
    def handle_mouse_release(self, event):
        """处理鼠标释放"""
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.dragging_axis = None
            self.last_mouse_pos = None
            self.plotter.enable()
            
            self.update_timer.stop()
            self.update_deformed_mesh_only()
            
            if self.selected_joint is not None:
                joint_name = self.skeleton.joints[self.selected_joint].name
                self.statusBar().showMessage(
                    f"✅ 关节 [{self.selected_joint}] {joint_name} 移动完成"
                )
                print(f"✅ 拖拽完成")
    
    def _deferred_update(self):
        """延迟更新"""
        if self.pending_update:
            self.pending_update = False
            self.update_deformed_mesh_only()
    
    def load_model(self):
        """加载模型"""
        try:
            glb_path = "data/cow/cow.glb"
            
            vertices, faces = load_mesh_from_glb(glb_path, scale=1.0)
            self.mesh = Mesh()
            self.mesh.set_vertices_faces(vertices, faces)
            
            self.skeleton, self.bones = load_skeleton_from_glb(glb_path, scale=1.0)
            
            joint_positions = self.skeleton.bind_positions()
            
            # 计算完整权重
            print("🔄 计算完整权重...")
            self.weights = idw_two_bones(self.mesh.v, joint_positions, self.bones)
            
            # 计算简化权重
            print("🔄 计算简化权重...")
            self.simple_weights = self.compute_simple_weights(self.mesh.v, joint_positions)
            
            bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            G_bind = self.skeleton.global_from_local(bind_local)
            self.G_bind_inv = np.linalg.inv(G_bind)
            
            self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            self.initial_joint_transforms = self.joint_transforms.copy()
            
            self.render_scene_full()
            
            self.statusBar().showMessage(
                f"✅ 加载成功：{self.skeleton.n} 个关节"
            )
            
        except Exception as e:
            print(f"加载失败：{e}")
            import traceback
            traceback.print_exc()
    
    def compute_simple_weights(self, vertices, joint_positions):
        """计算简化权重 - 每个顶点只跟随最近的关节"""
        n_verts = len(vertices)
        n_joints = len(joint_positions)
        
        distances = np.linalg.norm(
            vertices[:, None, :] - joint_positions[None, :, :],
            axis=2
        )
        
        nearest_joint = np.argmin(distances, axis=1)
        
        weights = np.zeros((n_verts, n_joints), dtype=np.float32)
        weights[np.arange(n_verts), nearest_joint] = 1.0
        
        print(f"✅ 简化权重计算完成")
        return weights
    
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
        
        # 根据模式选择权重
        weights = self.simple_weights if self.skinning_mode == 'simple' else self.weights
        
        deformed_vertices = apply_lbs(
            self.mesh.v, weights, self.bones, G_current, self.G_bind_inv
        )
        
        return deformed_vertices
    
    def render_scene_full(self):
        """完整渲染场景"""
        self.plotter.clear()
        self.joint_sphere_actors = {}
        self.axis_arrows = {}
        self.bone_actors = []
        self.joint_actors = []
        
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        
        G_current = self.skeleton.global_from_local(current_local)
        current_joint_positions = G_current[:, :3, 3]
        
        deformed_vertices = self.compute_deformed_vertices()
        
        mesh_size = np.linalg.norm(deformed_vertices.max(axis=0) - deformed_vertices.min(axis=0))
        sphere_radius = mesh_size * 0.015
        
        # 1. 渲染网格
        faces_with_count = np.hstack([np.full((len(self.mesh.f), 1), 3), self.mesh.f])
        mesh_pv = pv.PolyData(deformed_vertices, faces_with_count)
        self.mesh_actor = self.plotter.add_mesh(
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
            actor = self.plotter.add_mesh(
                line,
                color='darkred',
                line_width=8,
                opacity=0.8,
                pickable=False
            )
            self.bone_actors.append((actor, jp, jc))
        
        # 3. 渲染关节球体
        for i, pos in enumerate(current_joint_positions):
            sphere = pv.Sphere(
                radius=sphere_radius,
                center=pos.tolist(),
                theta_resolution=16,
                phi_resolution=16
            )
            
            color = 'yellow' if i == self.selected_joint else 'red'
            
            actor = self.plotter.add_mesh(
                sphere,
                color=color,
                opacity=0.9,
                pickable=True,
                lighting=True
            )
            
            self.joint_sphere_actors[actor] = i
            self.joint_actors.append((actor, i, sphere_radius))
        
        # 4. Gizmo
        self.update_gizmo_only()
        
        # 5. 设置相机
        if not hasattr(self, '_camera_set'):
            self.plotter.reset_camera()
            self.plotter.camera.elevation = 15
            self.plotter.camera.azimuth = -60
            self.plotter.camera.zoom(1.2)
            self._camera_set = True
        
        self.plotter.update()
    
    def update_deformed_mesh_only(self):
        """只更新变形的mesh"""
        if self.mesh_actor is None:
            return
        
        deformed_vertices = self.compute_deformed_vertices()
        
        vtk_points = self.mesh_actor.GetMapper().GetInput().GetPoints()
        vtk_array = numpy_to_vtk(deformed_vertices, deep=True)
        vtk_points.SetData(vtk_array)
        vtk_points.Modified()
        
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        G_current = self.skeleton.global_from_local(current_local)
        current_joint_positions = G_current[:, :3, 3]
        
        for actor, jp, jc in self.bone_actors:
            p1 = current_joint_positions[jp]
            p2 = current_joint_positions[jc]
            line = pv.Line(p1, p2)
            actor.GetMapper().SetInputData(line)
        
        for actor, joint_idx, radius in self.joint_actors:
            pos = current_joint_positions[joint_idx]
            sphere = pv.Sphere(radius=radius, center=pos.tolist(), theta_resolution=16, phi_resolution=16)
            actor.GetMapper().SetInputData(sphere)
        
        self.update_gizmo_only()
        self.plotter.update()
    
    def update_gizmo_only(self):
        """只更新Gizmo"""
        for actor in self.gizmo_actors:
            self.plotter.remove_actor(actor)
        self.gizmo_actors = []
        self.axis_arrows = {}
        
        if self.label_actor is not None:
            self.plotter.remove_actor(self.label_actor)
            self.label_actor = None
        
        if self.selected_joint is None:
            self.plotter.update()
            return
        
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        G_current = self.skeleton.global_from_local(current_local)
        current_joint_positions = G_current[:, :3, 3]
        
        pos = current_joint_positions[self.selected_joint]
        
        mesh_size = np.linalg.norm(self.mesh.v.max(axis=0) - self.mesh.v.min(axis=0))
        arrow_length = mesh_size * 0.1
        
        axes = [
            ('x', np.array([1.0, 0.0, 0.0]), 'red'),
            ('y', np.array([0.0, 1.0, 0.0]), 'green'),
            ('z', np.array([0.0, 0.0, 1.0]), 'blue')
        ]
        
        for axis_name, direction, color in axes:
            arrow = pv.Arrow(
                start=pos.tolist(),
                direction=direction.tolist(),
                tip_length=0.25,
                tip_radius=0.1,
                shaft_radius=0.03,
                scale=float(arrow_length)
            )
            
            actor = self.plotter.add_mesh(
                arrow,
                color=color,
                opacity=0.8,
                pickable=True,
                lighting=True
            )
            
            self.axis_arrows[actor] = (axis_name, direction)
            self.gizmo_actors.append(actor)
        
        joint_name = self.skeleton.joints[self.selected_joint].name
        sphere_radius = mesh_size * 0.015
        label_pos = pos + np.array([0, sphere_radius * 3, 0])
        
        self.label_actor = self.plotter.add_point_labels(
            [label_pos],
            [f"[{self.selected_joint}] {joint_name}"],
            font_size=14,
            bold=True,
            text_color='black',
            point_color='yellow',
            point_size=20,
            shape_opacity=0.8
        )
        
        self.plotter.update()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = OptimizedDragUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()