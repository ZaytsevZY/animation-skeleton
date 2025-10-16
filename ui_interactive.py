# ui_interactive_final.py
"""
交互式骨架绑定 UI - 修复点选版
功能：
- 点击选择关节（增大关节列表）✅ 修复点击
- 旋转控制
- 平移控制（增量式，滑块自动回中，增大范围）
- Gizmo 可视化（缩小版，无中心球）
- 实时变形预览
- 轻量级模式（减少卡顿）
- 多种绑定方法（热扩散、测地线、IDW）
- 修复LBS蒙皮计算 ✅
- 点击球体选择关节 ✅
- 导出功能
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QListWidget, QSplitter,
    QFileDialog, QMessageBox, QCheckBox, QProgressBar, QSlider,
    QComboBox, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
import pyvista as pv
from pyvistaqt import QtInteractor

from rigging.mesh_io import Mesh
from rigging.skeleton_loader import load_skeleton_from_glb, load_mesh_from_glb
from rigging.weights_nearest import idw_two_bones
from rigging.lbs import apply_lbs


class ImprovedWeights:
    """改进的权重计算方法"""
    
    @staticmethod
    def geodesic_skinning(vertices, joint_positions, bones, mesh_faces=None):
        """
        基于测地线距离的蒙皮权重（简化版）
        使用欧氏距离 + 平滑衰减
        """
        n_verts = len(vertices)
        n_bones = len(bones)
        weights = np.zeros((n_verts, n_bones), dtype=np.float32)
        
        for i, (jp, jc) in enumerate(bones):
            p1 = joint_positions[jp]
            p2 = joint_positions[jc]
            
            # 计算到骨骼线段的距离
            v = p2 - p1
            bone_length = np.linalg.norm(v)
            
            if bone_length < 1e-6:
                # 退化情况：骨骼长度为0
                dist = np.linalg.norm(vertices - p1, axis=1)
            else:
                v_normalized = v / bone_length
                
                # 每个顶点到骨骼的最近点
                t = np.clip(
                    np.dot(vertices - p1, v_normalized), 
                    0, 
                    bone_length
                )
                closest_points = p1 + t[:, None] * v_normalized
                dist = np.linalg.norm(vertices - closest_points, axis=1)
            
            # 使用平滑衰减函数（比 IDW 更强的影响）
            # w = exp(-dist^2 / (2 * sigma^2))
            sigma = bone_length * 0.8  # 影响范围
            if sigma < 0.1:
                sigma = 0.5
            
            weights[:, i] = np.exp(-dist**2 / (2 * sigma**2))
        
        # 归一化
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-8)
        weights = weights / weight_sum
        
        return weights
    
    @staticmethod
    def heat_diffusion(vertices, joint_positions, bones, mesh_faces=None, iterations=5):
        """
        热扩散蒙皮权重（简化版）
        使用距离场 + 拉普拉斯平滑
        """
        n_verts = len(vertices)
        n_bones = len(bones)
        weights = np.zeros((n_verts, n_bones), dtype=np.float32)
        
        # 第一步：计算初始权重（基于距离）
        for i, (jp, jc) in enumerate(bones):
            p1 = joint_positions[jp]
            p2 = joint_positions[jc]
            
            v = p2 - p1
            bone_length = np.linalg.norm(v)
            
            if bone_length < 1e-6:
                dist = np.linalg.norm(vertices - p1, axis=1)
            else:
                v_normalized = v / bone_length
                t = np.clip(np.dot(vertices - p1, v_normalized), 0, bone_length)
                closest_points = p1 + t[:, None] * v_normalized
                dist = np.linalg.norm(vertices - closest_points, axis=1)
            
            # 热核函数
            sigma = bone_length * 1.0
            if sigma < 0.1:
                sigma = 0.5
            weights[:, i] = np.exp(-dist / sigma)
        
        # 第二步：拉普拉斯平滑（使用简单的邻域平均）
        if mesh_faces is not None:
            # 构建邻接表
            adjacency = [set() for _ in range(n_verts)]
            for face in mesh_faces:
                adjacency[face[0]].update([face[1], face[2]])
                adjacency[face[1]].update([face[0], face[2]])
                adjacency[face[2]].update([face[0], face[1]])
            
            # 平滑迭代
            for _ in range(iterations):
                new_weights = weights.copy()
                for v_idx in range(n_verts):
                    if len(adjacency[v_idx]) > 0:
                        neighbors = list(adjacency[v_idx])
                        neighbor_weights = weights[neighbors].mean(axis=0)
                        # 混合当前权重和邻域权重
                        new_weights[v_idx] = 0.7 * weights[v_idx] + 0.3 * neighbor_weights
                weights = new_weights
        
        # 归一化
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-8)
        weights = weights / weight_sum
        
        return weights
    
    @staticmethod
    def enhanced_idw(vertices, joint_positions, bones, power=3.0):
        """
        增强的反距离加权（更强的影响力）
        """
        n_verts = len(vertices)
        n_bones = len(bones)
        weights = np.zeros((n_verts, n_bones), dtype=np.float32)
        
        for i, (jp, jc) in enumerate(bones):
            p1 = joint_positions[jp]
            p2 = joint_positions[jc]
            
            v = p2 - p1
            bone_length = np.linalg.norm(v)
            
            if bone_length < 1e-6:
                dist = np.linalg.norm(vertices - p1, axis=1)
            else:
                v_normalized = v / bone_length
                t = np.clip(np.dot(vertices - p1, v_normalized), 0, bone_length)
                closest_points = p1 + t[:, None] * v_normalized
                dist = np.linalg.norm(vertices - closest_points, axis=1)
            
            # 防止除零
            dist = np.maximum(dist, 1e-4)
            
            # 增强的 IDW：使用更高的幂次（默认3.0）
            weights[:, i] = 1.0 / (dist ** power)
        
        # 归一化
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-8)
        weights = weights / weight_sum
        
        return weights
    
    @staticmethod
    def rigid_skinning(vertices, joint_positions, bones):
        """
        刚性蒙皮（每个顶点只受最近骨骼影响）
        """
        n_verts = len(vertices)
        n_bones = len(bones)
        weights = np.zeros((n_verts, n_bones), dtype=np.float32)
        
        # 计算每个顶点到每根骨骼的距离
        distances = np.zeros((n_verts, n_bones))
        
        for i, (jp, jc) in enumerate(bones):
            p1 = joint_positions[jp]
            p2 = joint_positions[jc]
            
            v = p2 - p1
            bone_length = np.linalg.norm(v)
            
            if bone_length < 1e-6:
                dist = np.linalg.norm(vertices - p1, axis=1)
            else:
                v_normalized = v / bone_length
                t = np.clip(np.dot(vertices - p1, v_normalized), 0, bone_length)
                closest_points = p1 + t[:, None] * v_normalized
                dist = np.linalg.norm(vertices - closest_points, axis=1)
            
            distances[:, i] = dist
        
        # 每个顶点只受最近的骨骼影响
        closest_bone = np.argmin(distances, axis=1)
        for v_idx in range(n_verts):
            weights[v_idx, closest_bone[v_idx]] = 1.0
        
        return weights


class SkeletonRiggingUI(QMainWindow):
    """骨架绑定交互式 UI - 修复点选版"""
    
    def __init__(self):
        super().__init__()
        
        # 数据存储
        self.mesh = None
        self.skeleton = None
        self.bones = []
        self.weights = None
        self.joint_positions = None
        self.G_bind_inv = None
        
        # 关节变换状态
        self.joint_transforms = None
        self.selected_joint = None
        
        # 可视化对象
        self.mesh_actor = None
        self.joint_actors = []
        self.bone_actors = []
        self.gizmo_actors = []
        
        # ===== 修改1：新增存储关节球体 actor 映射 =====
        self.joint_sphere_actors = {}  # {actor: joint_index}
        
        # 性能优化
        self.deformed_vertices = None
        self.need_update = True
        
        # 添加更新定时器（延迟更新，减少卡顿）
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.delayed_render)
        self.update_delay = 50  # 50ms 延迟
        
        # 绑定方法
        self.binding_method = "geodesic"  # 默认使用测地线
        
        self.init_ui()
        QTimer.singleShot(100, self.auto_load_cow)
    
    def init_ui(self):
        """初始化 UI 界面"""
        self.setWindowTitle("骨架绑定交互式编辑器 - 修复点选版")
        self.setGeometry(100, 100, 1600, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 左侧控制面板（使用滚动区域）
        control_panel = self.create_control_panel()
        
        # 右侧 3D 视图
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        
        try:
            self.plotter.enable_anti_aliasing()
        except:
            pass
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.plotter.interactor)
        splitter.setSizes([400, 1200])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar().showMessage("准备就绪 - 从关节列表选择关节进行编辑")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_control_panel(self):
        """创建左侧控制面板"""
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(380)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # ========== 1. 关节列表（放在最上面，增大高度）==========
        joint_list_group = QGroupBox("📝 关节列表（点击选择）")
        joint_list_layout = QVBoxLayout()
        
        self.joint_list = QListWidget()
        self.joint_list.setMinimumHeight(250)
        self.joint_list.itemClicked.connect(self.on_joint_list_clicked)
        self.joint_list.setStyleSheet("""
            QListWidget {
                font-size: 13px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QListWidget::item:hover {
                background-color: #e8f5e9;
            }
        """)
        joint_list_layout.addWidget(self.joint_list)
        
        hint_label = QLabel(
            "<small>💡 <b>提示：</b>点击上方列表或3D视图中的红色球体选择关节</small>"
        )
        hint_label.setWordWrap(True)
        joint_list_layout.addWidget(hint_label)
        
        joint_list_group.setLayout(joint_list_layout)
        layout.addWidget(joint_list_group)
        
        # ========== 2. 当前选中关节信息 ==========
        joint_info_group = QGroupBox("🎯 当前选中关节")
        joint_info_layout = QVBoxLayout()
        
        self.selected_joint_label = QLabel(
            "<i>未选中关节</i>"
        )
        self.selected_joint_label.setWordWrap(True)
        self.selected_joint_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        joint_info_layout.addWidget(self.selected_joint_label)
        
        joint_info_group.setLayout(joint_info_layout)
        layout.addWidget(joint_info_group)
        
        # ========== 3. 关节控制（折叠） ==========
        joint_control_group = QGroupBox("🎮 关节控制")
        joint_control_layout = QVBoxLayout()
        
        # === 旋转控制 ===
        joint_control_layout.addWidget(QLabel("<b>🔄 旋转：</b>"))
        
        # X 轴旋转
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.rotation_x_slider = QSlider(Qt.Horizontal)
        self.rotation_x_slider.setRange(-180, 180)
        self.rotation_x_slider.setValue(0)
        self.rotation_x_slider.setTickInterval(45)
        self.rotation_x_slider.valueChanged.connect(self.on_rotation_changed)
        x_layout.addWidget(self.rotation_x_slider)
        self.rotation_x_label = QLabel("0°")
        self.rotation_x_label.setFixedWidth(50)
        x_layout.addWidget(self.rotation_x_label)
        joint_control_layout.addLayout(x_layout)
        
        # Y 轴旋转
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.rotation_y_slider = QSlider(Qt.Horizontal)
        self.rotation_y_slider.setRange(-180, 180)
        self.rotation_y_slider.setValue(0)
        self.rotation_y_slider.setTickInterval(45)
        self.rotation_y_slider.valueChanged.connect(self.on_rotation_changed)
        y_layout.addWidget(self.rotation_y_slider)
        self.rotation_y_label = QLabel("0°")
        self.rotation_y_label.setFixedWidth(50)
        y_layout.addWidget(self.rotation_y_label)
        joint_control_layout.addLayout(y_layout)
        
        # Z 轴旋转
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.rotation_z_slider = QSlider(Qt.Horizontal)
        self.rotation_z_slider.setRange(-180, 180)
        self.rotation_z_slider.setValue(0)
        self.rotation_z_slider.setTickInterval(45)
        self.rotation_z_slider.valueChanged.connect(self.on_rotation_changed)
        z_layout.addWidget(self.rotation_z_slider)
        self.rotation_z_label = QLabel("0°")
        self.rotation_z_label.setFixedWidth(50)
        z_layout.addWidget(self.rotation_z_label)
        joint_control_layout.addLayout(z_layout)
        
        # === 平移控制 ===
        joint_control_layout.addWidget(QLabel("<b>📍 平移：</b>"))
        
        # X 轴平移
        tx_layout = QHBoxLayout()
        tx_layout.addWidget(QLabel("X:"))
        self.translation_x_slider = QSlider(Qt.Horizontal)
        self.translation_x_slider.setRange(-100, 100)
        self.translation_x_slider.setValue(0)
        self.translation_x_slider.setTickInterval(20)
        self.translation_x_slider.sliderReleased.connect(self.on_translation_slider_released)
        self.translation_x_slider.valueChanged.connect(self.on_translation_increment)
        tx_layout.addWidget(self.translation_x_slider)
        self.translation_x_label = QLabel("0.00")
        self.translation_x_label.setFixedWidth(60)
        self.translation_x_label.setStyleSheet("font-family: monospace;")
        tx_layout.addWidget(self.translation_x_label)
        joint_control_layout.addLayout(tx_layout)
        
        # Y 轴平移
        ty_layout = QHBoxLayout()
        ty_layout.addWidget(QLabel("Y:"))
        self.translation_y_slider = QSlider(Qt.Horizontal)
        self.translation_y_slider.setRange(-100, 100)
        self.translation_y_slider.setValue(0)
        self.translation_y_slider.setTickInterval(20)
        self.translation_y_slider.sliderReleased.connect(self.on_translation_slider_released)
        self.translation_y_slider.valueChanged.connect(self.on_translation_increment)
        ty_layout.addWidget(self.translation_y_slider)
        self.translation_y_label = QLabel("0.00")
        self.translation_y_label.setFixedWidth(60)
        self.translation_y_label.setStyleSheet("font-family: monospace;")
        ty_layout.addWidget(self.translation_y_label)
        joint_control_layout.addLayout(ty_layout)
        
        # Z 轴平移
        tz_layout = QHBoxLayout()
        tz_layout.addWidget(QLabel("Z:"))
        self.translation_z_slider = QSlider(Qt.Horizontal)
        self.translation_z_slider.setRange(-100, 100)
        self.translation_z_slider.setValue(0)
        self.translation_z_slider.setTickInterval(20)
        self.translation_z_slider.sliderReleased.connect(self.on_translation_slider_released)
        self.translation_z_slider.valueChanged.connect(self.on_translation_increment)
        tz_layout.addWidget(self.translation_z_slider)
        self.translation_z_label = QLabel("0.00")
        self.translation_z_label.setFixedWidth(60)
        self.translation_z_label.setStyleSheet("font-family: monospace;")
        tz_layout.addWidget(self.translation_z_label)
        joint_control_layout.addLayout(tz_layout)
        
        # 平移步长
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("步长:"))
        self.translation_step_slider = QSlider(Qt.Horizontal)
        self.translation_step_slider.setRange(1, 50)
        self.translation_step_slider.setValue(10)
        self.translation_step_slider.setTickInterval(10)
        step_layout.addWidget(self.translation_step_slider)
        self.translation_step_label = QLabel("0.10")
        self.translation_step_label.setFixedWidth(60)
        self.translation_step_label.setStyleSheet("font-family: monospace;")
        self.translation_step_slider.valueChanged.connect(
            lambda v: self.translation_step_label.setText(f"{v/100:.2f}")
        )
        step_layout.addWidget(self.translation_step_label)
        joint_control_layout.addLayout(step_layout)
        
        joint_control_group.setLayout(joint_control_layout)
        layout.addWidget(joint_control_group)
        
        # ========== 4. 快捷操作 ==========
        action_group = QGroupBox("⚡ 快捷操作")
        action_layout = QVBoxLayout()
        
        reset_current_btn = QPushButton("🔄 重置当前关节")
        reset_current_btn.clicked.connect(self.reset_current_joint)
        action_layout.addWidget(reset_current_btn)
        
        reset_all_btn = QPushButton("🔄 重置所有关节")
        reset_all_btn.clicked.connect(self.reset_all_joints)
        action_layout.addWidget(reset_all_btn)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # ========== 5. 绑定方法 ==========
        binding_group = QGroupBox("🔗 绑定方法")
        binding_layout = QVBoxLayout()
        
        binding_method_layout = QHBoxLayout()
        binding_method_layout.addWidget(QLabel("方法:"))
        self.binding_method_combo = QComboBox()
        self.binding_method_combo.addItems([
            "测地线蒙皮 (推荐)",
            "热扩散",
            "增强IDW",
            "刚性绑定",
            "原始IDW"
        ])
        self.binding_method_combo.setCurrentIndex(0)
        binding_method_layout.addWidget(self.binding_method_combo)
        binding_layout.addLayout(binding_method_layout)
        
        rebind_btn = QPushButton("🔄 重新绑定")
        rebind_btn.clicked.connect(self.rebind_mesh)
        binding_layout.addWidget(rebind_btn)
        
        binding_group.setLayout(binding_layout)
        layout.addWidget(binding_group)
        
        # ========== 6. 显示选项 ==========
        display_group = QGroupBox("👁️ 显示选项")
        display_layout = QVBoxLayout()
        
        self.show_mesh_checkbox = QCheckBox("显示网格")
        self.show_mesh_checkbox.setChecked(True)
        self.show_mesh_checkbox.stateChanged.connect(self.on_display_changed)
        display_layout.addWidget(self.show_mesh_checkbox)
        
        self.show_skeleton_checkbox = QCheckBox("显示骨架")
        self.show_skeleton_checkbox.setChecked(True)
        self.show_skeleton_checkbox.stateChanged.connect(self.on_display_changed)
        display_layout.addWidget(self.show_skeleton_checkbox)
        
        self.show_joints_checkbox = QCheckBox("显示关节点")
        self.show_joints_checkbox.setChecked(True)
        self.show_joints_checkbox.stateChanged.connect(self.on_display_changed)
        display_layout.addWidget(self.show_joints_checkbox)
        
        self.show_gizmo_checkbox = QCheckBox("显示 Gizmo 轴向")
        self.show_gizmo_checkbox.setChecked(True)
        self.show_gizmo_checkbox.stateChanged.connect(self.on_display_changed)
        display_layout.addWidget(self.show_gizmo_checkbox)
        
        self.show_labels_checkbox = QCheckBox("显示关节标签")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.stateChanged.connect(self.on_display_changed)
        display_layout.addWidget(self.show_labels_checkbox)
        
        self.lightweight_mode_checkbox = QCheckBox("轻量级模式")
        self.lightweight_mode_checkbox.setChecked(True)
        self.lightweight_mode_checkbox.setToolTip("开启后，拖动滑块时只更新关节和骨架，松手后才变形网格")
        display_layout.addWidget(self.lightweight_mode_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # ========== 7. 模型信息 ==========
        info_group = QGroupBox("📊 模型信息")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("未加载模型")
        self.info_label.setStyleSheet("font-size: 11px;")
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # ========== 8. 导出功能 ==========
        export_group = QGroupBox("💾 导出")
        export_layout = QVBoxLayout()
        
        export_skeleton_btn = QPushButton("导出骨架")
        export_skeleton_btn.clicked.connect(self.export_skeleton_info)
        export_layout.addWidget(export_skeleton_btn)
        
        export_weights_btn = QPushButton("导出权重")
        export_weights_btn.clicked.connect(self.export_weights_info)
        export_layout.addWidget(export_weights_btn)
        
        export_mesh_btn = QPushButton("导出网格")
        export_mesh_btn.clicked.connect(self.export_deformed_mesh)
        export_layout.addWidget(export_mesh_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        scroll_area.setWidget(panel)
        return scroll_area
    
    def auto_load_cow(self):
        """自动加载牛模型 - 修复版"""
        glb_path = "data/cow/cow.glb"
        
        try:
            self.statusBar().showMessage("正在加载模型...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            
            # 加载网格
            vertices, faces = load_mesh_from_glb(glb_path, scale=1.0)
            self.mesh = Mesh()
            self.mesh.set_vertices_faces(vertices, faces)
            
            # 加载骨架
            self.skeleton, self.bones = load_skeleton_from_glb(
                glb_path, scale=1.0, verbose=False
            )
            
            # 计算权重（使用选定的方法）
            self.compute_weights()
            
            # 计算绑定姿态逆矩阵
            bind_local_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(
                self.skeleton.n, axis=0
            )
            G_bind = self.skeleton.global_from_local(bind_local_transforms)
            self.G_bind_inv = np.linalg.inv(G_bind)
            
            # 初始化为单位矩阵（增量变换）
            self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(
                self.skeleton.n, axis=0
            )
            self.deformed_vertices = self.mesh.v.copy()
            
            # 更新 UI
            self.update_info_label()
            self.update_joint_list()
            
            # ===== 修改2：启用网格点击（可以点击球体）=====
            self.plotter.enable_mesh_picking(
                callback=self.on_mesh_picked,
                show_message=False,
                use_picker=True
            )
            
            self.render_scene()
            
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage(
                f"✅ 加载成功：{self.skeleton.n} 个关节，{self.mesh.v.shape[0]} 个顶点"
            )
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")
            import traceback
            traceback.print_exc()
    
    def compute_weights(self):
        """根据选定的方法计算权重"""
        self.joint_positions = self.skeleton.bind_positions()
        method_idx = self.binding_method_combo.currentIndex()
        
        self.statusBar().showMessage("正在计算权重...")
        
        if method_idx == 0:  # 测地线蒙皮
            self.weights = ImprovedWeights.geodesic_skinning(
                self.mesh.v, self.joint_positions, self.bones, self.mesh.f
            )
        elif method_idx == 1:  # 热扩散
            self.weights = ImprovedWeights.heat_diffusion(
                self.mesh.v, self.joint_positions, self.bones, self.mesh.f, iterations=5
            )
        elif method_idx == 2:  # 增强IDW
            self.weights = ImprovedWeights.enhanced_idw(
                self.mesh.v, self.joint_positions, self.bones, power=3.0
            )
        elif method_idx == 3:  # 刚性绑定
            self.weights = ImprovedWeights.rigid_skinning(
                self.mesh.v, self.joint_positions, self.bones
            )
        else:  # 原始IDW
            self.weights = idw_two_bones(
                self.mesh.v, self.joint_positions, self.bones
            )
        
        self.need_update = True
    
    def rebind_mesh(self):
        """重新绑定网格"""
        if self.mesh is None or self.skeleton is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        reply = QMessageBox.question(
            self, '确认', 
            f'确定使用 "{self.binding_method_combo.currentText()}" 重新绑定？\n'
            '这将重置所有关节变换。',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.compute_weights()
            
            # 重置关节变换（单位矩阵）
            self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(
                self.skeleton.n, axis=0
            )
            
            if self.selected_joint is not None:
                self.update_sliders_from_transform()
            
            self.render_scene()
            self.statusBar().showMessage(
                f"✅ 重新绑定完成 (方法: {self.binding_method_combo.currentText()})"
            )
    
    def update_info_label(self):
        """更新模型信息"""
        if self.mesh and self.skeleton:
            info = f"""
<b>网格：</b><br>
• 顶点: {self.mesh.v.shape[0]}<br>
• 面片: {self.mesh.f.shape[0]}<br>
<br>
<b>骨架：</b><br>
• 关节: {self.skeleton.n}<br>
• 骨骼: {len(self.bones)}<br>
            """
            self.info_label.setText(info)
    
    def update_joint_list(self):
        """更新关节列表"""
        self.joint_list.clear()
        for i, joint in enumerate(self.skeleton.joints):
            self.joint_list.addItem(f"[{i:2d}] {joint.name}")
    
    def compute_deformed_vertices(self):
        """计算变形顶点 - 修复版"""
        if not self.need_update:
            return self.deformed_vertices
        
        # 1. 获取绑定姿态的局部变换
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(
            self.skeleton.n, axis=0
        )
        
        # 2. 将增量变换应用到绑定姿态
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        
        # 3. 通过FK计算全局变换
        G_current = self.skeleton.global_from_local(current_local)
        
        # 4. 应用LBS
        self.deformed_vertices = apply_lbs(
            self.mesh.v, self.weights, self.bones, G_current, self.G_bind_inv
        )
        
        self.need_update = False
        return self.deformed_vertices
    
    def render_scene(self):
        """渲染场景（完整版） - 修复版"""
        self.plotter.clear()
        
        # ===== 修改4a：清空映射 =====
        self.joint_sphere_actors = {}
        
        # 计算当前局部变换
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(
            self.skeleton.n, axis=0
        )
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        
        # 计算全局变换
        G_current = self.skeleton.global_from_local(current_local)
        current_joint_positions = G_current[:, :3, 3]
        
        # 计算变形顶点
        deformed_vertices = self.compute_deformed_vertices()
        
        # 计算球体半径
        mesh_size = np.linalg.norm(deformed_vertices.max(axis=0) - deformed_vertices.min(axis=0))
        base_sphere_radius = mesh_size * 0.015 * 2
        
        # 1. 绘制网格
        if self.show_mesh_checkbox.isChecked():
            faces_with_count = np.hstack([
                np.full((len(self.mesh.f), 1), 3), 
                self.mesh.f
            ])
            mesh_pv = pv.PolyData(deformed_vertices, faces_with_count)
            
            # ===== 修改6a：网格不可点击 =====
            self.mesh_actor = self.plotter.add_mesh(
                mesh_pv,
                color='lightblue',
                opacity=0.4,
                show_edges=True,
                edge_color='navy',
                line_width=0.2,
                smooth_shading=True,
                pickable=False  # 网格不可点击
            )
        
        # 2. 绘制骨架
        if self.show_skeleton_checkbox.isChecked():
            self.bone_actors = []
            for jp, jc in self.bones:
                p1 = current_joint_positions[jp]
                p2 = current_joint_positions[jc]
                line = pv.Line(p1, p2)
                
                # ===== 修改6b：骨骼不可点击 =====
                actor = self.plotter.add_mesh(
                    line, 
                    color='darkred', 
                    line_width=8, 
                    opacity=0.9,
                    pickable=False  # 骨骼不可点击
                )
                self.bone_actors.append(actor)
        
        # ===== 修改4b：绘制关节点（可点击的球体）=====
        if self.show_joints_checkbox.isChecked():
            self.joint_actors = []
            
            # 为每个关节创建单独的球体
            for i, pos in enumerate(current_joint_positions):
                # 创建球体
                sphere = pv.Sphere(
                    radius=base_sphere_radius * 0.3,
                    center=pos.tolist(),
                    theta_resolution=16,
                    phi_resolution=16
                )
                
                # 判断颜色（选中的关节用黄色）
                if i == self.selected_joint:
                    color = 'yellow'
                else:
                    color = 'red'
                
                # 添加球体并设置为可点击
                actor = self.plotter.add_mesh(
                    sphere,
                    color=color,
                    opacity=0.9,
                    pickable=True,  # 可点击
                    lighting=True
                )
                
                self.joint_actors.append(actor)
                
                # 存储 actor 到关节索引的映射
                self.joint_sphere_actors[actor] = i
            
            # 显示标签
            if self.selected_joint is not None and self.show_labels_checkbox.isChecked():
                pos = current_joint_positions[self.selected_joint]
                joint = self.skeleton.joints[self.selected_joint]
                label_pos = pos + np.array([0, base_sphere_radius * 3, 0])
                self.plotter.add_point_labels(
                    [label_pos],
                    [f"[{self.selected_joint}] {joint.name}"],
                    font_size=14,
                    bold=True,
                    text_color='black',
                    point_color='yellow',
                    point_size=20,
                    shape_opacity=0.8
                )
        
        # 4. 绘制 Gizmo
        if self.selected_joint is not None and self.show_gizmo_checkbox.isChecked():
            self.render_gizmo(current_joint_positions[self.selected_joint], base_sphere_radius)
        
        # 坐标轴
        if not hasattr(self, '_axes_added'):
            self.plotter.add_axes(interactive=False, line_width=3)
            self._axes_added = True
        
        # 相机
        if not hasattr(self, '_camera_set'):
            self.plotter.reset_camera()
            self.plotter.camera.elevation = 15
            self.plotter.camera.azimuth = -60
            self.plotter.camera.zoom(1.2)
            self._camera_set = True
        
        self.plotter.update()
    
    def render_scene_lightweight(self):
        """轻量级渲染 - 修复版"""
        for actor in self.joint_actors:
            self.plotter.remove_actor(actor)
        for actor in self.bone_actors:
            self.plotter.remove_actor(actor)
        for actor in self.gizmo_actors:
            self.plotter.remove_actor(actor)
        
        self.joint_actors = []
        self.bone_actors = []
        self.gizmo_actors = []
        
        # ===== 修改5a：清空映射 =====
        self.joint_sphere_actors = {}
        
        # 计算当前变换
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(
            self.skeleton.n, axis=0
        )
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        
        G_current = self.skeleton.global_from_local(current_local)
        current_joint_positions = G_current[:, :3, 3]
        
        if self.mesh is not None:
            mesh_size = np.linalg.norm(self.mesh.v.max(axis=0) - self.mesh.v.min(axis=0))
            base_sphere_radius = mesh_size * 0.015 * 2
        else:
            base_sphere_radius = 0.1
        
        # 绘制骨架
        if self.show_skeleton_checkbox.isChecked():
            for jp, jc in self.bones:
                p1 = current_joint_positions[jp]
                p2 = current_joint_positions[jc]
                line = pv.Line(p1, p2)
                actor = self.plotter.add_mesh(
                    line, 
                    color='darkred', 
                    line_width=8, 
                    opacity=0.9,
                    pickable=False
                )
                self.bone_actors.append(actor)
        
        # ===== 修改5b：绘制关节点（单独的球体）=====
        if self.show_joints_checkbox.isChecked():
            for i, pos in enumerate(current_joint_positions):
                sphere = pv.Sphere(
                    radius=base_sphere_radius * 0.3,
                    center=pos.tolist(),
                    theta_resolution=16,
                    phi_resolution=16
                )
                
                if i == self.selected_joint:
                    color = 'yellow'
                else:
                    color = 'red'
                
                actor = self.plotter.add_mesh(
                    sphere,
                    color=color,
                    opacity=0.9,
                    pickable=True,
                    lighting=True
                )
                
                self.joint_actors.append(actor)
                self.joint_sphere_actors[actor] = i
            
            if self.selected_joint is not None and self.show_labels_checkbox.isChecked():
                pos = current_joint_positions[self.selected_joint]
                joint = self.skeleton.joints[self.selected_joint]
                label_pos = pos + np.array([0, base_sphere_radius * 3, 0])
                self.plotter.add_point_labels(
                    [label_pos],
                    [f"[{self.selected_joint}] {joint.name}"],
                    font_size=14,
                    bold=True,
                    text_color='black',
                    point_color='yellow',
                    point_size=20,
                    shape_opacity=0.8
                )
        
        if self.selected_joint is not None and self.show_gizmo_checkbox.isChecked():
            self.render_gizmo(current_joint_positions[self.selected_joint], base_sphere_radius)
        
        self.plotter.update()
    
    def render_gizmo(self, center_pos, base_radius):
        """绘制 Gizmo"""
        self.gizmo_actors = []
        
        if isinstance(center_pos, np.ndarray):
            center_pos = center_pos.tolist()
        
        base_radius = float(base_radius)
        gizmo_scale = 2.5
        
        arrow_length = base_radius * gizmo_scale * 0.6
        cone_height = base_radius * gizmo_scale * 0.25
        shaft_radius = base_radius * gizmo_scale * 0.025
        cone_radius = base_radius * gizmo_scale * 0.06
        
        axes_data = [
            {'direction': np.array([1, 0, 0]), 'color': 'red'},
            {'direction': np.array([0, 1, 0]), 'color': 'green'},
            {'direction': np.array([0, 0, 1]), 'color': 'blue'}
        ]
        
        for axis in axes_data:
            direction = axis['direction']
            
            shaft_start = np.array(center_pos)
            shaft_end = shaft_start + direction * arrow_length
            
            shaft_line = pv.Line(shaft_start.tolist(), shaft_end.tolist())
            shaft_tube = shaft_line.tube(radius=float(shaft_radius), n_sides=8)
            
            self.gizmo_actors.append(
                self.plotter.add_mesh(
                    shaft_tube,
                    color=axis['color'],
                    opacity=0.85,
                    lighting=True
                )
            )
            
            cone_base = shaft_end
            
            cone = pv.Cone(
                center=cone_base.tolist(),
                direction=direction.tolist(),
                height=float(cone_height),
                radius=float(cone_radius),
                resolution=8,
                capping=True
            )
            
            self.gizmo_actors.append(
                self.plotter.add_mesh(
                    cone,
                    color=axis['color'],
                    opacity=0.85,
                    lighting=True
                )
            )
    
    # ===== 修改3：添加新的回调函数 =====
    def on_mesh_picked(self, mesh):
        """网格/物体点击回调 - 新版"""
        # 获取点击的 actor
        picker = self.plotter.picker
        actor = picker.GetActor()
        
        if actor is None:
            return
        
        # 检查是否点击了关节球体
        if actor in self.joint_sphere_actors:
            joint_idx = self.joint_sphere_actors[actor]
            self.selected_joint = joint_idx
            self.update_selected_joint_info()
            self.update_sliders_from_transform()
            self.render_scene()
            
            self.statusBar().showMessage(
                f"✅ 选中关节 [{joint_idx}] {self.skeleton.joints[joint_idx].name}"
            )
        else:
            # 点击了其他物体
            self.statusBar().showMessage("⚠️ 请点击红色/黄色关节球来选择关节")
    
    def on_point_picked(self, picked_point):
        """点击回调 - 备用版（保留兼容）"""
        if picked_point is None:
            return
        
        # 计算当前关节位置
        bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(
            self.skeleton.n, axis=0
        )
        current_local = np.zeros_like(bind_local)
        for i in range(self.skeleton.n):
            current_local[i] = bind_local[i] @ self.joint_transforms[i]
        
        G_current = self.skeleton.global_from_local(current_local)
        current_positions = G_current[:, :3, 3]
        
        distances = np.linalg.norm(current_positions - picked_point, axis=1)
        closest_joint = np.argmin(distances)
        
        if distances[closest_joint] < 0.5:
            self.selected_joint = closest_joint
            self.update_selected_joint_info()
            self.update_sliders_from_transform()
            self.render_scene()
            
            self.statusBar().showMessage(
                f"✅ 选中关节 [{closest_joint}] {self.skeleton.joints[closest_joint].name}"
            )
    
    def on_joint_list_clicked(self, item):
        """关节列表点击"""
        text = item.text()
        idx = int(text.split(']')[0][1:])
        self.selected_joint = idx
        self.update_selected_joint_info()
        self.update_sliders_from_transform()
        self.render_scene()
    
    def update_selected_joint_info(self):
        """更新选中关节信息"""
        if self.selected_joint is None:
            self.selected_joint_label.setText("<i>未选中关节</i>")
            self.joint_list.clearSelection()
            
            self.rotation_x_slider.setEnabled(False)
            self.rotation_y_slider.setEnabled(False)
            self.rotation_z_slider.setEnabled(False)
            self.translation_x_slider.setEnabled(False)
            self.translation_y_slider.setEnabled(False)
            self.translation_z_slider.setEnabled(False)
            self.translation_step_slider.setEnabled(False)
        else:
            joint = self.skeleton.joints[self.selected_joint]
            parent = (self.skeleton.joints[joint.parent].name 
                     if joint.parent >= 0 else "无")
            
            self.selected_joint_label.setText(
                f"<b>索引：</b>{self.selected_joint}<br>"
                f"<b>名称：</b>{joint.name}<br>"
                f"<b>父节点：</b>{parent}"
            )
            self.joint_list.setCurrentRow(self.selected_joint)
            
            self.rotation_x_slider.setEnabled(True)
            self.rotation_y_slider.setEnabled(True)
            self.rotation_z_slider.setEnabled(True)
            self.translation_x_slider.setEnabled(True)
            self.translation_y_slider.setEnabled(True)
            self.translation_z_slider.setEnabled(True)
            self.translation_step_slider.setEnabled(True)
    
    def update_sliders_from_transform(self):
        """从变换更新滑块"""
        if self.selected_joint is None:
            return
        
        transform = self.joint_transforms[self.selected_joint]
        
        R = transform[:3, :3]
        angles = self.rotation_matrix_to_euler(R)
        
        self.rotation_x_slider.blockSignals(True)
        self.rotation_y_slider.blockSignals(True)
        self.rotation_z_slider.blockSignals(True)
        
        self.rotation_x_slider.setValue(int(np.degrees(angles[0])))
        self.rotation_y_slider.setValue(int(np.degrees(angles[1])))
        self.rotation_z_slider.setValue(int(np.degrees(angles[2])))
        
        self.rotation_x_label.setText(f"{int(np.degrees(angles[0]))}°")
        self.rotation_y_label.setText(f"{int(np.degrees(angles[1]))}°")
        self.rotation_z_label.setText(f"{int(np.degrees(angles[2]))}°")
        
        self.rotation_x_slider.blockSignals(False)
        self.rotation_y_slider.blockSignals(False)
        self.rotation_z_slider.blockSignals(False)
        
        t = transform[:3, 3]
        
        self.translation_x_label.setText(f"{t[0]:.2f}")
        self.translation_y_label.setText(f"{t[1]:.2f}")
        self.translation_z_label.setText(f"{t[2]:.2f}")
        
        self.translation_x_slider.blockSignals(True)
        self.translation_y_slider.blockSignals(True)
        self.translation_z_slider.blockSignals(True)
        
        self.translation_x_slider.setValue(0)
        self.translation_y_slider.setValue(0)
        self.translation_z_slider.setValue(0)
        
        self.translation_x_slider.blockSignals(False)
        self.translation_y_slider.blockSignals(False)
        self.translation_z_slider.blockSignals(False)
    
    def on_rotation_changed(self):
        """旋转滑块改变"""
        if self.selected_joint is None:
            return
        
        rx = np.radians(self.rotation_x_slider.value())
        ry = np.radians(self.rotation_y_slider.value())
        rz = np.radians(self.rotation_z_slider.value())
        
        self.rotation_x_label.setText(f"{self.rotation_x_slider.value()}°")
        self.rotation_y_label.setText(f"{self.rotation_y_slider.value()}°")
        self.rotation_z_label.setText(f"{self.rotation_z_slider.value()}°")
        
        R = self.euler_to_rotation_matrix(rx, ry, rz)
        self.joint_transforms[self.selected_joint][:3, :3] = R
        
        self.need_update = True
        
        self.update_timer.stop()
        self.update_timer.start(self.update_delay)
    
    def on_translation_increment(self):
        """平移滑块增量变化"""
        if self.selected_joint is None:
            return
        
        step = self.translation_step_slider.value() / 100.0 * 5
        
        dx = self.translation_x_slider.value() / 100.0 * step
        dy = self.translation_y_slider.value() / 100.0 * step
        dz = self.translation_z_slider.value() / 100.0 * step
        
        current_t = self.joint_transforms[self.selected_joint][:3, 3]
        new_t = current_t + np.array([dx, dy, dz])
        
        self.joint_transforms[self.selected_joint][:3, 3] = new_t
        
        self.translation_x_label.setText(f"{new_t[0]:.2f}")
        self.translation_y_label.setText(f"{new_t[1]:.2f}")
        self.translation_z_label.setText(f"{new_t[2]:.2f}")
        
        self.need_update = True
        
        self.update_timer.stop()
        self.update_timer.start(self.update_delay)
    
    def on_translation_slider_released(self):
        """滑块释放"""
        self.translation_x_slider.blockSignals(True)
        self.translation_y_slider.blockSignals(True)
        self.translation_z_slider.blockSignals(True)
        
        self.translation_x_slider.setValue(0)
        self.translation_y_slider.setValue(0)
        self.translation_z_slider.setValue(0)
        
        self.translation_x_slider.blockSignals(False)
        self.translation_y_slider.blockSignals(False)
        self.translation_z_slider.blockSignals(False)
        
        self.update_timer.stop()
        self.need_update = True
        self.render_scene()
    
    def delayed_render(self):
        """延迟渲染"""
        if self.lightweight_mode_checkbox.isChecked():
            self.render_scene_lightweight()
        else:
            self.render_scene()
    
    def euler_to_rotation_matrix(self, rx, ry, rz):
        """欧拉角转旋转矩阵"""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        return (Rz @ Ry @ Rx).astype(np.float32)
    
    def rotation_matrix_to_euler(self, R):
        """旋转矩阵转欧拉角"""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def reset_current_joint(self):
        """重置当前关节"""
        if self.selected_joint is None:
            QMessageBox.information(self, "提示", "请先选择关节")
            return
        
        self.joint_transforms[self.selected_joint] = np.eye(4, dtype=np.float32)
        self.need_update = True
        self.update_sliders_from_transform()
        self.render_scene()
        self.statusBar().showMessage(f"✅ 关节 [{self.selected_joint}] 已重置")
    
    def reset_all_joints(self):
        """重置所有关节"""
        reply = QMessageBox.question(
            self, '确认', '确定重置所有关节？',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(
                self.skeleton.n, axis=0
            )
            self.need_update = True
            if self.selected_joint is not None:
                self.update_sliders_from_transform()
            self.render_scene()
            self.statusBar().showMessage("✅ 所有关节已重置")
    
    def on_display_changed(self):
        """显示选项改变"""
        self.render_scene()
    
    def export_skeleton_info(self):
        """导出骨架信息"""
        if self.skeleton is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出骨架", "skeleton_info.json", "JSON (*.json)"
        )
        
        if not file_path:
            return
        
        import json
        
        data = {
            "joints": [],
            "bones": [],
            "transforms": []
        }
        
        for i, joint in enumerate(self.skeleton.joints):
            data["joints"].append({
                "index": i,
                "name": joint.name,
                "position": joint.pos.tolist(),
                "parent": joint.parent
            })
            data["transforms"].append(self.joint_transforms[i].tolist())
        
        for jp, jc in self.bones:
            data["bones"].append({
                "parent": jp,
                "child": jc,
                "parent_name": self.skeleton.joints[jp].name,
                "child_name": self.skeleton.joints[jc].name
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        QMessageBox.information(self, "成功", f"已导出到:\n{file_path}")
        self.statusBar().showMessage(f"✅ 骨架信息已导出")
    
    def export_weights_info(self):
        """导出权重"""
        if self.weights is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出权重", "weights.npz", "NPZ (*.npz)"
        )
        
        if not file_path:
            return
        
        np.savez(
            file_path,
            weights=self.weights,
            bones=np.array(self.bones),
            joint_positions=self.joint_positions
        )
        
        QMessageBox.information(
            self, "成功", 
            f"已导出到:\n{file_path}"
        )
        self.statusBar().showMessage(f"✅ 权重信息已导出")
    
    def export_deformed_mesh(self):
        """导出网格"""
        if self.mesh is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出网格", "mesh.obj", "OBJ (*.obj)"
        )
        
        if not file_path:
            return
        
        vertices = self.compute_deformed_vertices()
        
        with open(file_path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in self.mesh.f:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        QMessageBox.information(self, "成功", f"已导出到:\n{file_path}")
        self.statusBar().showMessage(f"✅ 变形网格已导出")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SkeletonRiggingUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()