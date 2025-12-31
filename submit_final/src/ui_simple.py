"""
骨架绑定 UI - 带工具栏
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

# 动画导出相关
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import platform
import os
import json
from datetime import datetime
from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
import subprocess


class AnimationExporter(QThread):
    """动画导出线程"""
    progress_updated = pyqtSignal(int)
    status_message = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, mesh, skeleton, bones, weights, parent=None):
        super().__init__(parent)
        self.mesh = mesh
        self.skeleton = skeleton
        self.bones = bones
        self.weights = weights
        self.G_bind_inv = None

    def setup_chinese_font(self):
        """配置matplotlib中文字体"""
        system = platform.system()

        if system == 'Darwin':  # macOS
            fonts = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti SC']
        elif system == 'Windows':
            fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
        else:  # Linux
            fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']

        # 尝试设置可用字体
        for font in fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                break
            except:
                continue

        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    def create_rotation_matrix(self, axis, angle):
        """创建绕轴旋转的旋转矩阵"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x, y, z = axis

        R = np.array([
            [cos_a + x*x*(1-cos_a),     x*y*(1-cos_a) - z*sin_a,   x*z*(1-cos_a) + y*sin_a],
            [y*x*(1-cos_a) + z*sin_a,  cos_a + y*y*(1-cos_a),     y*z*(1-cos_a) - x*sin_a],
            [z*x*(1-cos_a) - y*sin_a,  z*y*(1-cos_a) + x*sin_a,   cos_a + z*z*(1-cos_a)]
        ])
        return R

    def create_transform_matrix(self, R=None, t=None):
        """创建4x4变换矩阵"""
        T = np.eye(4, dtype=np.float32)
        if R is not None:
            T[:3, :3] = R
        if t is not None:
            T[:3, 3] = t
        return T

    def get_joint_role(self, skeleton, joint_idx):
        """判断关节的角色"""
        joint = skeleton.joints[joint_idx]
        name = joint.name.lower()

        # 根节点
        if 'rig' in name and joint.parent == -1:
            return 'root'

        # 躯干
        if 'body_bot' in name:
            return 'body_bot'
        if 'body_top' in name:
            return 'body_top'
        if name == 'body':
            return 'body'

        # 左右判断
        is_left = 'left' in name
        is_right = 'right' in name

        # 后腿 (hind)
        if 'leg_hind' in name:
            if 'top0' in name:
                return 'l_hip0' if is_left else 'r_hip0'
            elif 'top1' in name:
                return 'l_hip1' if is_left else 'r_hip1'
            elif 'bot0' in name:
                return 'l_knee0' if is_left else 'r_knee0'
            elif 'bot1' in name:
                return 'l_knee1' if is_left else 'r_knee1'
            elif 'end' in name:
                return 'l_ankle' if is_left else 'r_ankle'

        # 前腿 (front)
        if 'leg_front' in name:
            if 'top0' in name:
                return 'l_shoulder0' if is_left else 'r_shoulder0'
            elif 'top1' in name:
                return 'l_shoulder1' if is_left else 'r_shoulder1'
            elif 'bot0' in name:
                return 'l_elbow0' if is_left else 'r_elbow0'
            elif 'bot1' in name:
                return 'l_elbow1' if is_left else 'r_elbow1'
            elif 'end' in name:
                return 'l_wrist' if is_left else 'r_wrist'

        # 颈部
        if 'neck' in name:
            if '0' in name:
                return 'neck0'
            elif '1' in name:
                return 'neck1'

        # 头部
        if 'head' in name:
            return 'head'

        return 'unknown'

    def create_walking_animation(self, num_frames=120):
        """创建行走动画"""
        animations = []

        for frame_idx in range(num_frames):
            t = (frame_idx / num_frames) * 4 * np.pi  # 2个完整周期
            is_running = frame_idx >= 60  # 前60帧摇头，后60帧奔跑

            local_transforms = []

            for joint_idx in range(self.skeleton.n):
                role = self.get_joint_role(self.skeleton, joint_idx)
                T = np.eye(4, dtype=np.float32)

                if role == 'root':
                    if is_running:
                        # 奔跑时的身体上下移动
                        height_offset = 0.05 * np.sin(t * 2)
                        T[1, 3] = height_offset
                    else:
                        # 摇头时的轻微上下移动
                        height_offset = 0.02 * np.sin(t)
                        T[1, 3] = height_offset

                elif role in ['body_bot', 'body_top', 'body']:
                    if is_running:
                        # 奔跑时的躯干倾斜
                        angle = 0.1 * np.sin(t)
                        R = self.create_rotation_matrix(np.array([0, 0, 1]), angle)
                        T[:3, :3] = R

                # 后腿
                elif role in ['l_hip0', 'r_hip0']:
                    if is_running:
                        phase = 0 if 'l_' in role else np.pi
                        angle = np.sin(t + phase) * 0.4
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role in ['l_hip1', 'r_hip1']:
                    if is_running:
                        phase = 0 if 'l_' in role else np.pi
                        angle = np.sin(t + phase) * 0.2
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role in ['l_knee0', 'r_knee0']:
                    if is_running:
                        phase = 0 if 'l_' in role else np.pi
                        angle = -np.abs(np.sin(t + phase)) * 0.8
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role in ['l_knee1', 'r_knee1']:
                    if is_running:
                        phase = 0 if 'l_' in role else np.pi
                        angle = -np.abs(np.sin(t + phase)) * 0.4
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role in ['l_ankle', 'r_ankle']:
                    if is_running:
                        phase = 0 if 'l_' in role else np.pi
                        angle = np.sin(t + phase) * 0.15
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                # 前腿
                elif role in ['l_shoulder0', 'r_shoulder0']:
                    if is_running:
                        phase = np.pi if 'l_' in role else 0
                        angle = np.sin(t + phase) * 0.3
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role in ['l_shoulder1', 'r_shoulder1']:
                    if is_running:
                        phase = np.pi if 'l_' in role else 0
                        angle = np.sin(t + phase) * 0.15
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role in ['l_elbow0', 'r_elbow0']:
                    if is_running:
                        phase = np.pi if 'l_' in role else 0
                        angle = -np.abs(np.sin(t + phase)) * 0.5
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role in ['l_elbow1', 'r_elbow1']:
                    if is_running:
                        phase = np.pi if 'l_' in role else 0
                        angle = -np.abs(np.sin(t + phase)) * 0.25
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role in ['l_wrist', 'r_wrist']:
                    if is_running:
                        phase = np.pi if 'l_' in role else 0
                        angle = np.sin(t + phase) * 0.1
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                # 颈部
                elif role == 'neck0':
                    if not is_running:
                        angle = np.sin(t * 3) * 0.6
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R
                    else:
                        angle = np.sin(t * 3) * 0.1
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                elif role == 'neck1':
                    if not is_running:
                        angle = np.sin(t * 3) * 0.4
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R
                    else:
                        angle = np.sin(t * 3) * 0.08
                        R = self.create_rotation_matrix(np.array([1, 0, 0]), angle)
                        T[:3, :3] = R

                # 头部
                elif role == 'head':
                    if not is_running:
                        angle = np.sin(t * 2.5) * 0.5
                        R = self.create_rotation_matrix(np.array([0, 0, 1]), angle)
                        T[:3, :3] = R
                    else:
                        angle = np.sin(t * 2.5) * 0.05
                        R = self.create_rotation_matrix(np.array([0, 0, 1]), angle)
                        T[:3, :3] = R

                local_transforms.append(T)

            animations.append(np.array(local_transforms))

        return animations

    def render_frame_with_skeleton(self, vertices, faces, skeleton, G_current, bones, filename, frame_idx):
        """带骨架可视化的渲染函数"""
        # 旋转顶点：将模型绕X轴旋转+90度，使Z轴向上的模型变为Y轴向上
        rotation_angle = np.pi/2  # +90度
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
            [0, np.sin(rotation_angle), np.cos(rotation_angle)]
        ])

        # 应用旋转到网格顶点
        vertices_rotated = vertices @ rotation_matrix.T

        # 计算当前帧的关节位置（从全局变换矩阵中提取）
        joint_positions = G_current[:, :3, 3]  # 提取平移部分
        joint_positions_rotated = joint_positions @ rotation_matrix.T

        # 设置固定的偶数分辨率
        fig_width, fig_height = 10, 8
        dpi = 100

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        # 1. 绘制网格
        triangles = vertices_rotated[faces]
        mesh = Poly3DCollection(triangles, alpha=0.6, facecolors='lightblue',
                               edgecolors='navy', linewidths=0.3)
        ax.add_collection3d(mesh)

        # 2. 绘制关节点
        ax.scatter(joint_positions_rotated[:, 0],
                   joint_positions_rotated[:, 1],
                   joint_positions_rotated[:, 2],
                   c='red', s=80, alpha=0.9, label='Joints', marker='o')

        # 3. 绘制骨骼连接线
        for bone_idx, (jp, jc) in enumerate(bones):
            parent_pos = joint_positions_rotated[jp]
            child_pos = joint_positions_rotated[jc]

            # 绘制骨骼线段
            ax.plot([parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    [parent_pos[2], child_pos[2]],
                    color='darkred', linewidth=4, alpha=0.8)

            # 在骨骼中点添加小圆点
            mid_point = (parent_pos + child_pos) / 2
            ax.scatter([mid_point[0]], [mid_point[1]], [mid_point[2]],
                      c='orange', s=20, alpha=0.7)

        # 设置坐标轴
        ax.set_xlabel('X (Forward/Back)')
        ax.set_ylabel('Y (Up/Down)')
        ax.set_zlabel('Z (Left/Right)')

        # 动态标题
        if frame_idx <= 60:
            title = f'Frame {frame_idx:04d} - Shaking Head'
        else:
            title = f'Frame {frame_idx:04d} - Running'
        ax.set_title(title, fontsize=14)

        # 计算范围
        all_points = np.vstack([vertices_rotated, joint_positions_rotated])
        center = np.mean(all_points, axis=0)
        max_range = np.max(np.abs(all_points - center)) * 1.2

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        # 从侧面观看
        ax.view_init(elev=15, azim=-45)

        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.grid(True, alpha=0.3)

        # 添加图例
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

    def run(self):
        """执行动画导出"""
        try:
            # 设置matplotlib后端为非GUI，避免与PyQt冲突
            import matplotlib
            matplotlib.use('Agg')
            
            self.setup_chinese_font()

            # 计算绑定姿态的变换矩阵
            bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            G_bind = self.skeleton.global_from_local(bind_local)
            self.G_bind_inv = np.linalg.inv(G_bind)

            # 创建动画
            num_frames = 120
            animations = self.create_walking_animation(num_frames)

            # 准备输出目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "output", "frames")
            os.makedirs(output_dir, exist_ok=True)

            self.status_message.emit("[INFO] Rendering animation frames...")
            frame_files = []

            for frame_idx, local_transforms in enumerate(animations):
                try:
                    self.status_message.emit(f"   Rendering frame {frame_idx+1}/{num_frames}...")

                    # 计算当前帧的全局变换
                    G_current = self.skeleton.global_from_local(local_transforms)

                    # 应用LBS变形
                    deformed_vertices = apply_lbs(
                        self.mesh.v, self.weights, self.bones, G_current, self.G_bind_inv
                    )

                    # 渲染帧
                    frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                    self.render_frame_with_skeleton(
                        deformed_vertices, self.mesh.f, self.skeleton, G_current,
                        self.bones, frame_path, frame_idx+1
                    )
                    frame_files.append(frame_path)
                except Exception as e:
                    self.status_message.emit(f"[WARN] Failed to render frame {frame_idx+1}: {str(e)}")
                    continue

            # 生成GIF
            self.status_message.emit("[INFO] Generating GIF animation...")
            gif_path = os.path.join(script_dir, "output", "cow_animation.gif")
            try:
                import imageio.v2 as imageio_v2
                with imageio_v2.get_writer(gif_path, mode='I', duration=0.08) as writer:
                    for frame_file in frame_files:
                        image = imageio_v2.imread(frame_file)
                        writer.append_data(image)
            except ImportError:
                with imageio.get_writer(gif_path, mode='I', duration=0.08) as writer:
                    for frame_file in frame_files:
                        image = imageio.imread(frame_file)
                        writer.append_data(image)

            self.status_message.emit("[OK] GIF animation saved")

            # 尝试生成MP4
            self.status_message.emit("[INFO] Generating MP4 video...")
            mp4_path = os.path.join(script_dir, "output", "cow_animation.mp4")
            try:
                import subprocess
                # 检查ffmpeg是否存在
                try:
                    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                    ffmpeg_available = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    ffmpeg_available = False

                if ffmpeg_available:
                    cmd = [
                        'ffmpeg', '-y', '-framerate', '10', '-i', f'{output_dir}/frame_%04d.png',
                        '-vf', 'scale=1000:800', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', mp4_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0 and os.path.exists(mp4_path):
                        self.status_message.emit("[OK] MP4 video saved")
                    else:
                        # 备用命令
                        cmd_backup = [
                            'ffmpeg', '-y', '-framerate', '10', '-i', f'{output_dir}/frame_%04d.png',
                            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '23', mp4_path
                        ]
                        result = subprocess.run(cmd_backup, capture_output=True, text=True)
                        if result.returncode == 0 and os.path.exists(mp4_path):
                            self.status_message.emit("[OK] MP4 video saved")
                        else:
                            self.status_message.emit("[WARN] MP4 generation failed")
                else:
                    self.status_message.emit("[WARN] ffmpeg not detected, skipping MP4 generation")
            except Exception as e:
                self.status_message.emit(f"[WARN] MP4 generation error: {str(e)}")

            self.finished_signal.emit(True, f"Animation export completed!\nGIF: {gif_path}\nMP4: {mp4_path}")

        except Exception as e:
            error_msg = f"Animation export failed: {str(e)}"
            self.status_message.emit(error_msg)
            self.finished_signal.emit(False, error_msg)


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

        # 导出状态
        self.is_exporting = False

        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("骨架绑定工具")
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
        self.statusBar().showMessage("点击红色球体选择关节，拖拽箭头沿轴移动")
    
    def create_toolbar(self):
        """创建左侧工具栏"""
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
        self.reset_button = QPushButton("重置到初始状态")
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
        
        # ===== 导出功能组 =====
        export_group = QGroupBox("导出功能")
        export_layout = QVBoxLayout()
        
        # 一键导出动画按钮
        self.export_animation_button = QPushButton("一键导出动画")
        self.export_animation_button.setToolTip("基于原始cow文件导出摇头+奔跑动画")
        self.export_animation_button.clicked.connect(self.export_animation)
        export_layout.addWidget(self.export_animation_button)
        
        # 导出骨架信息按钮
        self.export_skeleton_button = QPushButton("导出骨架信息")
        self.export_skeleton_button.clicked.connect(self.export_skeleton_info)
        export_layout.addWidget(self.export_skeleton_button)
        
        # 导出权重信息按钮
        self.export_weights_button = QPushButton("权重导出")
        self.export_weights_button.clicked.connect(self.export_weights_info)
        export_layout.addWidget(self.export_weights_button)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # ===== 介绍和使用方法 =====
        help_group = QGroupBox("介绍和使用方法")
        help_layout = QVBoxLayout()

        help_text = QLabel(
            "【骨架绑定工具】\n"
            "• 基于线性混合蒙皮(LBS)算法\n"
            "• 支持实时交互式关节调整\n"
            "• 支持完整蒙皮和简化蒙皮模式\n\n"

            "【使用方法】\n"
            "1. 点击红色球体选择关节\n"
            "2. 拖拽彩色箭头沿轴移动\n"
            "3. 右键拖拽旋转视角\n"
            "4. 滚轮缩放场景\n\n"

            "【导出功能】\n"
            "• 一键导出动画：生成摇头+奔跑动画\n"
            "• 导出骨架信息：JSON格式骨架数据\n"
            "• 权重导出：顶点-关节权重矩阵"
        )
        help_text.setStyleSheet(
            "font-size: 11px; color: #333; line-height: 1.4; "
            "background-color: #fff; padding: 8px; "
            "border-radius: 3px; border: 1px solid #ddd;"
        )
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        layout.addWidget(help_group)

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
            self.statusBar().showMessage("没有可重置的初始状态")
            return
        
        self.joint_transforms = self.initial_joint_transforms.copy()
        self.selected_joint = None
        self.update_deformed_mesh_only()
        
        self.statusBar().showMessage("已重置到初始状态")
        print("重置到初始状态")
    
    def on_skinning_mode_changed(self, index):
        """蒙皮模式切换"""
        self.skinning_mode = self.skinning_combo.itemData(index)
        self.update_deformed_mesh_only()
        
        mode_name = self.skinning_combo.currentText()
        self.statusBar().showMessage(f"切换到：{mode_name}")
        print(f"蒙皮模式切换为：{self.skinning_mode}")
    
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
                    print(f"开始拖拽 {axis_name.upper()} 轴")
                    return
                
                for sphere_actor, joint_idx in self.joint_sphere_actors.items():
                    if sphere_actor == picked_actor:
                        if self.selected_joint == joint_idx:
                            self.is_dragging = True
                            self.last_mouse_pos = (mouse_x, mouse_y)
                            self.plotter.disable()
                            print(f"开始拖拽关节 [{joint_idx}]")
                        else:
                            self.selected_joint = joint_idx
                            self.update_gizmo_only()
                            joint_name = self.skeleton.joints[joint_idx].name
                            self.statusBar().showMessage(
                                f"选中关节 [{joint_idx}] {joint_name}"
                            )
                            print(f"选中关节 [{joint_idx}] {joint_name}")
                        return
                
                if self.selected_joint is not None:
                    self.selected_joint = None
                    self.update_gizmo_only()
                    self.statusBar().showMessage("点击红色球体选择关节")
            else:
                if self.selected_joint is not None:
                    self.selected_joint = None
                    self.update_gizmo_only()
                    self.statusBar().showMessage("点击红色球体选择关节")
    
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
            print("计算完整权重...")
            self.weights = idw_two_bones(self.mesh.v, joint_positions, self.bones)
            
            # 计算简化权重
            print("计算简化权重...")
            self.simple_weights = self.compute_simple_weights(self.mesh.v, joint_positions)
            
            bind_local = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            G_bind = self.skeleton.global_from_local(bind_local)
            self.G_bind_inv = np.linalg.inv(G_bind)
            
            self.joint_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(self.skeleton.n, axis=0)
            self.initial_joint_transforms = self.joint_transforms.copy()
            
            self.render_scene_full()
            
            self.statusBar().showMessage(
                f"加载成功：{self.skeleton.n} 个关节"
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
        
        print(f"简化权重计算完成")
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


    def export_animation(self):
        """一键导出动画"""
        if self.mesh is None or self.skeleton is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        # 创建导出线程
        self.exporter = AnimationExporter(self.mesh, self.skeleton, self.bones, self.weights)
        self.exporter.progress_updated.connect(self.update_status_progress)
        self.exporter.status_message.connect(self.statusBar().showMessage)
        self.exporter.finished_signal.connect(self.on_animation_export_finished)

        # 启动导出
        self.is_exporting = True
        self.export_animation_button.setEnabled(False)
        self.exporter.start()

    def update_status_progress(self, value):
        """更新状态栏进度"""
        self.statusBar().showMessage(f"[导出中] 进度: {value}%")

    def on_animation_export_finished(self, success, message):
        """动画导出完成回调"""
        self.export_animation_button.setEnabled(True)
        self.is_exporting = False
        if success:
            QMessageBox.information(self, "成功", f"动画导出完成！\n\n{message}")
            self.statusBar().showMessage("[OK] 动画导出完成")
        else:
            QMessageBox.critical(self, "错误", message)
            self.statusBar().showMessage("[ERROR] 动画导出失败")

    def export_skeleton_info(self):
        """导出骨架信息"""
        if self.skeleton is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "导出骨架信息", "", "JSON files (*.json);;All files (*)"
        )
        if not filename:
            return

        try:
            skeleton_data = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "joints": []
            }

            for i, joint in enumerate(self.skeleton.joints):
                joint_info = {
                    "index": i,
                    "name": joint.name,
                    "parent": joint.parent,
                    "position": joint.pos.tolist()
                }
                skeleton_data["joints"].append(joint_info)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(skeleton_data, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f"[OK] 骨架信息已导出：{filename}")
            QMessageBox.information(self, "成功", f"骨架信息已导出到：\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def export_weights_info(self):
        """导出权重信息"""
        if self.weights is None:
            QMessageBox.warning(self, "警告", "请先加载模型和计算权重")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "导出权重信息", "", "JSON files (*.json);;All files (*)"
        )
        if not filename:
            return

        try:
            weights_data = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "num_vertices": self.weights.shape[0],
                "num_bones": self.weights.shape[1],
                "weights": self.weights.tolist()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(weights_data, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f"[OK] 权重信息已导出：{filename}")
            QMessageBox.information(self, "成功", f"权重信息已导出到：\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def closeEvent(self, event):
        """关闭事件处理"""
        if hasattr(self, 'is_exporting') and self.is_exporting:
            reply = QMessageBox.question(
                self, '确认关闭',
                '正在渲染视频，是否要关闭窗口？\n\n关闭窗口将中断正在进行的视频导出。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # 如果用户确认关闭，停止导出线程
                if hasattr(self, 'exporter') and self.exporter.isRunning():
                    self.exporter.terminate()
                    self.exporter.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = OptimizedDragUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()