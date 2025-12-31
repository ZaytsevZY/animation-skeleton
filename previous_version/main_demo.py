# main_demo.py (GLB版本 - 摇头+奔跑动画)
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import platform

# ============ 配置中文字体 ============
def setup_chinese_font():
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

setup_chinese_font()


# 导入自定义模块
from rigging.mesh_io import Mesh
from rigging.skeleton import quadruped_auto_place, Skeleton
from rigging.weights_nearest import hard_nearest_bone_weights, idw_two_bones
from rigging.lbs import apply_lbs
from rigging.skeleton_loader import (
    load_skeleton_from_glb,
    visualize_skeleton_structure,
    load_mesh_from_glb
)



def create_rotation_matrix(axis, angle):
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

def create_transform_matrix(R=None, t=None):
    """创建4x4变换矩阵"""
    T = np.eye(4, dtype=np.float32)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T

def find_joint_by_keywords(skeleton, keywords):
    """
    根据关键字列表查找关节索引（模糊匹配）
    
    Parameters:
    -----------
    skeleton : Skeleton
        骨架对象
    keywords : list of str
        关键字列表，按优先级排序
    
    Returns:
    --------
    int or None
        关节索引，如果未找到返回 None
    """
    joint_name_map = {joint.name.lower(): i for i, joint in enumerate(skeleton.joints)}
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for name, idx in joint_name_map.items():
            if keyword_lower in name:
                return idx
    
    return None

def get_joint_role(skeleton, joint_idx):
    """
    判断关节的角色（根据实际的牛模型骨架结构）
    
    Returns:
    --------
    str : 'root', 'l_hip', 'r_hip', 'l_knee', 'r_knee', etc.
    """
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
        elif 'bot2' in name and 'end' not in name:
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
        elif 'bot2' in name and 'end' not in name:
            return 'l_wrist' if is_left else 'r_wrist'
    
    # 颈部
    if 'neck0' in name:
        return 'neck0'
    if 'neck1' in name:
        return 'neck1'
    
    # 头部
    if 'head0' in name and 'end' not in name:
        return 'head'
    
    # 末端节点忽略
    if '_end' in name:
        return 'end'
    
    return 'unknown'

def create_walking_animation(skeleton, num_frames=60):
    """创建摇头+奔跑动画 - 专门为牛模型优化
    
    前20帧: 站立摇头
    后40帧: 奔跑动画（修正旋转轴方向）
    
    坐标系：
    - X轴：左右方向
    - Y轴：前后方向（负方向是前进方向）
    - Z轴：上下方向（正方向是上）
    """
    print("🎬 创建摇头+奔跑动画（专为牛模型优化）...")
    
    # 分析骨架中每个关节的角色
    joint_roles = [get_joint_role(skeleton, i) for i in range(skeleton.n)]
    
    print(f"   关节角色分配:")
    for i, role in enumerate(joint_roles):
        if role != 'unknown' and role != 'end':
            print(f"      关节{i:2d} ({skeleton.joints[i].name:25s}): {role}")
    
    animations = []
    
    # 定义两个阶段
    shake_frames = 20
    run_frames = num_frames - shake_frames
    
    for frame in range(num_frames):
        # 判断当前阶段
        if frame < shake_frames:
            # 摇头阶段
            t = frame / shake_frames * 2 * np.pi
            is_running = False
            run_progress = 0.0
        else:
            # 奔跑阶段
            t = (frame - shake_frames) / run_frames * 4 * np.pi
            is_running = True
            run_progress = (frame - shake_frames) / run_frames
        
        local_transforms = []
        
        for i in range(skeleton.n):
            role = joint_roles[i]
            T = np.eye(4, dtype=np.float32)  # 默认不动
            
            # ========== 根节点 ==========
            if role == 'root':
                if is_running:
                    # 奔跑时沿Y负方向前进 + Z方向轻微上下起伏
                    forward = np.array([
                        0,  # X方向不动（左右）
                        -run_progress * 0.5,  # Y负方向前进
                        np.sin(t * 4) * 0.02  # Z方向上下起伏
                    ])
                    T = create_transform_matrix(t=forward)
            
            # ========== 躯干 ==========
            elif role in ['body', 'body_bot', 'body_top']:
                if is_running:
                    # 奔跑时绕Z轴轻微扭动（左右摇摆）
                    angle = np.sin(t * 2) * 0.03
                    R = create_rotation_matrix(np.array([0, 0, 1]), angle)
                    T = create_transform_matrix(R=R)
            
            # ========== 后腿髋部 (top0, top1) ==========
            # 腿的前后摆动应该绕X轴旋转（左右轴）
            elif role in ['l_hip0', 'r_hip0']:
                if is_running:
                    phase = 0 if 'l_' in role else np.pi
                    angle = np.sin(t + phase) * 0.35  # 绕X轴旋转实现前后摆动
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            elif role in ['l_hip1', 'r_hip1']:
                if is_running:
                    phase = 0 if 'l_' in role else np.pi
                    angle = np.sin(t + phase) * 0.18
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            # ========== 后腿膝部 (bot0, bot1) ==========
            elif role in ['l_knee0', 'r_knee0']:
                if is_running:
                    phase = 0 if 'l_' in role else np.pi
                    angle = -np.abs(np.sin(t + phase)) * 0.5  # 绕X轴弯曲
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            elif role in ['l_knee1', 'r_knee1']:
                if is_running:
                    phase = 0 if 'l_' in role else np.pi
                    angle = -np.abs(np.sin(t + phase)) * 0.25
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            # ========== 后腿脚踝 (bot2) ==========
            elif role in ['l_ankle', 'r_ankle']:
                if is_running:
                    phase = 0 if 'l_' in role else np.pi
                    angle = np.sin(t + phase) * 0.12
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            # ========== 前腿肩部 (top0, top1) ==========
            elif role in ['l_shoulder0', 'r_shoulder0']:
                if is_running:
                    phase = np.pi if 'l_' in role else 0  # 与后腿相反相位
                    angle = np.sin(t + phase) * 0.25  # 绕X轴旋转
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            elif role in ['l_shoulder1', 'r_shoulder1']:
                if is_running:
                    phase = np.pi if 'l_' in role else 0
                    angle = np.sin(t + phase) * 0.12
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            # ========== 前腿肘部 (bot0, bot1) ==========
            elif role in ['l_elbow0', 'r_elbow0']:
                if is_running:
                    phase = np.pi if 'l_' in role else 0
                    angle = -np.abs(np.sin(t + phase)) * 0.4
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            elif role in ['l_elbow1', 'r_elbow1']:
                if is_running:
                    phase = np.pi if 'l_' in role else 0
                    angle = -np.abs(np.sin(t + phase)) * 0.2
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            # ========== 前腿手腕 (bot2) ==========
            elif role in ['l_wrist', 'r_wrist']:
                if is_running:
                    phase = np.pi if 'l_' in role else 0
                    angle = np.sin(t + phase) * 0.08
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            # ========== 颈部 ==========
            # 点头应该绕X轴旋转
            elif role == 'neck0':
                if not is_running:
                    # 摇头阶段：绕X轴大幅上下点头
                    angle = np.sin(t * 3) * 0.5
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
                else:
                    # 奔跑阶段：轻微晃动
                    angle = np.sin(t * 3) * 0.08
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            elif role == 'neck1':
                if not is_running:
                    # 摇头阶段：辅助点头
                    angle = np.sin(t * 3) * 0.3
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
                else:
                    # 奔跑阶段：轻微晃动
                    angle = np.sin(t * 3) * 0.06
                    R = create_rotation_matrix(np.array([1, 0, 0]), angle)
                    T = create_transform_matrix(R=R)
            
            # ========== 头部 ==========
            # 左右摇头应该绕Z轴旋转
            elif role == 'head':
                if not is_running:
                    # 摇头阶段：绕Z轴左右大幅摇头
                    angle = np.sin(t * 2.5) * 0.4
                    R = create_rotation_matrix(np.array([0, 0, 1]), angle)
                    T = create_transform_matrix(R=R)
                else:
                    # 奔跑阶段：轻微转动
                    angle = np.sin(t * 2.5) * 0.04
                    R = create_rotation_matrix(np.array([0, 0, 1]), angle)
                    T = create_transform_matrix(R=R)
            
            local_transforms.append(T)
        
        animations.append(np.array(local_transforms))
    
    print(f"✅ 动画创建完成: {len(animations)} 帧")
    print(f"   - 前 {shake_frames} 帧: 摇头动画（头部和颈部运动）")
    print(f"   - 后 {run_frames} 帧: 奔跑动画（沿Y负方向前进）")
    return animations

def interactive_skeleton_viewer(mesh, skeleton, bones, weights_soft):
    """交互式骨架查看器"""
    print("🎮 启动交互式骨架查看器...")
    print("💡 使用鼠标拖拽旋转视角，滚轮缩放")
    print("💡 关闭窗口继续执行后续步骤")
    
    # 旋转矩阵：Z轴向上转为Y轴向上
    rotation_angle = np.pi/2
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
        [0, np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    
    # 应用旋转到网格顶点
    vertices_rotated = mesh.v @ rotation_matrix.T
    
    # 计算绑定姿态的关节位置
    joint_positions = skeleton.bind_positions()
    joint_positions_rotated = joint_positions @ rotation_matrix.T
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 绘制网格
    triangles = vertices_rotated[mesh.f]
    mesh_collection = Poly3DCollection(triangles, alpha=0.6, facecolors='lightblue', 
                                     edgecolors='navy', linewidths=0.3)
    ax.add_collection3d(mesh_collection)
    
    # 2. 绘制关节点
    ax.scatter(joint_positions_rotated[:, 0], 
               joint_positions_rotated[:, 1], 
               joint_positions_rotated[:, 2], 
               c='red', s=100, alpha=0.9, label='Joints', marker='o')
    
    # 3. 绘制骨骼连接线
    for bone_idx, (jp, jc) in enumerate(bones):
        parent_pos = joint_positions_rotated[jp]
        child_pos = joint_positions_rotated[jc]
        
        ax.plot([parent_pos[0], child_pos[0]], 
                [parent_pos[1], child_pos[1]], 
                [parent_pos[2], child_pos[2]], 
                color='darkred', linewidth=4, alpha=0.8)
        
        # 骨骼中点标记
        mid_point = (parent_pos + child_pos) / 2
        ax.scatter([mid_point[0]], [mid_point[1]], [mid_point[2]], 
                  c='orange', s=30, alpha=0.7)
    
    # 4. 添加关键关节标签（仅显示根节点和主要关节）
    labeled_joints = set()
    for idx in range(min(skeleton.n, 10)):  # 只标注前10个关节
        if idx not in labeled_joints:
            pos = joint_positions_rotated[idx]
            name = skeleton.joints[idx].name
            ax.text(pos[0], pos[1], pos[2], name, 
                   fontsize=8, alpha=0.7, color='darkred', weight='bold')
            labeled_joints.add(idx)
    
    # 5. 设置坐标轴和标签
    ax.set_xlabel('X (Forward/Back)', fontsize=12)
    ax.set_ylabel('Y (Up/Down)', fontsize=12)
    ax.set_zlabel('Z (Left/Right)', fontsize=12)
    ax.set_title('Interactive Skeleton Viewer - 拖拽鼠标旋转视角', fontsize=14, weight='bold')
    
    # 计算合适的显示范围
    all_points = np.vstack([vertices_rotated, joint_positions_rotated])
    center = np.mean(all_points, axis=0)
    max_range = np.max(np.abs(all_points - center)) * 1.3
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    # 设置初始视角
    ax.view_init(elev=15, azim=-45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # 添加统计信息文本
    info_text = f"""模型统计信息:
• 顶点数: {mesh.v.shape[0]}
• 面片数: {mesh.f.shape[0]}
• 关节数: {skeleton.n}
• 骨骼数: {len(bones)}

权重统计:
"""
    
    # 权重分布统计（显示前5个影响最大的骨骼）
    bone_influence = (weights_soft > 0.01).sum(axis=0)
    top_bones = sorted(enumerate(bone_influence), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (bone_idx, count) in enumerate(top_bones):
        jp, jc = bones[bone_idx]
        joint_p = skeleton.joints[jp].name
        joint_c = skeleton.joints[jc].name
        info_text += f"• {joint_p[:8]}->{joint_c[:8]}: {count}\n"
    
    # 在图形右侧添加信息文本
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure, 
             fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ 交互式查看器已关闭")

def save_obj(vertices, faces, filename):
    """保存OBJ文件"""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def render_frame_with_skeleton(vertices, faces, skeleton, G_current, bones, filename, frame_idx):
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
    if frame_idx <= 20:
        title = f'Frame {frame_idx:04d} - 摇头阶段'
    else:
        title = f'Frame {frame_idx:04d} - 奔跑阶段'
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

def main():
    print("🚀 开始骨架绑定演示程序 (GLB版本)")
    print("=" * 60)
    print("💡 配置信息:")
    print(f"   GLB 文件路径: data/cow/cow.glb")
    print(f"   OBJ 备用路径: data/cow/cow.obj")
    print("=" * 60)
    
    # 1. 加载模型和骨架
    print("\n📂 步骤1：加载3D模型和骨架")
    glb_path = "data/cow/cow.glb"
    obj_path = "data/cow/cow.obj"
    
    # 优先尝试从 GLB 加载
    use_glb = False
    mesh = None
    skeleton = None
    bones = []
    
    if os.path.exists(glb_path):
        try:
            print(f"   尝试从 GLB 加载网格: {glb_path}")
            vertices, faces = load_mesh_from_glb(glb_path, scale=1.0)
            
            # 修改这里：使用新的API
            mesh = Mesh()
            mesh.set_vertices_faces(vertices, faces)
            
            use_glb = True
            print(f"✅ 从 GLB 加载网格成功: {mesh.v.shape[0]} 顶点, {mesh.f.shape[0]} 面")
            
            # 同时加载骨架
            print(f"\n   尝试从 GLB 加载骨架...")
            skeleton, bones = load_skeleton_from_glb(
                glb_path,
                scale=1.0,  # 根据实际情况调整
                verbose=True
            )
            
            # 可视化骨架结构
            visualize_skeleton_structure(skeleton, bones)
            
            print(f"\n✅ 从 GLB 加载骨架成功: {skeleton.n} 个关节, {len(bones)} 段骨骼")
            
        except Exception as e:
            print(f"⚠️ GLB 加载失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"\n   回退到 OBJ 加载...")
            use_glb = False
    else:
        print(f"⚠️ GLB 文件不存在: {glb_path}")
        print(f"   使用 OBJ 和自动生成骨架...")
    
    # 如果 GLB 失败,使用 OBJ + 自动骨架
    if not use_glb:
        if not os.path.exists(obj_path):
            print(f"❌ 模型文件不存在: {obj_path}")
            return
        
        mesh = Mesh(obj_path)
        print(f"✅ 从 OBJ 加载模型成功: {mesh.v.shape[0]} 顶点, {mesh.f.shape[0]} 面")
        
        # 自动生成骨架
        print("\n🦴 使用自动生成的四足动物骨架")
        bbox_min = mesh.v.min(axis=0)
        bbox_max = mesh.v.max(axis=0)
        skeleton = quadruped_auto_place(bbox_min, bbox_max)
        
        # 定义骨骼连接
        bones = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # 躯干
            (2, 5), (5, 6), (6, 7),              # 左前腿
            (2, 8), (8, 9), (9, 10),             # 右前腿
            (1, 11), (11, 12), (12, 13),         # 左后腿
            (1, 14), (14, 15), (15, 16),         # 右后腿
        ]
        
        print(f"✅ 自动骨架创建成功: {skeleton.n} 个关节, {len(bones)} 段骨骼")
        
        # 打印骨架信息
        for i, joint in enumerate(skeleton.joints):
            parent_name = skeleton.joints[joint.parent].name if joint.parent >= 0 else "None"
            print(f"   关节{i}: {joint.name} (父节点: {parent_name}) 位置: {joint.pos}")
    
    # 打印模型信息
    print(f"\n📊 模型统计信息:")
    print(f"   顶点范围: X[{mesh.v[:,0].min():.2f}, {mesh.v[:,0].max():.2f}]")
    print(f"            Y[{mesh.v[:,1].min():.2f}, {mesh.v[:,1].max():.2f}]")
    print(f"            Z[{mesh.v[:,2].min():.2f}, {mesh.v[:,2].max():.2f}]")
    
    # 检查骨架与模型的位置关系
    print(f"\n🔍 检查骨架与模型的位置关系:")
    joint_positions = skeleton.bind_positions()
    print(f"   骨架中心: {joint_positions.mean(axis=0)}")
    print(f"   模型中心: {mesh.v.mean(axis=0)}")
    print(f"   骨架范围: X[{joint_positions[:,0].min():.2f}, {joint_positions[:,0].max():.2f}]")
    print(f"            Y[{joint_positions[:,1].min():.2f}, {joint_positions[:,1].max():.2f}]")
    print(f"            Z[{joint_positions[:,2].min():.2f}, {joint_positions[:,2].max():.2f}]")
    
    # 2. 验证骨骼连接关系
    print("\n🔗 步骤2：骨骼连接关系验证")
    print(f"   共有 {len(bones)} 段骨骼")
    print(f"   骨骼连接详情:")
    for i, (jp, jc) in enumerate(bones):
        joint_p = skeleton.joints[jp].name
        joint_c = skeleton.joints[jc].name
        print(f"   骨骼{i:2d}: {joint_p:20s} -> {joint_c:20s}")
    
    # 3. 计算权重
    print("\n⚖️ 步骤3：计算顶点权重")
    
    print("   使用最近骨骼权重方法...")
    weights_hard = hard_nearest_bone_weights(mesh.v, joint_positions, bones)
    print(f"✅ 硬权重计算完成，形状: {weights_hard.shape}")
    
    print("   使用双骨插值权重方法...")
    weights_soft = idw_two_bones(mesh.v, joint_positions, bones)
    print(f"✅ 软权重计算完成，形状: {weights_soft.shape}")
    
    # 验证权重
    weight_sums = weights_soft.sum(axis=1)
    print(f"   权重和检查: min={weight_sums.min():.3f}, max={weight_sums.max():.3f}")
    
    # 权重分布统计
    bone_influence = (weights_soft > 0.01).sum(axis=0)
    print(f"\n   权重分布统计（影响最大的前5个骨骼）:")
    top_bones = sorted(enumerate(bone_influence), key=lambda x: x[1], reverse=True)[:5]
    for i, (bone_idx, count) in enumerate(top_bones):
        jp, jc = bones[bone_idx]
        joint_p = skeleton.joints[jp].name
        joint_c = skeleton.joints[jc].name
        print(f"      {i+1}. {joint_p} -> {joint_c}: 影响 {count} 个顶点")
    
    # 4. 交互式预览
    print("\n👀 步骤4：交互式骨架预览")
    print("=" * 60)
    interactive_skeleton_viewer(mesh, skeleton, bones, weights_soft)
    
    # 询问用户是否继续生成动画
    print("\n❓ 是否继续生成动画？")
    print("   输入 'y' 或回车键继续生成动画")
    print("   输入其他任意键退出程序")
    user_input = input("请选择: ").strip().lower()
    
    if user_input not in ['', 'y', 'yes']:
        print("👋 程序退出")
        return
    
    # 5. 计算绑定姿态的变换矩阵
    print("\n🔧 步骤5：计算绑定姿态变换矩阵")
    bind_local_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(skeleton.n, axis=0)
    G_bind = skeleton.global_from_local(bind_local_transforms)
    G_bind_inv = np.linalg.inv(G_bind)
    print(f"✅ 绑定变换矩阵计算完成: {G_bind.shape}")
    
    # 6. 创建动画 - 修改为60帧
    print("\n🎬 步骤6：创建动画序列")
    num_frames = 60  # 修改为60帧
    animations = create_walking_animation(skeleton, num_frames)
    
    # 7. 渲染动画
    print("\n🎨 步骤7：渲染动画帧（包含骨架）")
    os.makedirs("out/frames", exist_ok=True)
    os.makedirs("out/debug", exist_ok=True)
    
    frame_files = []
    
    for frame_idx, local_transforms in enumerate(animations):
        print(f"   渲染第 {frame_idx+1}/{num_frames} 帧...")
        
        # 计算当前帧的全局变换
        G_current = skeleton.global_from_local(local_transforms)
        
        # 应用LBS变形
        deformed_vertices = apply_lbs(
            mesh.v, weights_soft, bones, G_current, G_bind_inv
        )
        
        # 保存变形后的OBJ（仅保存前10帧用于调试）
        if frame_idx < 10:
            debug_obj_path = f"out/debug/deformed_frame_{frame_idx+1:04d}.obj"
            save_obj(deformed_vertices, mesh.f, debug_obj_path)
            if frame_idx == 0:
                print(f"   调试文件已保存: {debug_obj_path}")
        
        # 使用渲染函数
        frame_path = f"out/frames/frame_{frame_idx+1:04d}.png"
        render_frame_with_skeleton(deformed_vertices, mesh.f, skeleton, G_current, 
                                 bones, frame_path, frame_idx+1)
        frame_files.append(frame_path)
    
    print("✅ 所有帧渲染完成")
    
    # 8. 生成动画视频/GIF
    print("\n📹 步骤8：生成动画")
    
    # 生成GIF
    gif_path = "out/rig_demo_shake_and_run.gif"
    try:
        import imageio.v2 as imageio_v2
        with imageio_v2.get_writer(gif_path, mode='I', duration=0.1) as writer:
            for frame_file in frame_files:
                image = imageio_v2.imread(frame_file)
                writer.append_data(image)
    except ImportError:
        # 回退到旧版本
        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)
    
    print(f"✅ GIF动画已保存: {gif_path}")
    
    # 尝试生成MP4
    try:
        mp4_path = "out/rig_demo_shake_and_run.mp4"
        cmd = (f"ffmpeg -y -framerate 10 -i out/frames/frame_%04d.png "
               f"-vf 'scale=1000:800' -c:v libx264 -pix_fmt yuv420p {mp4_path}")
        
        print(f"   执行命令: {cmd}")
        result = os.system(cmd)
        
        if result == 0 and os.path.exists(mp4_path):
            print(f"✅ MP4视频已保存: {mp4_path}")
        else:
            print("⚠️ MP4生成失败，尝试备用方法...")
            cmd_backup = (f"ffmpeg -y -framerate 10 -i out/frames/frame_%04d.png "
                         f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 "
                         f"-pix_fmt yuv420p -crf 23 {mp4_path}")
            result = os.system(cmd_backup)
            if result == 0 and os.path.exists(mp4_path):
                print(f"✅ MP4视频已保存: {mp4_path}")
            else:
                print("❌ MP4生成失败，请检查ffmpeg安装")
    except Exception as e:
        print(f"❌ MP4生成过程中出错: {e}")
    
    # 9. 输出统计信息
    print("\n📊 步骤9：最终统计信息")
    print("=" * 60)
    print("🎯 骨架绑定演示完成!")
    print(f"📁 输出目录: out/")
    print(f"🖼️ 动画帧数: {num_frames} (前20帧摇头 + 后40帧奔跑)")
    print(f"🦴 骨架来源: {'GLB 文件' if use_glb else '自动生成'}")
    print(f"🦴 骨架关节: {skeleton.n} 个")
    print(f"🔗 骨骼段数: {len(bones)} 段")
    print(f"📐 网格顶点: {mesh.v.shape[0]} 个")
    print(f"📐 网格面片: {mesh.f.shape[0]} 个")
    
    print("=" * 60)
    print("🎉 程序执行完成！")
    print("📁 请查看 out/ 目录下的输出文件")
    print("🎬 动画文件: out/rig_demo_shake_and_run.gif")
    if os.path.exists("out/rig_demo_shake_and_run.mp4"):
        print("🎬 视频文件: out/rig_demo_shake_and_run.mp4")
    print("=" * 60)

if __name__ == "__main__":
    main()