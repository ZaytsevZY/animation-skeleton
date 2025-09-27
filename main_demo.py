# main_demo.py (交互式版本)
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

# 导入自定义模块
from rigging.mesh_io import Mesh
from rigging.skeleton import quadruped_auto_place, Skeleton
from rigging.weights_nearest import hard_nearest_bone_weights, idw_two_bones
from rigging.lbs import apply_lbs

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
    fig = plt.figure(figsize=(12, 10))
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
    
    # 4. 添加关键关节标签
    key_joints_indices = {
        'root': 0, 'spine1': 1, 'spine2': 2, 'neck': 3, 'head': 4,
        'L_shoulder': 5, 'R_shoulder': 8, 'L_hip': 11, 'R_hip': 14
    }
    
    for name, idx in key_joints_indices.items():
        if idx < len(joint_positions_rotated):
            pos = joint_positions_rotated[idx]
            ax.text(pos[0], pos[1], pos[2], name, 
                   fontsize=10, alpha=0.8, color='darkred', weight='bold')
    
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
        info_text += f"• {joint_p}->{joint_c}: {count}个顶点\n"
    
    # 在图形右侧添加信息文本
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure, 
             fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ 交互式查看器已关闭")

def create_walking_animation(skeleton, num_frames=60):
    """创建简单的行走动画 - Y轴向上的坐标系"""
    print("🎬 创建行走动画...")
    
    animations = []
    
    for frame in range(num_frames):
        t = frame / num_frames * 2 * np.pi  # 一个完整周期
        
        # 创建局部变换矩阵
        local_transforms = []
        
        for i, joint in enumerate(skeleton.joints):
            if joint.name == "root":
                # 根节点前进运动 (沿X轴前进)
                forward = np.array([np.sin(t * 2) * 0.1, 0, 0])
                T = create_transform_matrix(t=forward)
            
            elif joint.name in ["L_hip", "R_hip"]:
                # 髋关节摆动 - 绕Z轴旋转（前后摆动，Y轴向上时）
                phase = 0 if "L_" in joint.name else np.pi
                angle = np.sin(t + phase) * 0.3
                R = create_rotation_matrix(np.array([0, 0, 1]), angle)
                T = create_transform_matrix(R=R)
            
            elif joint.name in ["L_knee", "R_knee"]:
                # 膝关节弯曲 - 绕Z轴旋转
                phase = 0 if "L_" in joint.name else np.pi
                angle = -np.abs(np.sin(t + phase)) * 0.5
                R = create_rotation_matrix(np.array([0, 0, 1]), angle)
                T = create_transform_matrix(R=R)
            
            elif joint.name in ["L_shoulder", "R_shoulder"]:
                # 肩关节摆动 - 绕Z轴旋转（前腿与后腿相位相反）
                phase = np.pi if "L_" in joint.name else 0
                angle = np.sin(t + phase) * 0.2
                R = create_rotation_matrix(np.array([0, 0, 1]), angle)
                T = create_transform_matrix(R=R)
            
            elif joint.name in ["L_elbow", "R_elbow"]:
                # 肘关节弯曲 - 绕Z轴旋转
                phase = np.pi if "L_" in joint.name else 0
                angle = -np.abs(np.sin(t + phase)) * 0.3
                R = create_rotation_matrix(np.array([0, 0, 1]), angle)
                T = create_transform_matrix(R=R)
            
            elif joint.name == "spine2":
                # 脊椎轻微摆动 - 绕Y轴旋转（左右摆动）
                angle = np.sin(t * 2) * 0.1
                R = create_rotation_matrix(np.array([0, 1, 0]), angle)
                T = create_transform_matrix(R=R)
            
            elif joint.name == "neck":
                # 颈部点头 - 绕Z轴旋转
                angle = np.sin(t * 3) * 0.15
                R = create_rotation_matrix(np.array([0, 0, 1]), angle)
                T = create_transform_matrix(R=R)
            
            else:
                # 其他关节保持不动
                T = np.eye(4, dtype=np.float32)
            
            local_transforms.append(T)
        
        animations.append(np.array(local_transforms))
    
    return animations

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
    ax.set_title(f'Frame {frame_idx:04d} - Cow Walking Animation with Skeleton', fontsize=14)
    
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
    print("🚀 开始骨架绑定演示程序")
    print("=" * 50)
    
    # 1. 加载模型
    print("📂 步骤1：加载3D模型")
    model_path = "data/cow/cow.obj"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    mesh = Mesh(model_path)
    print(f"✅ 模型加载成功: {mesh.v.shape[0]} 顶点, {mesh.f.shape[0]} 面")
    print(f"   顶点范围: X[{mesh.v[:,0].min():.2f}, {mesh.v[:,0].max():.2f}]")
    print(f"            Y[{mesh.v[:,1].min():.2f}, {mesh.v[:,1].max():.2f}]")
    print(f"            Z[{mesh.v[:,2].min():.2f}, {mesh.v[:,2].max():.2f}]")
    
    # 2. 创建骨架
    print("\n🦴 步骤2：创建骨架结构")
    bbox_min = mesh.v.min(axis=0)
    bbox_max = mesh.v.max(axis=0)
    skeleton = quadruped_auto_place(bbox_min, bbox_max)
    
    print(f"✅ 骨架创建成功: {skeleton.n} 个关节")
    for i, joint in enumerate(skeleton.joints):
        parent_name = skeleton.joints[joint.parent].name if joint.parent >= 0 else "None"
        print(f"   关节{i}: {joint.name} (父节点: {parent_name}) 位置: {joint.pos}")
    
    # 3. 定义骨骼连接
    print("\n🔗 步骤3：定义骨骼连接关系")
    bones = [
        (0, 1),   # root -> spine1
        (1, 2),   # spine1 -> spine2  
        (2, 3),   # spine2 -> neck
        (3, 4),   # neck -> head
        (2, 5),   # spine2 -> L_shoulder
        (5, 6),   # L_shoulder -> L_elbow
        (6, 7),   # L_elbow -> L_wrist
        (2, 8),   # spine2 -> R_shoulder
        (8, 9),   # R_shoulder -> R_elbow
        (9, 10),  # R_elbow -> R_wrist
        (1, 11),  # spine1 -> L_hip
        (11, 12), # L_hip -> L_knee
        (12, 13), # L_knee -> L_ankle
        (1, 14),  # spine1 -> R_hip
        (14, 15), # R_hip -> R_knee
        (15, 16), # R_knee -> R_ankle
    ]
    
    print(f"✅ 定义了 {len(bones)} 段骨骼")
    for i, (jp, jc) in enumerate(bones):
        joint_p = skeleton.joints[jp].name
        joint_c = skeleton.joints[jc].name
        print(f"   骨骼{i}: {joint_p} -> {joint_c}")
    
    # 4. 计算权重
    print("\n⚖️ 步骤4：计算顶点权重")
    joint_positions = skeleton.bind_positions()
    
    print("   使用最近骨骼权重方法...")
    weights_hard = hard_nearest_bone_weights(mesh.v, joint_positions, bones)
    print(f"✅ 硬权重计算完成，形状: {weights_hard.shape}")
    
    print("   使用双骨插值权重方法...")
    weights_soft = idw_two_bones(mesh.v, joint_positions, bones)
    print(f"✅ 软权重计算完成，形状: {weights_soft.shape}")
    
    # 验证权重
    weight_sums = weights_soft.sum(axis=1)
    print(f"   权重和检查: min={weight_sums.min():.3f}, max={weight_sums.max():.3f}")
    
    # 5. 交互式预览
    print("\n👀 步骤5：交互式骨架预览")
    print("=" * 30)
    interactive_skeleton_viewer(mesh, skeleton, bones, weights_soft)
    
    # 询问用户是否继续生成动画
    print("\n❓ 是否继续生成动画？")
    print("   输入 'y' 或回车键继续生成动画")
    print("   输入其他任意键退出程序")
    user_input = input("请选择: ").strip().lower()
    
    if user_input not in ['', 'y', 'yes']:
        print("👋 程序退出")
        return
    
    # 6. 计算绑定姿态的变换矩阵
    print("\n🔧 步骤6：计算绑定姿态变换矩阵")
    bind_local_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(skeleton.n, axis=0)
    G_bind = skeleton.global_from_local(bind_local_transforms)
    G_bind_inv = np.linalg.inv(G_bind)
    print(f"✅ 绑定变换矩阵计算完成: {G_bind.shape}")
    
    # 7. 创建动画
    print("\n🎬 步骤7：创建动画序列")
    num_frames = 30
    animations = create_walking_animation(skeleton, num_frames)
    print(f"✅ 动画创建完成: {len(animations)} 帧")
    
    # 8. 渲染动画
    print("\n🎨 步骤8：渲染动画帧（包含骨架）")
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
        
        # 保存变形后的OBJ（仅保存第一帧用于调试）
        if frame_idx == 0:
            debug_obj_path = f"out/debug/deformed_frame_{frame_idx+1:04d}.obj"
            save_obj(deformed_vertices, mesh.f, debug_obj_path)
            print(f"   调试文件已保存: {debug_obj_path}")
        
        # 使用渲染函数
        frame_path = f"out/frames/frame_{frame_idx+1:04d}.png"
        render_frame_with_skeleton(deformed_vertices, mesh.f, skeleton, G_current, 
                                 bones, frame_path, frame_idx+1)
        frame_files.append(frame_path)
    
    print("✅ 所有帧渲染完成")
    
    # 9. 生成动画视频/GIF
    print("\n📹 步骤9：生成动画")
    
    # 生成GIF
    gif_path = "out/rig_demo_with_skeleton.gif"
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
        mp4_path = "out/rig_demo_with_skeleton.mp4"
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
    
    # 10. 输出统计信息
    print("\n📊 步骤10：输出统计信息")
    print("=" * 50)
    print("🎯 骨架绑定演示完成!")
    print(f"📁 输出目录: out/")
    print(f"🖼️ 动画帧数: {num_frames}")
    print(f"🦴 骨架关节: {skeleton.n} 个")
    print(f"🔗 骨骼段数: {len(bones)} 段")
    print(f"📐 网格顶点: {mesh.v.shape[0]} 个")
    print(f"📐 网格面片: {mesh.f.shape[0]} 个")
    
    # 权重分布统计
    bone_influence = (weights_soft > 0.01).sum(axis=0)
    print(f"📊 权重分布统计:")
    for i, count in enumerate(bone_influence):
        joint_p = skeleton.joints[bones[i][0]].name
        joint_c = skeleton.joints[bones[i][1]].name
        print(f"   骨骼 {joint_p}->{joint_c}: 影响 {count} 个顶点")
    
    print("=" * 50)
    print("🎉 程序执行完成！")
    print("📁 请查看 out/ 目录下的输出文件")
    print("🎬 动画文件: out/rig_demo_with_skeleton.gif")
    if os.path.exists("out/rig_demo_with_skeleton.mp4"):
        print("🎬 视频文件: out/rig_demo_with_skeleton.mp4")

if __name__ == "__main__":
    main()