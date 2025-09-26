# main_demo.py (修改版本)
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

def create_walking_animation(skeleton, num_frames=60):
    """创建简单的行走动画 - 修正Y轴向上的坐标系"""
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
                # 髋关节摆动 - 绕X轴旋转（前后摆动）
                phase = 0 if "L_" in joint.name else np.pi
                angle = np.sin(t + phase) * 0.3
                R = create_rotation_matrix(np.array([1, 0, 0]), angle)  # 改为绕X轴
                T = create_transform_matrix(R=R)
            
            elif joint.name in ["L_knee", "R_knee"]:
                # 膝关节弯曲 - 绕X轴旋转
                phase = 0 if "L_" in joint.name else np.pi
                angle = -np.abs(np.sin(t + phase)) * 0.5
                R = create_rotation_matrix(np.array([1, 0, 0]), angle)  # 改为绕X轴
                T = create_transform_matrix(R=R)
            
            elif joint.name in ["L_shoulder", "R_shoulder"]:
                # 肩关节摆动 - 绕X轴旋转（前腿与后腿相位相反）
                phase = np.pi if "L_" in joint.name else 0
                angle = np.sin(t + phase) * 0.2
                R = create_rotation_matrix(np.array([1, 0, 0]), angle)  # 改为绕X轴
                T = create_transform_matrix(R=R)
            
            elif joint.name in ["L_elbow", "R_elbow"]:
                # 肘关节弯曲 - 绕X轴旋转
                phase = np.pi if "L_" in joint.name else 0
                angle = -np.abs(np.sin(t + phase)) * 0.3
                R = create_rotation_matrix(np.array([1, 0, 0]), angle)  # 改为绕X轴
                T = create_transform_matrix(R=R)
            
            elif joint.name == "spine2":
                # 脊椎轻微摆动 - 绕Z轴旋转（左右摆动）
                angle = np.sin(t * 2) * 0.1
                R = create_rotation_matrix(np.array([0, 0, 1]), angle)  # 改为绕Z轴
                T = create_transform_matrix(R=R)
            
            elif joint.name == "neck":
                # 颈部点头 - 绕X轴旋转
                angle = np.sin(t * 3) * 0.15
                R = create_rotation_matrix(np.array([1, 0, 0]), angle)  # 改为绕X轴
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

def render_frame_simple(vertices, faces, filename, frame_idx):
    """改进的渲染函数，固定分辨率和相机视角"""
    # 设置固定的偶数分辨率
    fig_width, fig_height = 8, 8  # 英寸
    dpi = 100
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制网格 - 使用三角形面片
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # 创建三角形集合
    triangles = vertices[faces]
    
    # 添加面片集合
    mesh = Poly3DCollection(triangles, alpha=0.7, facecolors='lightblue', 
                           edgecolors='navy', linewidths=0.5)
    ax.add_collection3d(mesh)
    
    # 设置坐标轴标签 - 明确Y轴向上
    ax.set_xlabel('X (Forward/Backward)')
    ax.set_ylabel('Y (Up/Down)')
    ax.set_zlabel('Z (Left/Right)')
    ax.set_title(f'Frame {frame_idx:04d} - Cow Walking Animation', fontsize=14)
    
    # 计算合适的范围
    center = np.mean(vertices, axis=0)
    max_range = np.max(np.abs(vertices - center)) * 1.2
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    # 固定视角 - 不再旋转，从侧面观看动画
    # elev: 仰角（正值向上看）
    # azim: 方位角（0度是从前面看，90度从右侧看）
    ax.view_init(elev=10, azim=0)  # 从正面稍微向上的角度观看
    
    # 保持坐标轴显示，这样可以清楚看到Y轴向上
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='z', labelsize=8)
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    
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
    
    # 5. 计算绑定姿态的变换矩阵
    print("\n🔧 步骤5：计算绑定姿态变换矩阵")
    bind_local_transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(skeleton.n, axis=0)
    G_bind = skeleton.global_from_local(bind_local_transforms)
    G_bind_inv = np.linalg.inv(G_bind)
    print(f"✅ 绑定变换矩阵计算完成: {G_bind.shape}")
    
    # 6. 创建动画
    print("\n🎬 步骤6：创建动画序列")
    num_frames = 30
    animations = create_walking_animation(skeleton, num_frames)
    print(f"✅ 动画创建完成: {len(animations)} 帧")
    
    # 7. 渲染动画
    print("\n🎨 步骤7：渲染动画帧")
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
        
        # 渲染帧
        frame_path = f"out/frames/frame_{frame_idx+1:04d}.png"
        render_frame_simple(deformed_vertices, mesh.f, frame_path, frame_idx+1)
        frame_files.append(frame_path)
    
    print("✅ 所有帧渲染完成")
    
    # 8. 生成动画视频/GIF
    print("\n📹 步骤8：生成动画")
    
    # 生成GIF
    gif_path = "out/rig_demo.gif"
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
    
    # 尝试生成MP4（修复分辨率问题）
    try:
        mp4_path = "out/rig_demo.mp4"
        # 使用scale滤镜确保分辨率为偶数
        cmd = (f"ffmpeg -y -framerate 10 -i out/frames/frame_%04d.png "
               f"-vf 'scale=800:800' -c:v libx264 -pix_fmt yuv420p {mp4_path}")
        
        print(f"   执行命令: {cmd}")
        result = os.system(cmd)
        
        if result == 0 and os.path.exists(mp4_path):
            print(f"✅ MP4视频已保存: {mp4_path}")
        else:
            print("⚠️ MP4生成失败，尝试备用方法...")
            # 备用方法：使用更简单的参数
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
    print("\n📊 步骤9：输出统计信息")
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
    print("🎬 动画文件: out/rig_demo.gif")
    if os.path.exists("out/rig_demo.mp4"):
        print("🎬 视频文件: out/rig_demo.mp4")

if __name__ == "__main__":
    main()