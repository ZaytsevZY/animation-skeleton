import numpy as np
import trimesh
from trimesh import creation as tc
import pyrender
import PIL.Image

def test_simple_render():
    """测试最简单的渲染设置"""
    print("=== 测试简单渲染 ===")
    
    # 1. 创建一个简单的球体在原点
    sphere = tc.icosphere(subdivisions=2, radius=0.5)
    sphere_center = sphere.vertices.mean(axis=0)
    print(f"球体中心: {sphere_center}")
    print(f"球体边界: {sphere.vertices.min(axis=0)} 到 {sphere.vertices.max(axis=0)}")
    
    # 2. 创建场景
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])  # 黑色背景
    
    # 3. 添加球体 - 白色材质
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # 白色
        metallicFactor=0.0,
        roughnessFactor=0.5
    )
    mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
    scene.add(mesh)
    
    # 4. 设置相机 - 从 Z 正方向看向原点
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    
    # 简单的相机姿态：在 Z=2 位置，看向原点
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],  # 相机在 Z=2
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    scene.add(camera, pose=camera_pose)
    print(f"相机位置: [0, 0, 2]")
    
    # 5. 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=camera_pose)
    
    # 6. 渲染
    r = pyrender.OffscreenRenderer(400, 400)
    try:
        color, depth = r.render(scene)
        
        print(f"渲染结果:")
        print(f"  颜色范围: {color.min()} - {color.max()}")
        print(f"  深度范围: {depth.min()} - {depth.max()}")
        
        # 检查非黑色像素
        non_black = (color > 0).any(axis=2).sum()
        print(f"  非黑色像素数量: {non_black}")
        
        # 保存测试图像
        PIL.Image.fromarray(color).save("test_render.png")
        print("保存测试图像到 test_render.png")
        
        return depth.max() > 0  # 如果有深度信息说明渲染成功
        
    finally:
        r.delete()

def test_cow_render():
    """测试牛模型的基本渲染"""
    print("\n=== 测试牛模型渲染 ===")
    
    from rigging.mesh_io import Mesh
    
    # 1. 加载牛模型
    M = Mesh("data/cow/cow.obj")
    V, F = M.v, M.f
    
    cow_mesh = trimesh.Trimesh(V, F, process=False)
    center = cow_mesh.bounds.mean(axis=0)
    size = np.linalg.norm(cow_mesh.bounds[1] - cow_mesh.bounds[0])
    
    print(f"牛模型中心: {center}")
    print(f"牛模型尺寸: {size}")
    print(f"牛模型边界: {cow_mesh.bounds[0]} 到 {cow_mesh.bounds[1]}")
    
    # 2. 创建场景
    scene = pyrender.Scene(bg_color=[0.2, 0.2, 0.2, 1.0])  # 深灰背景
    
    # 3. 添加牛模型 - 红色材质
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.3, 0.3, 1.0],  # 红色
        metallicFactor=0.0,
        roughnessFactor=0.7
    )
    mesh = pyrender.Mesh.from_trimesh(cow_mesh, material=material)
    scene.add(mesh)
    
    # 4. 设置相机 - 从侧面看
    eye = center + np.array([size * 2.0, size * 0.2, 0.0])
    target = center
    
    print(f"相机位置: {eye}")
    print(f"目标位置: {target}")
    
    # 构建 look-at 矩阵 (右手坐标系，-Z 为前方)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_actual = np.cross(right, forward)
    
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[0, :3] = right
    camera_pose[1, :3] = up_actual
    camera_pose[2, :3] = -forward  # pyrender 使用 -Z 作为前方
    camera_pose[:3, 3] = eye
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)
    
    # 5. 添加强光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=8.0)
    scene.add(light, pose=camera_pose)
    
    # 6. 渲染
    r = pyrender.OffscreenRenderer(800, 600)
    try:
        color, depth = r.render(scene)
        
        print(f"渲染结果:")
        print(f"  颜色范围: {color.min()} - {color.max()}")
        print(f"  深度范围: {depth.min()} - {depth.max()}")
        
        # 检查非背景像素 (深灰 = 51)
        bg_pixels = (color == 51).all(axis=2).sum()
        total_pixels = color.shape[0] * color.shape[1]
        non_bg_pixels = total_pixels - bg_pixels
        print(f"  非背景像素数量: {non_bg_pixels} / {total_pixels}")
        
        # 保存测试图像
        PIL.Image.fromarray(color).save("test_cow_render.png")
        print("保存牛模型测试图像到 test_cow_render.png")
        
        return depth.max() > 0
        
    finally:
        r.delete()

if __name__ == "__main__":
    # 先测试简单球体
    success1 = test_simple_render()
    print(f"简单球体渲染{'成功' if success1 else '失败'}")
    
    # 再测试牛模型
    success2 = test_cow_render()
    print(f"牛模型渲染{'成功' if success2 else '失败'}")
    
    if not success1 and not success2:
        print("\n两个测试都失败了，pyrender 可能有配置问题")
        print("尝试检查 OpenGL 驱动或使用 matplotlib 后备方案")
    elif success1 and not success2:
        print("\n简单渲染成功但牛模型失败，可能是坐标系或相机位置问题")
    else:
        print("\n渲染测试通过！问题可能在动画循环中")