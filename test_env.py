#!/usr/bin/env python3
"""
诊断脚本：检查 pyrender 导入问题
"""

print("=== pyrender 诊断脚本 ===")

# 检查基础导入
print("\n1. 检查基础模块导入...")
try:
    import numpy as np
    print("✓ numpy 导入成功")
except ImportError as e:
    print(f"✗ numpy 导入失败: {e}")

try:
    import trimesh
    print("✓ trimesh 导入成功")
except ImportError as e:
    print(f"✗ trimesh 导入失败: {e}")

try:
    import PIL.Image
    print("✓ PIL.Image 导入成功")
except ImportError as e:
    print(f"✗ PIL.Image 导入失败: {e}")

# 检查 OpenGL
print("\n2. 检查 OpenGL...")
try:
    import OpenGL
    print(f"✓ OpenGL 导入成功, 版本: {OpenGL.__version__}")
except ImportError as e:
    print(f"✗ OpenGL 导入失败: {e}")

try:
    import OpenGL.GL as gl
    print("✓ OpenGL.GL 导入成功")
except ImportError as e:
    print(f"✗ OpenGL.GL 导入失败: {e}")

# 检查 pyglet
print("\n3. 检查 pyglet...")
try:
    import pyglet
    print(f"✓ pyglet 导入成功, 版本: {pyglet.version}")
except ImportError as e:
    print(f"✗ pyglet 导入失败: {e}")

# 尝试导入 pyrender 并获取详细错误信息
print("\n4. 检查 pyrender 导入...")
try:
    import pyrender
    print(f"✓ pyrender 导入成功, 版本: {pyrender.__version__}")
except ImportError as e:
    print(f"✗ pyrender 导入失败: {e}")
    print("详细错误信息:")
    import traceback
    traceback.print_exc()

# 检查环境变量
print("\n5. 检查环境变量...")
import os
opengl_platform = os.environ.get('PYOPENGL_PLATFORM', 'None')
print(f"PYOPENGL_PLATFORM: {opengl_platform}")

# 如果 pyrender 导入成功，尝试创建渲染器
print("\n6. 尝试创建离线渲染器...")
try:
    import pyrender
    r = pyrender.OffscreenRenderer(640, 480)
    print("✓ OffscreenRenderer 创建成功")
    r.delete()
    print("✓ OffscreenRenderer 删除成功")
except Exception as e:
    print(f"✗ OffscreenRenderer 创建失败: {e}")
    import traceback
    traceback.print_exc()