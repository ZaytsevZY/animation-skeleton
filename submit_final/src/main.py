#!/usr/bin/env python3
"""
骨架绑定系统 - 主程序
功能：启动带有完整骨架绑定功能的UI界面

运行环境要求：
- Python 3.7+
- PyQt5
- PyVista
- NumPy
- 其他依赖详见requirements.txt

使用方法：
1. 直接运行此文件启动程序
2. 使用UI界面进行骨架绑定操作
3. 支持导出骨架信息、绑定权重等数据

作者：计算机动画课程大作业
日期：2025年
"""

import sys
import os
import warnings

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 忽略警告
warnings.filterwarnings('ignore')

def check_dependencies():
    """检查依赖项"""
    missing_deps = []

    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")

    try:
        import pyvista
    except ImportError:
        missing_deps.append("pyvista")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import vtk
    except ImportError:
        missing_deps.append("vtk")

    if missing_deps:
        print("❌ 缺少依赖项:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n请使用以下命令安装缺失的依赖项:")
        for dep in missing_deps:
            if dep == "PyQt5":
                print("   pip install PyQt5")
            elif dep == "pyvista":
                print("   pip install pyvista")
            elif dep == "numpy":
                print("   pip install numpy")
            elif dep == "vtk":
                print("   pip install vtk")
        return False

    return True

def main():
    """主函数 - 启动骨架绑定系统"""
    print("[INFO] 启动骨架绑定系统...")

    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)

    try:
        # 导入UI模块
        from ui_simple import main as run_ui

        print("[INFO] 所有依赖项已满足")
        print("[INFO] 输出目录:", os.path.join(os.path.dirname(__file__), 'output'))
        print("[INFO] 正在启动UI界面...")

        # 启动主界面
        run_ui()

    except Exception as e:
        print(f"[ERROR] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()