@echo off
echo 🚀 启动骨架绑定系统...

REM 检查Python是否安装
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 未检测到Python，请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

REM 检查依赖并尝试安装
python -c "import PyQt5, pyvista, numpy, trimesh, vtk" > nul 2>&1
if errorlevel 1 (
    echo 📦 正在安装依赖...
    pip install -r ../src/requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败，请手动安装依赖
        pause
        exit /b 1
    )
)

REM 运行主程序
echo ✅ 正在启动程序...
cd ../src
python main.py
cd ../bin
pause