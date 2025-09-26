# animation-skeleton
├─ data/ # 放模型（已下载）
├─ out/ # 渲染帧与视频输出
├─ rigging/ # 算法核心（你要自己实现的部分）
│ ├─ __init__.py
│ ├─ mesh_io.py # OBJ 读写、预处理
│ ├─ skeleton.py # 关节/骨骼数据结构 + FK
│ ├─ weights_nearest.py # 最近骨骼/双骨插值权重（基础必做）
│ ├─ weights_heat.py # 热扩散/拉普拉斯权重（进阶）
│ └─ lbs.py # 线性混合蒙皮 (Linear Blend Skinning)
├─ render/
│ ├─ offscreen_mgl.py # 用 moderngl 的离屏渲染为 PNG
│ └─ make_video.sh # 调 ffmpeg 合成 mp4
├─ tools/
│ └─ preview_obj.py # 预览网格，检查拓扑
├─ main_demo.py # 一键：加载 → 放置骨架 → 算权重 → 播放动作 → 渲染
├─ requirements.txt
└─ README.md