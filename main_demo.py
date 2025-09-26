# main_demo.py
import os, subprocess
import numpy as np
import trimesh

from rigging.mesh_io import Mesh
from rigging.skeleton import quadruped_auto_place, Skeleton
from rigging.weights_nearest import idw_two_bones
from rigging.lbs import apply_lbs

# ---------- 工具函数 ----------
def rot_axis(axis, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4, dtype=np.float32)
    if axis == 'x':
        R[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
    elif axis == 'y':
        R[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
    elif axis == 'z':
        R[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
    return R

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------- 1) 读取模型 ----------
mesh_path = "data/cow/cow.obj"   # 你可换成 data/horse/horse.obj
M = Mesh(mesh_path)
V0, F = M.v, M.f
print("[mesh] V=", V0.shape, "F=", F.shape)

# ---------- 2) 放置骨架 ----------
Sk = quadruped_auto_place(V0.min(0), V0.max(0))
Jpos = Sk.bind_positions()
parents = Sk.parents()
bones = [(p_i, j) for j, p_i in enumerate(parents) if p_i >= 0]
print("[skel] J=", Sk.n, "bones=", len(bones))

# ---------- 3) 权重（先用双骨IDW） ----------
W = idw_two_bones(V0, Jpos, bones)
row_sum = W.sum(1)
print("[weights] W=", W.shape, "row-sum min/max:", row_sum.min(), row_sum.max())
assert (W > 0).any(), "all-zero weights!"
assert np.allclose(row_sum, 1.0, atol=1e-4), "weights rows must sum to 1"

# ---------- 4) 绑定姿态的全局矩阵与其逆 ----------
J = Sk.n
local_bind = np.repeat(np.eye(4, dtype=np.float32)[None, ...], J, axis=0)
G_bind = Sk.global_from_local(local_bind)
G_bind_inv = np.linalg.inv(G_bind)

# ---------- 5) 动画（让头部和左肘摆动），逐帧 LBS + 简易渲染 ----------
name2idx = {j.name: i for i, j in enumerate(Sk.joints)}
L_elbow = name2idx.get("L_elbow", None)
head    = name2idx.get("head", None)

frames = 48
ensure_dir("out/frames")
ensure_dir("out/debug")

for t in range(frames):
    theta = 0.6 * np.sin(2*np.pi * t/frames)  # -0.6~0.6 rad
    local_T = local_bind.copy()

    if L_elbow is not None:
        local_T[L_elbow] = rot_axis('y', theta)  # 肘绕 y 摆
    if head is not None:
        local_T[head] = rot_axis('z', -0.5*theta)  # 头绕 z 点头/摆头

    G_cur = Sk.global_from_local(local_T)
    V1 = apply_lbs(V0, W, bones, G_cur, G_bind_inv)

    # 第 1 帧存个变形后的 OBJ 方便肉眼对比
    if t == 0:
        trimesh.Trimesh(V1, F, process=False).export("out/debug/deformed_frame_0001.obj")
        print("[debug] wrote out/debug/deformed_frame_0001.obj")

    # 用 trimesh 的简易渲染快速出 PNG（避免 GL 配置问题）
    scene = trimesh.Scene(trimesh.Trimesh(V1, F, process=False))
    png = scene.save_image(resolution=(900, 700), visible=True)
    with open(f"out/frames/frame_{t+1:04d}.png", "wb") as fp:
        fp.write(png)

print(f"[render] wrote {frames} frames to out/frames")

# ---------- 6) 合成视频（优先 mp4；无 ffmpeg 则写 gif） ----------
mp4_path = "out/rig_demo.mp4"
gif_path = "out/rig_demo.gif"
try:
    # 如果系统有 ffmpeg
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "24",
        "-i", "out/frames/frame_%04d.png",
        "-pix_fmt", "yuv420p", mp4_path
    ], check=True)
    print(f"[video] {mp4_path}")
except Exception as e:
    print("[video] ffmpeg not available, writing GIF instead...", e)
    import imageio.v2 as imageio
    imgs = [imageio.imread(f"out/frames/frame_{i+1:04d}.png") for i in range(frames)]
    imageio.mimsave(gif_path, imgs, fps=12)
    print(f"[video] {gif_path}")
