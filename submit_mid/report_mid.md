# 骨架绑定-技术文档（中期）

查益 2022012107 chay22@mails.tsinghua.edu.cn

## 一、项目概述

本项目计划实现一个基于骨架的三维模型动画系统，支持骨架绑定、蒙皮权重计算和实时动画预览。

## 二、运行环境

### 软件依赖

- python 3.8+
- PyQt5
- pyvista
- numpy
- Vtk

### 运行方式

目前功能相对耦合，支持一键测试样例：

```python
python main_demo.py
```

可以导入cow模型，并渲染生成相应的动画视频。

```python
python ui_simple.py
```

可以导入cow模型，打开交互界面，支持移动视角、缩放、移动关节并使蒙皮跟随。

## 三、运行效果

`main_demo.py`可以加载模型和骨架，计算权重，生成“牛模型先摇头后奔跑”的视频和gif图。

![image-20251112163658123](/Users/zhayi/University/大四上/计算机动画的算法与技术/animation-skeleton/report_mid.assets/image-20251112163658123.png)

`ui_simple.py`可以生成一个交互界面。交互界面中可以拖拽编辑模型的骨架。图示内容是加载模型后的ui界面，以及拖拽了腿部和头部的模型。

![image-20251112163742326](/Users/zhayi/University/大四上/计算机动画的算法与技术/animation-skeleton/report_mid.assets/image-20251112163742326.png)

![image-20251112163838080](/Users/zhayi/University/大四上/计算机动画的算法与技术/animation-skeleton/report_mid.assets/image-20251112163838080.png)

## 四、功能实现

### 蒙皮获取

参考https://github.com/alecjacobson/common-3d-test-models

### 骨架设计

针对牛的体型特征，在关键的脊柱、头部、四肢处设置骨架。

在关键且符合生理学特征的位置设置关节点，大体设计按照脊椎动物的头-躯干-四肢进行设计。

因动画模拟能力有限，关节点设置数量比正常动物的大型关节稍多，使得动作能够相对生动。

### 蒙皮权重计算

使用IDW方法，基于逆距离加权计算蒙皮权重。
$$
w_{ij} = \frac{1/d_{ij}^p}{\sum_{k=1}^M 1/d_{ik}^p}
$$
其中w为顶点到骨骼的权重；d为顶点到骨骼的距离；p为幂指数，可以控制衰减速度。分母为归一化因子，保证最终权重和为1。

当距离变大时，权重以幂指数速度衰减。这也符合正常运动的规律：蒙皮的控制大部分取决于临近骨骼，而远处骨骼移动时，则会轻微牵动蒙皮移动。

### LBS变形

线性混合蒙皮变形
$$
\mathbf{v}' = \sum_{j=1}^M w_j \mathbf{T}_j \mathbf{T}_{\text{bind},j}^{-1} \mathbf{v}_{\text{bind}}
$$
其中v为顶点在初始姿态下的位置，v'为顶点在变形之后的位置，w为顶点对骨骼的权重，T为变换矩阵，M为影响该顶点的骨骼总数。

变形过程中，首先变换到骨骼j的局部坐标系，应用变换，并回到世界坐标系，最终加权混合所有变换结果。

### 交互界面

使用PYQt5库制作ui界面，VTK库实现3D交互，PyVista实现3D可视化与Qt集成，以及实时更新渲染。

提供相机控制，支持旋转，缩放，平移。

支持关节移动，并优化逻辑。单击关节选中后呈现类Blender的Gismo箭头，可以对关节进行精确的三维拖拽。

![57d5a9b4d72f84562eb1582b635acfcb](/Users/zhayi/University/大四上/计算机动画的算法与技术/animation-skeleton/report_mid.assets/57d5a9b4d72f84562eb1582b635acfcb.png)

## 五、参考材料

获取蒙皮：https://github.com/alecjacobson/common-3d-test-models

骨架和动作参考：https://www.anything.world/

blender使用和骨架编辑：https://docs.blender.org/manual/zh-hans/2.80/animation/armatures/bones/editing/bones.html

部分想法和部分代码实现使用大模型辅助参考。