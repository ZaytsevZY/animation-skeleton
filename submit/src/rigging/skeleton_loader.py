# rigging/skeleton_loader.py
"""
GLB 骨架加载模块 - 全新实现
从 GLB 文件中提取骨架结构和网格数据
"""

import numpy as np
import trimesh
from .skeleton import Joint, Skeleton


def load_skeleton_from_glb(glb_path, scale=1.0, verbose=True):
    """
    从 GLB 文件加载骨架

    Parameters:
    -----------
    glb_path : str
        GLB 文件路径
    scale : float
        缩放因子
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    skeleton : Skeleton
        骨架对象
    bones : list of tuple
        骨骼连接关系 [(parent_idx, child_idx), ...]
    """

    if verbose:
        print(f"   正在加载 GLB 文件: {glb_path}")

    # 加载场景
    scene = trimesh.load(glb_path, process=False)

    if not hasattr(scene, 'graph'):
        raise ValueError("GLB 文件中未找到场景图结构")

    graph = scene.graph

    # 第一步：识别所有骨骼节点
    joint_nodes = _extract_joint_nodes(graph, scene, verbose)

    if len(joint_nodes) == 0:
        raise ValueError("未找到骨骼节点")

    # 第二步：构建父子关系
    parent_map = _build_parent_map(graph, joint_nodes, verbose)

    # 第三步：拓扑排序（确保父节点在前）
    sorted_nodes = _topological_sort(joint_nodes, parent_map)

    # 第四步：创建 Joint 对象
    joints_list = []
    node_to_idx = {}

    for idx, node_name in enumerate(sorted_nodes):
        # 获取节点的世界变换矩阵
        transform, _ = graph.get(node_name)
        position = transform[:3, 3] * scale  # 提取平移部分

        # 找父节点索引
        parent_name = parent_map.get(node_name)
        parent_idx = -1
        if parent_name and parent_name in node_to_idx:
            parent_idx = node_to_idx[parent_name]

        # 创建关节
        joint = Joint(
            name=_clean_name(node_name),
            pos=position.astype(np.float32),
            parent=parent_idx
        )

        joints_list.append(joint)
        node_to_idx[node_name] = idx

        if verbose:
            parent_str = _clean_name(parent_name) if parent_name else "Root"
            print(f"      [{idx:2d}] {joint.name:25s} <- {parent_str}")

    # 创建骨架
    skeleton = Skeleton(joints=joints_list)

    # 构建骨骼列表
    bones = []
    for i in range(skeleton.n):
        if skeleton.joints[i].parent >= 0:
            bones.append((skeleton.joints[i].parent, i))

    if verbose:
        print(f"   ✅ 加载完成: {skeleton.n} 关节, {len(bones)} 骨骼")

    return skeleton, bones


def _extract_joint_nodes(graph, scene, verbose=True):
    """提取所有骨骼节点"""
    joint_nodes = []

    # 排除明确不是骨骼的节点
    exclude_nodes = {'world'}
    exclude_keywords = ['camera', 'light', 'Camera', 'Light']

    # 获取网格节点（排除它们）
    mesh_nodes = set()
    if hasattr(scene, 'geometry'):
        mesh_nodes = set(scene.geometry.keys())

    for node_name in graph.nodes:
        # 跳过排除项
        if node_name in exclude_nodes:
            continue
        if node_name in mesh_nodes:
            continue
        if any(kw in node_name for kw in exclude_keywords):
            continue

        # 包含骨骼相关的节点
        joint_keywords = [
            'rig', 'bone', 'joint', 'armature',
            'body', 'leg', 'arm', 'head', 'neck', 'spine',
            'hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist',
            'tail', 'foot', 'hand'
        ]

        if any(kw.lower() in node_name.lower() for kw in joint_keywords):
            joint_nodes.append(node_name)

    if verbose:
        print(f"   找到 {len(joint_nodes)} 个骨骼节点")

    return joint_nodes


def _build_parent_map(graph, joint_nodes, verbose=True):
    """
    构建父子关系映射

    尝试多种方法从场景图中提取父子关系
    """
    parent_map = {}

    # 方法1：通过 graph 的边直接查找
    if hasattr(graph, 'transforms') and hasattr(graph.transforms, 'parents'):
        if verbose:
            print(f"   尝试方法1: graph.transforms.parents")

        for node in joint_nodes:
            parents = graph.transforms.parents.get(node, [])
            parent = None

            # 找到第一个也是骨骼的父节点
            for p in parents:
                if p in joint_nodes:
                    parent = p
                    break

            parent_map[node] = parent

        # 检查是否成功
        valid_parents = sum(1 for p in parent_map.values() if p is not None)
        if verbose:
            print(f"      找到 {valid_parents} 个有效父子关系")

        if valid_parents > 0:
            return parent_map

    # 方法2：通过边迭代
    if verbose:
        print(f"   尝试方法2: 遍历场景图的边")

    parent_map = {}

    # 初始化所有节点为无父节点
    for node in joint_nodes:
        parent_map[node] = None

    # 遍历所有边
    if hasattr(graph.transforms, 'edge_data'):
        for (parent_node, child_node) in graph.transforms.edge_data.keys():
            # 如果两个都是骨骼节点
            if parent_node in joint_nodes and child_node in joint_nodes:
                # 只有在子节点还没有父节点时才设置
                if parent_map[child_node] is None:
                    parent_map[child_node] = parent_node

    valid_parents = sum(1 for p in parent_map.values() if p is not None)
    if verbose:
        print(f"      找到 {valid_parents} 个有效父子关系")

    if valid_parents > 0:
        return parent_map

    # 方法3：基于名称推断（作为最后的备选）
    if verbose:
        print(f"   ⚠️  方法1和2都失败，尝试方法3: 基于名称推断")

    parent_map = _infer_hierarchy_from_names(joint_nodes)

    return parent_map


def _infer_hierarchy_from_names(joint_nodes):
    """
    基于节点名称推断层级关系（备选方案）

    规则：
    - 'rig' 是根节点
    - 'body' 的父节点是 'rig'
    - 'body_bot' 和 'body_top' 的父节点是 'body'
    - 'leg_*_top0' 的父节点是 'body_bot'
    - 'leg_*_top1' 的父节点是 'leg_*_top0'
    - 以此类推...
    """
    parent_map = {}

    # 创建节点名到节点的映射
    name_to_node = {node: node for node in joint_nodes}

    for node in joint_nodes:
        parent = None

        # 根节点
        if node == 'rig':
            parent = None

        # body 系列
        elif node == 'body':
            parent = 'rig'
        elif 'body_bot' in node:
            parent = 'body'
        elif 'body_top' in node:
            if 'top1' in node:
                parent = 'body_top0'
            else:
                parent = 'body'

        # 腿部（后腿）
        elif 'leg_hind' in node:
            if 'top0' in node:
                parent = 'body_bot'
            elif 'top1' in node:
                parent = node.replace('top1', 'top0')
            elif 'bot0' in node:
                parent = node.replace('bot0', 'top1')
            elif 'bot1' in node:
                parent = node.replace('bot1', 'bot0')
            elif 'bot2' in node and 'end' not in node:
                parent = node.replace('bot2', 'bot1')
            elif 'end' in node:
                parent = node.replace('_end', '')

        # 腿部（前腿）
        elif 'leg_front' in node:
            if 'top0' in node:
                parent = 'body_top1'
            elif 'top1' in node:
                parent = node.replace('top1', 'top0')
            elif 'bot0' in node:
                parent = node.replace('bot0', 'top1')
            elif 'bot1' in node or 'lleg' in node:  # 处理拼写错误的节点
                parent = node.replace('bot1', 'bot0').replace('lleg_', 'leg_')
            elif 'bot2' in node and 'end' not in node:
                parent = node.replace('bot2', 'bot1').replace('lleg_', 'leg_')
            elif 'end' in node:
                parent = node.replace('_end', '')

        # 颈部和头部
        elif 'neck0' in node:
            parent = 'body_top1'
        elif 'neck1' in node:
            parent = 'neck0'
        elif 'head0' in node and 'end' not in node:
            parent = 'neck1'
        elif 'head0_end' in node:
            parent = 'head0'

        # 检查推断的父节点是否存在
        if parent and parent in name_to_node:
            parent_map[node] = parent
        else:
            parent_map[node] = None

    return parent_map


def _topological_sort(nodes, parent_map):
    """拓扑排序：确保父节点在子节点之前"""

    # 找根节点
    roots = [n for n in nodes if parent_map.get(n) is None]

    if not roots:
        # 如果没有根节点，返回原始顺序
        return nodes

    # BFS 遍历
    sorted_nodes = []
    queue = roots[:]
    visited = set()

    # 构建子节点映射
    children = {n: [] for n in nodes}
    for child, parent in parent_map.items():
        if parent:
            children[parent].append(child)

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue

        visited.add(node)
        sorted_nodes.append(node)

        # 添加子节点
        for child in children[node]:
            if child not in visited:
                queue.append(child)

    # 添加未访问的节点
    for node in nodes:
        if node not in visited:
            sorted_nodes.append(node)

    return sorted_nodes


def _clean_name(name):
    """清理节点名称"""
    if not name:
        return ""

    # 移除常见前缀
    prefixes = ['mixamorig:', 'Armature|', 'Skeleton|']
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]

    return name


# ========== 网格加载 ==========

def load_mesh_from_glb(glb_path, scale=1.0):
    """
    从 GLB 文件加载网格

    Returns:
    --------
    vertices : np.ndarray (N, 3)
    faces : np.ndarray (M, 3)
    """
    scene = trimesh.load(glb_path)

    if isinstance(scene, trimesh.Trimesh):
        return scene.vertices * scale, scene.faces

    elif isinstance(scene, trimesh.Scene):
        # 合并所有网格
        meshes = []
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)

        if not meshes:
            raise ValueError("场景中无网格数据")

        combined = trimesh.util.concatenate(meshes)
        return combined.vertices * scale, combined.faces

    else:
        raise ValueError(f"未知的场景类型: {type(scene)}")


# ========== 可视化 ==========

def visualize_skeleton_structure(skeleton, bones):
    """打印骨架树形结构"""
    print("\n📊 骨架层级结构:")
    print("=" * 60)

    # 构建子节点映射
    children = {i: [] for i in range(skeleton.n)}
    for parent_idx, child_idx in bones:
        children[parent_idx].append(child_idx)

    # 找根节点
    roots = [i for i in range(skeleton.n) if skeleton.joints[i].parent == -1]

    def print_tree(idx, depth=0):
        joint = skeleton.joints[idx]
        indent = "  " * depth
        symbol = "├─" if depth > 0 else "●"
        print(f"{indent}{symbol} [{idx:2d}] {joint.name}")

        for child in children[idx]:
            print_tree(child, depth + 1)

    for root in roots:
        print_tree(root)

    print("=" * 60)


# ========== 测试 ==========

def test_glb_loader():
    """测试函数"""
    glb_path = "data/cow/cow.glb"

    print("🧪 测试 GLB 加载器\n")

    try:
        skeleton, bones = load_skeleton_from_glb(glb_path, verbose=True)
        visualize_skeleton_structure(skeleton, bones)

        print(f"\n✅ 测试成功！")
        print(f"   关节数: {skeleton.n}")
        print(f"   骨骼数: {len(bones)}")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_glb_loader()