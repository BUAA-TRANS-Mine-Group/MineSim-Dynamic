import copy

import numpy as np


class PathSearchTree:
    """用于路径搜索,采取树搜索方法.
    search_tree.tree = [
    [RootNode],  # 第0层:只有一个根节点
    [ChildNode1, ChildNode2, ChildNode3],  # 第1层:根节点的子节点,假设有3个
    [GrandChildNode1, GrandChildNode2, GrandChildNode3, GrandChildNode4, GrandChildNode5, GrandChildNode6]
    # 第2层:第1层节点的子节点,假设每个第1层的节点又各有2个子节点
    ]
    # 假设结构如下:
    # RootNode
    # ├── ChildNode1
    # │   ├── GrandChildNode1
    # │   └── GrandChildNode2
    # ├── ChildNode2
    # │   ├── GrandChildNode3
    # │   └── GrandChildNode4
    # └── ChildNode3
    #     ├── GrandChildNode5
    #     └── GrandChildNode6
    层级的组织方式使得对搜索树进行遍历,查找,添加或删除节点等操作变得直接且高效.
    """

    def __init__(self, node_num_max: int = 50):
        self.tree = [[]]  # 初始化搜索树,第一层为根节点层
        self.prune_info_per_layer = []  # 存储每层的剪枝信息
        self.node_num_max = node_num_max
        self.reached_goal = False
        self.goal_node_path = None

    def add_node_to_search_tree(self, parent_node=None, child_node=None, layer: int = 0):
        """添加节点到搜索树."""
        while len(self.tree) <= layer:
            self.tree.append([])
        # 如果父节点不为None,添加子节点到父节点的子节点列表中
        if parent_node is not None:
            if "parent" not in child_node.keys():
                child_node["parent"] = []
                child_node["parent"] = parent_node  # 设置子节点的父节点引用

        # 将子节点添加到搜索树的指定层中
        self.tree[layer].append(child_node)

    def get_nodes_of_layer(self, layer: int):
        nodes = self.tree[layer]
        return nodes

    def backtrack_from_leaf_node(self):
        """从搜索树的目标叶节点回溯到根节点"""
        path_segments = []
        current_node = self.goal_node_path
        while current_node is not None:
            path_segments.insert(0, current_node)  # 将当前节点添加到路径段列表的开头
            if "parent" not in current_node.keys():
                # current_node = None
                break
            else:
                current_node = current_node["parent"]  # 移动到父节点
        return path_segments

    def concatenate_paths(self, path_segments):
        """如果每个路径段是一系列点，简单的拼接可能就足够了。否则，你可能需要考虑路径段之间的过渡，确保整个路径的连续性和平滑性。"""
        full_path_waypoints = []
        for segment in path_segments:
            full_path_waypoints += segment["waypoints"]
        return full_path_waypoints

    def concatenate_paths_delete_middle_segment_points(self, path_segments):
        """对于中间的路径段，去掉首尾各一个点，但保留第一个和最后一个路径段的所有点"""
        full_path_waypoints = []
        for i, segment in enumerate(path_segments):
            if i == 0 or i == len(path_segments) - 1:
                # 对于第一个和最后一个路径段，保留所有点
                full_path_waypoints += segment["waypoints"]
            else:
                # 对于中间的路径段，如果长度允许，去掉首尾各一个点
                if len(segment["waypoints"]) > 3:
                    full_path_waypoints += segment["waypoints"][1:-1]
                else:
                    # 如果路径段太短，则保留所有点
                    full_path_waypoints += segment["waypoints"]
        return full_path_waypoints
