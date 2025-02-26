import copy
import sys, os

import numpy as np


from devkit.common.utils.math_utils import unify_angle_range
from devkit.common.utils.math_utils import unify_angle_range


class State(object):
    def __init__(self, t: float = 0.0, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, v: float = 0.0, a: float = 0.0, yaw_rate: float = 0.0):
        """
        - FOP,OnsiteMine 定义的 ego 车辆后轴中心 状态
        - 注:yaw x方向为正0,逆时针为正,取值范围[-PI,PI]."""
        self.t = t
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.a = a
        self.yaw_rate = yaw_rate


class FrenetState(object):
    def __init__(
        self,
        t: float = 0.0,
        s: float = 0.0,
        s_d: float = 0.0,
        s_dd: float = 0.0,
        s_ddd: float = 0.0,
        d: float = 0.0,
        d_d: float = 0.0,
        d_dd: float = 0.0,
        d_ddd: float = 0.0,
    ):
        """FOP,OnsiteMine 定义的 ego 车辆后轴中心 Frenet 状态"""
        self.t = t
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.s_ddd = s_ddd
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.d_ddd = d_ddd

    def __str__(self):
        return f"FrenetState with d={self.d:.2f}, s_d={self.s_d:.2f}, t={self.t:.2f}"

    def update_frenetstate_from_state(self, state: State, polyline: np.ndarray):
        """根据自车笛卡尔坐标系的状态值,自车参考路径 计算自车 frenet 坐标系的状态值.

        - 笛卡尔Cartesian坐标系 转Frenet 坐标系.
        - 结果: Frenet坐标系下车辆某一time step的 State.
        - 左正右负,前正后负.

        Args:
           state (State): 自车 状态.
           polyline (np.ndarray):  全局规划的参考路径 中心线,0.1m等间隔采样[x, y, yaw , curve]*N.
        """

        def find_nearest_point_idx(state: State, polyline: np.ndarray):
            """找到当前状态最近的路径点索引."""
            distances = np.hypot(polyline[:, 0] - state.x, polyline[:, 1] - state.y)
            return np.argmin(distances)

        def find_next_point_idx(state: State, polyline: np.ndarray):
            """Get  the next waypoint ids of 参考路径"""
            # 1)
            nearest_idx = find_nearest_point_idx(state, polyline)

            # 2) 使用 np.arctan2 计算从当前状态位置指向最近点的方向角度.
            heading = np.arctan2(polyline[nearest_idx, 1] - state.y, polyline[nearest_idx, 0] - state.x)
            # 3) 计算当前状态的 yaw（朝向角度）与上述方向角度之间的差值.
            angle = abs(state.yaw - heading)
            angle = min(2 * np.pi - angle, angle)  # 将角度调整到 0 至 2*pi 之间的最小值

            # 4) 判断下一个点:
            # 如果角度差大于 pi/2,表示当前状态朝向与最近点之间的角度差距较大,此时选择最近点的下一个点作为下一个路径点.
            # 否则,选择最近点作为下一个路径点.
            if angle > np.pi / 2:
                next_wp_id = nearest_idx + 1
            else:
                next_wp_id = nearest_idx

            # 5)边界情况处理:
            # 如果计算出的下一个点索引小于 1,则将其设置为 1.
            # if it is behind the start of the waypoint list
            if next_wp_id < 1:
                next_wp_id = 1
            # 如果计算出的下一个点索引超过了路径点的总数,则将其设置为路径点总数的最后一个索引.
            # if it reaches the end of the waypoint list
            elif next_wp_id >= polyline.shape[0]:
                next_wp_id = polyline.shape[0] - 1

            return next_wp_id

        # 1）Get the previous and the next waypoint ids
        next_wp_id = find_next_point_idx(state, polyline)  # 最近匹配点的下一个路径点
        prev_wp_id = max(next_wp_id - 1, 0)  # 最近匹配点

        # compute_two_pose_error
        # 2）计算向量:
        # 计算从前一个路径点到下一个路径点的向量（n_x, n_y）.
        # vector n from previous waypoint to next waypoint
        n_x = polyline[next_wp_id, 0] - polyline[prev_wp_id, 0]
        n_y = polyline[next_wp_id, 1] - polyline[prev_wp_id, 1]
        # 计算从前一个路径点到当前位置的向量（x_x, x_y）.
        # vector x from previous waypoint to current position
        x_x = state.x - polyline[prev_wp_id, 0]
        x_y = state.y - polyline[prev_wp_id, 1]

        # 3)计算投影:
        # 计算当前位置向量在路径向量上的投影（proj_x, proj_y）.
        # 计算横向偏差（d）,即当前位置到投影点的距离.
        x_yaw = unify_angle_range(np.arctan2(x_y, x_x))
        # find the projection of x on n
        # print(f"numerator: {(x_x * n_x + x_y * n_y)}")
        # print(f"demominator: {(n_x * n_x + n_y * n_y)}")
        proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
        proj_x = proj_norm * n_x
        proj_y = proj_norm * n_y
        self.d = np.hypot(x_x - proj_x, x_y - proj_y)

        # 4) 调整横向偏差的符号:
        # 根据路径点和当前位置的方向角度调整横向偏差的符号.
        # calculate d value
        wp_yaw = polyline[prev_wp_id, 2]
        delta_yaw = unify_angle_range(state.yaw - wp_yaw)

        # 计算叉积,确定点在路径的左侧还是右侧
        cross = n_x * x_y - n_y * x_x
        if cross < 0:
            self.d *= -1

        # 请确保航向角（yaw）值在计算时是统一的,即它们都应该在相同的范围内（例如0到2 pi或-pi到pi）
        # if wp_yaw <= x_yaw: # CommonRoad
        #     self.d *= -1

        # 5)计算沿道路的距离（s）:
        # 累加从路径起点到前一个路径点的距离.
        # calculate s value
        self.s = 0
        for i in range(prev_wp_id):
            self.s += np.hypot(polyline[i + 1, 0] - polyline[i, 0], polyline[i + 1, 1] - polyline[i, 1])

        # 6)计算 Frenet 坐标的导数:
        # 计算沿道路距离的一阶导数（s_d）,即沿道路的速度.
        # 计算横向偏差的一阶导数（d_d）,即横向速度.
        # 二阶和三阶导数被设为0,表示没有加速度和加加速度的信息.
        # calculate s_d and d_d
        self.t = state.t
        self.s_d = state.v * np.cos(delta_yaw)
        self.s_dd = 0.0
        self.s_ddd = 0.0
        self.d_d = state.v * np.sin(delta_yaw)
        self.d_dd = 0.0
        self.d_ddd = 0.0


class FrenetTrajectory:
    """FrenetTrajectory
    Frenet 坐标系中规划的轨迹.该类不仅包含轨迹的状态,还包含与路径规划相关的附加信息,如代价和状态标志.

    属性:
    ------
       `idx`: 轨迹的唯一标识符（例如 0-100）.
       `t, s, d 等`: Frenet 坐标系 中轨迹的各个状态变量,如时间,位置,速度等.
       `x, y, yaw, ds, c, c_d`:全局坐标系中的轨迹点信息,
                               路径上的全局 x, y 坐标,偏航角（yaw）,曲率（curvature）以及曲率的一阶和二阶导数（c_d 和 c_dd）

    方法:
    ------
       `state_at_time_step`:在特定时间步获取轨迹的状态.
       `frenet_state_at_time_step`:在特定时间步获取 Frenet 坐标系下的轨迹状态.


    特殊方法:
    ------
       (`__eq__`, `__lt__`, 等):定义了轨迹对象间的比较逻辑,主要基于最终代价 cost_total.
       `__repr__` 和 `__str__ `方法提供了类的字符串表示,便于调试和输出.
    """

    def __init__(self):
        self.end_state = None
        # 属性:Cost建模值
        self.cost_total = float("inf")

        # 属性:路径的时间步长
        self.t = []  # list 时间按照固定间隔离散采样 ,时间步list

        # 属性:frenet 坐标系中的路径点信息
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []

        # 属性:全局坐标系中的路径点信息
        self.x = []  # list x 按照固定时间 self.t  间隔离散采样
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []  # 曲率（curvature）以及曲率的一阶和二阶导数（c_d 和 c_dd）
        self.c_d = []
        self.c_dd = []

    # 特殊方法 (__eq__, __lt__, 等):定义了轨迹对象间的比较逻辑,主要基于代价 cost_total.
    def __eq__(self, other):
        return self.cost_total == other.cost_total

    def __ne__(self, other):
        return self.cost_total != other.cost_total

    def __lt__(self, other):
        return self.cost_total < other.cost_total

    def __le__(self, other):
        return self.cost_total <= other.cost_total

    def __gt__(self, other):
        return self.cost_total > other.cost_total

    def __ge__(self, other):
        return self.cost_total >= other.cost_total

    def __repr__(self):
        return "%f" % (self.cost_total)

    def __str__(self):
        return f"FrenetTrajectory with cost_total={self.cost_total:.2f},  d={self.end_state.d:.2f}, s_d={self.end_state.s_d:.2f}, t={self.end_state.t:.2f}"

    def state_at_time_step(self, t: int) -> State:
        """在特定时间步获取轨迹状态(笛卡尔坐标系)
        当前时间步为0,下一个时间步为1,......
        Function to get the state of a trajectory at a specific time instance.

        :param time_step: considered time step
        :return: state of the trajectory at time_step
        """
        assert t < len(self.s) and t >= 0  # assert 语句确保在请求轨迹的特定时间步状态时不会超出轨迹的长度.

        return State(self.t[t], self.x[t], self.y[t], self.yaw[t], self.s_d[t], self.s_dd[t])

    def frenet_state_at_time_step(self, t: int) -> FrenetState:
        """在特定时间步获取 Frenet 坐标系下的轨迹状态.
        当前时间步为0,下一个时间步为1,......
        Function to get the state of a trajectory at a specific time instance.

        :param time_step: considered time step
        :return: state of the trajectory at time_step
        """
        assert t < len(self.s) and t >= 0

        return FrenetState(self.t[t], self.s[t], self.s_d[t], self.s_dd[t], self.s_ddd[t], self.d[t], self.d_d[t], self.d_dd[t], self.d_ddd[t])


class GoalSampledFrenetTrajectory(FrenetTrajectory):
    def __init__(self):
        super().__init__()
        """
        除了继承自父类的属性, GoalSampledFrenetTrajectory 还定义了自己的属性,
        """
        # 属性:轨迹的各种状态标志,
        self.is_generated = False  # 轨迹的各种状态标志, 是否已生成
        self.is_searched = False  # 轨迹的各种状态标志, 是否被搜索过
        self.constraint_passed = False  # 轨迹的各种状态标志, 是否 通过约束检查
        self.collision_passed = False  # 轨迹的各种状态标志, 是否已通过碰撞检查


class JerkSpaceSamplingFrenetTrajectory(FrenetTrajectory):
    def __init__(self):
        super().__init__()
        # 属性:轨迹的各种状态标志
        # self.is_pruned_by_constraint = False  # 未通过约束检查,碰撞检查,将会被剪枝
        # self.is_pruned_by_collision = False  # 未通过约束检查,碰撞检查,将会被剪枝
        # self.is_pruned_by_clustered = False  # 聚合后,将会被剪枝
        self.is_pruned = False  # 剪枝
