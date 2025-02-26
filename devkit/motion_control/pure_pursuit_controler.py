import numpy as np
from configuration_parameters import parameters, paras_control


# TODO
class PurePursuitController:
    """纯跟踪算法.[横向跟踪控制模块]"""

    def __init__(self):
        self._min_prev_dist = 14.0  # 最小预瞄距离

    # todo 待调试
    def pure_pursuit_control(self, kv: float = 0.25, ld0: float = 2.8):
        """路径跟踪控制,纯跟踪控制器.获得控制量:前轮转向角度

        Args:
            kv (float, optional): 预瞄距离系数,用于根据车速调整预瞄距离.
            ld0 (float, optional):  预瞄距离的下限值,确保在低速时有一个最小的预瞄距离. Defaults to 5.0.

        Returns:
            float: steer_angle
        """
        kv = paras_control["pure_pursuit"]["kv_ld0"][0]
        ld0 = paras_control["pure_pursuit"]["kv_ld0"][1]
        coff_wheel_base = paras_control["pure_pursuit"]["coff_wheel_base"]

        if parameters["sim_config"]["kinematics_path_tracing_type"] == "ref_path":
            ref_pos = self.ref_ego_lane_pts[:, 0:2]  # 参考路径
        elif parameters["sim_config"]["kinematics_path_tracing_type"] == "plan_path":
            ref_pos = np.column_stack((self.fplanner.best_traj.x, self.fplanner.best_traj.y))  # 规划路径
        else:
            raise Exception("#log#kinematics_path_tracing_type error")

        # 计算预瞄距离
        ld = kv * self.ego_state.v + ld0
        pos = np.array([self.ego_state.x, self.ego_state.y])
        # l = self.ego_vehicle.l / 1.7  # 轴距

        # lookahead_point, idx = self.find_lookahead_point_by_linear_distance(pos, ref_pos, ld)
        lookahead_point, idx = self.find_lookahead_point_by_curve_distance(pos, ref_pos, ld)
        if idx < len(ref_pos):
            point_temp = lookahead_point
        else:
            point_temp = ref_pos[-1]  # 使用最后一个参考点

        # 计算航向误差 alpha =预瞄航向-车辆当前航向
        alpha = np.arctan2(point_temp[1] - pos[1], point_temp[0] - pos[0]) - self.ego_state.yaw
        # 计算前轮转角s
        wheel_base = self.ego_vehicle.l / coff_wheel_base  # 车辆的轴距
        # wheel_base = self.ego_vehicle.l
        steer_angle = np.arctan2(2 * wheel_base * np.sin(alpha), ld)  # coff_wheel_base越小角度越大

        return steer_angle

    @staticmethod
    def find_lookahead_point_by_curve_distance(pos: np.array, ref_pos: np.array, ld: float = 5.0):
        """根据 `曲线`预瞄距离 查找预瞄点.curve distance

        Args:
            pos (np.array): 自车当前位置,格式为[x, y].
            v (float): 车速,单位为米/秒.
            ref_pos (np.array): 待跟踪路径点序列[x, y]*N;
            ld (float): 曲线预瞄距离 ,单位为米.

        Returns:
            tuple:  包含预瞄点的坐标和该点在参考路径数组中的索引.
        """
        # 计算所有参考点到当前位置的距离
        dist = np.linalg.norm(ref_pos - pos, axis=1)
        # 找到距离最近的参考点索引
        idx = np.argmin(dist)

        # 计算预瞄距离
        l_steps = 0  # 累计距离
        size_of_ref_pos = len(ref_pos)

        # 从最近点开始累加距离,直到达到预瞄距离
        while l_steps < ld and idx + 1 < size_of_ref_pos:
            l_steps += np.linalg.norm(ref_pos[idx + 1] - ref_pos[idx])
            idx += 1

        lookahead_point = ref_pos[idx]

        return lookahead_point, idx

    @staticmethod
    def find_lookahead_point_by_linear_distance(pos: np.array, ref_pos: np.array, ld: float = 5.0):
        """根据 `直线`预瞄距离 查找预瞄点.linear distance

        Args:
            pos (np.array): 自车当前位置,格式为[x, y].
            v (float): 车速,单位为米/秒.
            ref_pos (np.array): 待跟踪路径点序列[x, y]*N;
            ld (float): 直线预瞄距离 ,单位为米.

        Returns:
            tuple:  包含预瞄点的坐标和该点在参考路径数组中的索引.
        """
        # 计算所有参考点到当前位置的距离
        # Compute the straight-line distances from the current position to each point on the reference path
        dist = np.linalg.norm(ref_pos - pos, axis=1)

        # Find the index of the first point on the path that is at least ld meters away
        idx = np.where(dist >= ld)[0]

        # If such a point exists, use the first one. Otherwise, use the last point on the path.
        if idx.size > 0:
            idx = idx[0]
            lookahead_point = ref_pos[idx]
        else:
            idx = len(ref_pos) - 1
            lookahead_point = ref_pos[-1]

        return lookahead_point, idx
