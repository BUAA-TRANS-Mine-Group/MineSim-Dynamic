import os
import sys
import math
import numpy as np

# local lib
from devkit.common.coordinate_system.frenet import GoalSampledFrenetTrajectory
from devkit.configuration.configuration_parameters import paras_planner
from devkit.common.actor_state.commonroad_vehicle.vehicle import Vehicle
from devkit.common.coordinate_system.frenet import JerkSpaceSamplingFrenetTrajectory

# ZOOM_FACTOR = 0.9999
ZOOM_FACTOR = 1.0


class GoalSampledCostFunction:
    # FISS+论文中的成本评估函数：对每个轨迹采样 进行代价估计;
    def __init__(self, cost_type: str, vehicle_info: Vehicle, max_road_width: float):
        if cost_type == "WX1":
            # !权重范围设置在[0.0,10.0]
            # self.w_time = 1.0
            self.w_time = paras_planner["planner_frenet"]["cost_wights"][0]
            # w_distance 应该是其它的50-80倍,因为其它的每个step均有
            # self.w_distance = 5.4
            self.w_distance = paras_planner["planner_frenet"]["cost_wights"][1] * 65
            # self.w_laneoffset = 10.0
            self.w_laneoffset = paras_planner["planner_frenet"]["cost_wights"][2]
            # self.w_lateral_accel = 0.9  # lateral_accel_max
            self.w_lateral_accel = paras_planner["planner_frenet"]["cost_wights"][3]
            # self.w_lateral_jerk = 1.0  # lateral_jerk_max
            self.w_lateral_jerk = paras_planner["planner_frenet"]["cost_wights"][4]
            # self.w_longitudinal_a = 3  # longitudinal_a_max
            self.w_longitudinal_a = paras_planner["planner_frenet"]["cost_wights"][5]
            # self.w_longitudinal_jerk = 4  # longitudinal_jerk_max
            self.w_longitudinal_jerk = paras_planner["planner_frenet"]["cost_wights"][6]
            # self.w_longitudinal_v = 0.5  # longitudinal_v_max
            self.w_longitudinal_v = paras_planner["planner_frenet"]["cost_wights"][7]
            # self.w_centripetal_accel = 0.35  # centripetal_accel_max

            self.lane_width_max_half = (max_road_width - vehicle_info.w) * 0.5
            self.vehicle_info = vehicle_info
            self.max_distance = paras_planner["planner_frenet"]["max_distance"]

    def cost_time(self) -> float:
        pass

    def cost_terminal_time(self, terminal_time: float, t_max: float, t_min: float) -> float:
        """最终时间成本."""
        return self.w_time * (1.0 - (terminal_time - t_min) / t_max - t_min) ** 2

    def cost_dist_obstacle(self, obstacles: list, time_step_now: int = 0) -> float:
        """当前帧 所有障碍物距离成本"""
        pass

    def cost_distance(self, final_distance: float, max_distance: float):
        if final_distance >= max_distance:
            return 0.0
        else:
            return self.w_distance * (1.0 - final_distance / max_distance) ** 2

    def cost_velocity_offset(self, vels: list, v_target: float, weight: float) -> float:
        """成本项:偏离参考速度

        Args:
            vels (list): final states sampling 速度
            v_target (float): final位置的参考速度值

        Returns:
            float: _description_
        """
        if v_target == 0:
            raise ValueError("目标速度 v_target 不能为 0,以避免除以 0 的错误.")

        # 计算每个速度相对于目标速度的比例偏差的平方
        costs = []
        for vel in vels:
            costs.append(weight * (1.0 - vel / v_target) ** 2)
            weight *= ZOOM_FACTOR  # 权重系数依据时间减弱

        return sum(costs)

    def cost_acceleration(self, accels: list[float], accel_max: float, weight: float) -> float:
        """加速度代价."""
        costs = []
        for accel in accels:
            accel_temp = abs(accel)
            if accel_temp >= accel_max:
                costs.append(weight)
            else:
                costs.append(weight * (accel_temp / accel_max) ** 2)
            weight *= ZOOM_FACTOR  # 权重系数依据时间减弱
        return sum(costs)

    def cost_jerk(self, jerks: list[float], jerk_max: float, weight: float) -> float:
        """加加速度代价."""
        costs = []
        for jerk in jerks:
            jerk_temp = abs(jerk)
            if jerk_temp >= jerk_max:
                costs.append(weight)
            else:
                costs.append(weight * (jerk_temp / jerk_max) ** 2)
            weight *= ZOOM_FACTOR  # 权重系数依据时间减弱
        return sum(costs)

    def cost_lane_center_offset(self, offsets: list[float], weight: float) -> float:
        """偏离车道中心线的代价."""
        costs = []
        for offset in offsets:
            offset_temp = abs(offset)
            if offset_temp >= self.lane_width_max_half:
                costs.append(weight)
            else:
                costs.append(weight * (offset_temp / self.lane_width_max_half) ** 2)
            weight *= ZOOM_FACTOR  # 权重系数依据时间减弱
        return sum(costs)

    def cost_total(self, traj: GoalSampledFrenetTrajectory, target_speed: float, t_max: float, t_min: float) -> float:
        """对采样轨迹 进行代价估计：
        耗时+期望速度偏离+加速度支出+加加速度支出+偏离车道线中心的代价,公式4
        """
        cost_time = self.cost_terminal_time(terminal_time=traj.t[-1], t_max=t_max, t_min=t_min)
        cost_obstacle = 0.0  # self.cost_dist_obstacle()
        cost_dis = self.cost_distance(final_distance=traj.s[-1], max_distance=self.max_distance)
        cost_speed = self.cost_velocity_offset(vels=traj.s_d, v_target=target_speed, weight=self.w_longitudinal_v)
        cost_accel = self.cost_acceleration(
            accels=traj.s_dd, accel_max=self.vehicle_info.max_accel, weight=self.w_longitudinal_a
        ) + self.cost_acceleration(accels=traj.d_dd, accel_max=self.vehicle_info.max_lateral_accel, weight=self.w_lateral_accel)

        cost_jerk = self.cost_jerk(jerks=traj.s_ddd, jerk_max=self.vehicle_info.max_jerk, weight=self.w_longitudinal_jerk) + self.cost_jerk(
            jerks=traj.d_ddd, jerk_max=self.vehicle_info.max_lateral_jerk, weight=self.w_lateral_jerk
        )

        cost_offset = self.cost_lane_center_offset(offsets=traj.d, weight=self.w_laneoffset)
        # return cost_speed + cost_accel + cost_jerk + cost_offset
        cost_total = (cost_time + cost_dis + cost_speed + cost_accel + cost_jerk + cost_offset) / len(traj.t)

        return cost_total


class JerkSpaceSamplingCostFunction:
    def __init__(self, cost_type: str, vehicle_info: None, max_road_width: float = None):
        if cost_type == "WX1":
            # !权重范围设置在[0.0,10.0]
            # self.w_time = paras_planner["planner_JSSP"]["cost_wights"][0]
            # w_distance 应该是其它的50倍,因为其它的每个step均有
            self.w_distance = paras_planner["planner_JSSP"]["cost_wights"][1] * paras_planner["planner_JSSP"]["plan_horizon"] / 0.1
            self.w_laneoffset = paras_planner["planner_JSSP"]["cost_wights"][2]
            self.w_lateral_accel = paras_planner["planner_JSSP"]["cost_wights"][3]
            self.w_lateral_jerk = paras_planner["planner_JSSP"]["cost_wights"][4]
            self.w_longitudinal_a = paras_planner["planner_JSSP"]["cost_wights"][5]
            self.w_longitudinal_jerk = paras_planner["planner_JSSP"]["cost_wights"][6]
            self.w_longitudinal_v = paras_planner["planner_JSSP"]["cost_wights"][7]

            self.lane_width_max_half = (max_road_width - vehicle_info.w) * 0.5
            self.vehicle_info = vehicle_info
            self.max_distance = paras_planner["planner_frenet"]["max_distance"]
            # parameters["vehicle_para"]["speed_max"] * paras_planner["planner_JSSP"]["plan_horizon"]

    def cost_time(self) -> float:
        pass

    def cost_terminal_time(self, terminal_time: float, t_max: float, t_min: float) -> float:
        """最终时间成本."""
        return self.w_time * (1.0 - (terminal_time - t_min) / t_max - t_min) ** 2

    def cost_dist_obstacle(self, obstacles: list, time_step_now: int = 0) -> float:
        """当前帧 所有障碍物距离成本"""
        pass

    def cost_distance(self, final_distance: float, max_distance: float):
        if final_distance >= max_distance:
            return 0.0
        else:
            return self.w_distance * (1.0 - final_distance / max_distance) ** 2

    def cost_velocity_offset(self, vels: list, v_target: float, weight: float) -> float:
        """成本项:偏离参考速度

        Args:
            vels (list): final states sampling 速度
            v_target (float): final位置的参考速度值

        Returns:
            float: _description_
        """
        if v_target == 0:
            raise ValueError("目标速度 v_target 不能为 0,以避免除以 0 的错误.")

        # 计算每个速度相对于目标速度的比例偏差的平方
        costs = []
        for vel in vels:
            costs.append(weight * (1.0 - vel / v_target) ** 2)
            weight *= ZOOM_FACTOR  # 权重系数依据时间减弱

        return sum(costs)

    def cost_acceleration(self, accels: list[float], accel_max: float, weight: float) -> float:
        """加速度代价."""
        costs = []
        for accel in accels:
            accel_temp = abs(accel)
            if accel_temp >= accel_max:
                costs.append(weight)
            else:
                costs.append(weight * (accel_temp / accel_max) ** 2)
            weight *= ZOOM_FACTOR  # 权重系数依据时间减弱
        return sum(costs)

    def cost_jerk(self, jerks: list[float], jerk_max: float, weight: float) -> float:
        """加加速度代价."""
        costs = []
        for jerk in jerks:
            jerk_temp = abs(jerk)
            if jerk_temp >= jerk_max:
                costs.append(weight)
            else:
                costs.append(weight * (jerk_temp / jerk_max) ** 2)
            weight *= ZOOM_FACTOR  # 权重系数依据时间减弱
        return sum(costs)

    def cost_lane_center_offset(self, offsets: list[float], weight: float) -> float:
        """偏离车道中心线的代价."""
        costs = []
        for offset in offsets:
            offset_temp = abs(offset)
            if offset_temp >= self.lane_width_max_half:
                costs.append(weight)
            else:
                costs.append(weight * (offset_temp / self.lane_width_max_half) ** 2)
            weight *= ZOOM_FACTOR  # 权重系数依据时间减弱
        return sum(costs)

    def cost_total(self, traj: JerkSpaceSamplingFrenetTrajectory, target_speed: float) -> float:
        """对采样轨迹 进行代价估计：
        耗时+期望速度偏离+加速度支出+加加速度支出+偏离车道线中心的代价,公式4
        """
        cost_obstacle = 0.0  # self.cost_dist_obstacle()
        cost_dis = self.cost_distance(final_distance=traj.s[-1], max_distance=self.max_distance)
        cost_speed = self.cost_velocity_offset(vels=traj.s_d, v_target=target_speed, weight=self.w_longitudinal_v)
        cost_accel = self.cost_acceleration(
            accels=traj.s_dd, accel_max=self.vehicle_info.max_accel, weight=self.w_longitudinal_a
        ) + self.cost_acceleration(accels=traj.d_dd, accel_max=self.vehicle_info.max_lateral_accel, weight=self.w_lateral_accel)

        cost_jerk = self.cost_jerk(jerks=traj.s_ddd, jerk_max=self.vehicle_info.max_jerk, weight=self.w_longitudinal_jerk) + self.cost_jerk(
            jerks=traj.d_ddd, jerk_max=self.vehicle_info.max_lateral_jerk, weight=self.w_lateral_jerk
        )
        cost_offset = self.cost_lane_center_offset(offsets=traj.d, weight=self.w_laneoffset)
        # return cost_speed + cost_accel + cost_jerk + cost_offset
        traj.cost_total = (cost_dis + cost_speed + cost_accel + cost_jerk + cost_offset) / len(traj.t)

        return traj
