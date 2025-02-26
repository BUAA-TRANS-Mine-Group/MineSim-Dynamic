# 导入Python library
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union


class LogMessagePreprocessor:
    """
    可以处理每帧要记录的信息；
    可以处理初始化时候记录的单次消息；
    # 注意：JSON序列化只支持 List dict等python基础结构
    Object of type ndarray is not JSON serializable
    """

    def __init__(self):
        # 用于存储初始化时记录的单次消息
        self.single_log_message = None
        # 用于存储每帧要记录的信息
        self.frame_log_messages = None

        self.now_t = 0.0
        self.now_t_str: str = "0.0"
        self.now_t_step: int = 0

    def update_now_time(self, observation):
        self.now_t = observation["test_setting"]["t"]
        self.now_t_str = str(round(self.now_t, 1))
        self.now_t_step = int(round(self.now_t / observation["test_setting"]["dt"], 1))

    def process_single_log_message(
        self,
        observation,
        predicor=None,
        decision=None,
        route_planner=None,
        local_planner=None,
        control=None,
        configuration=None,
    ) -> dict:
        self.update_now_time(observation=observation)

        route_planner_info = self._process_route_planner_info(route_planner=route_planner)

        self.single_log_message = {"sceniro_name": observation["test_setting"]["scenario_name"], "route_planner_info": route_planner_info}

        return self.single_log_message

    def process_frame_log_messages(
        self,
        observation,
        predicor=None,
        decision=None,
        local_planner=None,
        control=None,
    ) -> dict:
        """处理每帧要记录的信息, 用于传入JSON存储函数中.
            note：记录 prediction, decision, planning, control等算法中间变量

        参数:
        observation : 仿真环境观测信息。必须的部分


        返回:
        log_message (Dict):要记录的日志消息。
        """
        # 如果提供了extra_info，则将其添加到日志记录中
        self.update_now_time(observation=observation)

        # TODO 规划信息还需要精简
        local_planner_info = self._process_local_planner_info(local_planner=local_planner)

        extra_info = {
            "vehicle_info": observation["vehicle_info"],  # 动态障碍物车辆+自车信息
            # "static_obstacles": observation["static_obstacles"],
            "local_planner": local_planner_info,
        }

        self.frame_log_messages = {"time_s": self.now_t, "details": extra_info}

        return self.frame_log_messages

    def _process_route_planner_info(self, route_planner):
        pass
        ego_start_state = {
            "x": route_planner.ego_start_state.x,
            "y": route_planner.ego_start_state.y,
            "yaw_rad": route_planner.ego_start_state.yaw,
            "v_mps": route_planner.ego_start_state.v,
            "a_mpss": route_planner.ego_start_state.a,
            "yaw_rate": route_planner.ego_start_state.yaw_rate,
        }
        ego_goal_range = route_planner.ego_goal_range
        refline = route_planner.refline.tolist()
        refline_smooth = route_planner.refline_smooth.tolist()

        route_planner_info = {
            "ego_start_state": ego_start_state,
            "refline": refline,
            "refline_smooth": refline_smooth,
            "ego_goal_range": ego_goal_range,
        }

        return route_planner_info

    def _process_local_planner_info(self, local_planner):
        """# TODO 规划信息还需要精简"""

        if local_planner.method == "frenet":
            trajectories = local_planner.fplanner.all_trajs[self.now_t_step]
            best_traj = self._process_sampling_trajectory(local_planner.fplanner.best_traj)
        else:
            best_traj = None

        local_planner_info = {
            "best_traj": best_traj,
            "trajectories": [0.0, 1, 22, 33],
        }
        return local_planner_info

    def _process_sampling_trajectory(self, traj):
        traj_log = {
            "x": traj.x.tolist(),
            "y": traj.y.tolist(),
            "yaw": traj.yaw.tolist(),
            "s": traj.s,  # List
            "s_d": traj.s_d,
            "d": traj.d,
            "c": traj.c.tolist(),
            "c_d": traj.c_d.tolist(),
        }

        return traj_log
