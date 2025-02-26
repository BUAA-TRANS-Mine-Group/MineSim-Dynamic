from __future__ import annotations

import json
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Generator, List, Optional, Set, Tuple, Type, Dict

import numpy as np


from devkit.common.actor_state.agent import Agent
from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.state_representation import TimePoint
from devkit.common.actor_state.state_representation import Point2D
from devkit.common.actor_state.state_representation import StateSE2
from devkit.common.actor_state.state_representation import StateVector2D
from devkit.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.common.actor_state.vehicle_parameters import get_mine_truck_parameters
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.common.trajectory.trajectory_sampling import TrajectorySampling

from devkit.sim_engine.observation_manager.observation_type import DetectionsTracks
from devkit.sim_engine.scenario_manager.scenario_info import ScenarioInfo
from devkit.sim_engine.map_manager.abstract_map import AbstractMap
from devkit.sim_engine.map_manager.minesim_map.minesim_map_factory import get_maps_api

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_to_nuplan import convert_to_MineSimScenarioVehicleTraj_batch
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTasks
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTask
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTaskFinalPose
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import ScenarioFileBaseInfo
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import MineSimScenarioTrackedMetaData
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import SimTimePointOrigin

logger = logging.getLogger(__name__)


class MineSimDynamicScenario(AbstractScenario):
    """Scenario implementation for the MineSim dataset that is used in training and simulation.
    用于训练和模拟的 MineSim 数据集的方案实现。
    """

    def __init__(self, scenario_file: ScenarioFileBaseInfo) -> None:
        """
        Initialize the MineSim scenario.
        """
        self._scenario_file = scenario_file
        self._sim_time_origin = SimTimePointOrigin(sim_time_point_origin=TimePoint(1626469658900612), minesim_metadata_time_step=int(0), dt=0.1)
        self._detections_tracks_frames: List[DetectionsTracks] = []
        self._scenario_info = ScenarioInfo()
        self._scenario_info = self._extraction_info()
        self.minesim_tracked_metadata: MineSimScenarioTrackedMetaData = self._get_minesim_scenario_meta_data()
        self._ego_vehicle_parameters: VehicleParameters = get_mine_truck_parameters(mine_name=self._scenario_file.location)
        self._scenario_type = self._scenario_file.scenario_type
        self.metadata_time_s_list: List[float] = []
        self.metadata_time_str_list: List[str] = []

    # def __reduce__(self) -> Tuple[Type[MineSimDynamicScenario], Tuple[Any, ...]]:
    #     """
    #     Hints on how to reconstruct the object when pickling.
    #     :return: Object type and constructor arguments to be used.
    #     """
    #     return (self.__class__, (self._scenario_file))

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return self._ego_vehicle_parameters

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        # e.g. "2021.07.16.20.45.29_veh-35_01095_01486.db"
        # minesim 'Scenario-dapai_intersection_1_3_4.json'
        return self._scenario_file.log_file_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_file.scenario_name

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_file.scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        mine_map_api = get_maps_api(map_root=self._scenario_file.map_root, map_name=self._scenario_file.location)
        mine_map_api.load_bitmap_using_utm_local_range(utm_local_range=self.calculate_square_region())
        return mine_map_api

    @property
    def map_root(self) -> str:
        """Get the map root folder."""
        return self._scenario_file.map_root

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return float(self._scenario_info.test_setting["dt"])

    @property
    def scenario_info(self) -> ScenarioInfo:
        return self._scenario_info

    @property
    def scenario_file(self) -> ScenarioFileBaseInfo:
        return self._scenario_file

    @property
    def initial_ego_state(self) -> EgoState:
        """lazy load"""
        return self._get_planning_problem_initial_ego_state()

    @property
    def planning_problem_goal_task(self) -> PlanningProblemGoalTasks:
        """lazy load"""
        return self._get_planning_problem_goal_task()

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""
        # return TimePoint(time_us=self._sim_time_origin.sim_time_point_origin.time_us + int(0.1 * iteration * 1e6))
        return TimePoint(time_us=self._sim_time_origin.sim_time_point_origin.time_us + int(iteration * 1e5))

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass.
        场景中obstacles track 的最大时间，时间步
        """
        return len(self._detections_tracks_frames)

    def get_planning_problems_max_iterations(self) -> int:
        """场景中运行的最大迭代值"""
        return int(float(self._scenario_info.test_setting["max_t"]) / self.database_interval)

    def _get_planning_problem_goal_task(self) -> PlanningProblemGoalTasks:
        """lazy load
        e.g. "goal_box": {"x": [...],"y": [...]}, "goal_pose": [2161.9,808.3,-2.1407]
        根据场景信息获取规划问题的目标任务。
        返回 PlanningProblemGoalTaskFinalPose 或 PlanningProblemGoalTask 实例，或者在遇到错误时返回 None。
        """
        if "goal_state" in self._scenario_info.test_setting and "goal" in self._scenario_info.test_setting:
            return PlanningProblemGoalTaskFinalPose(
                scenario_name=self._scenario_file.scenario_name,
                goal_range_xs=self._scenario_info.test_setting["goal"]["x"],
                goal_range_ys=self._scenario_info.test_setting["goal"]["y"],
                final_pose=StateSE2(
                    x=self._scenario_info.test_setting["goal_state"][0],
                    y=self._scenario_info.test_setting["goal_state"][1],
                    heading=self._scenario_info.test_setting["goal_state"][2],
                ),
            )
        elif "goal" in self._scenario_info.test_setting and "goal_state" not in self._scenario_info.test_setting:
            return PlanningProblemGoalTask(
                scenario_name=self._scenario_file.scenario_name,
                goal_range_xs=self._scenario_info.test_setting["goal"]["x"],
                goal_range_ys=self._scenario_info.test_setting["goal"]["y"],
            )
        else:
            logging.error(f"#log# scenario_name={self._scenario_file.scenario_name}: Missing 'goal' or 'goal_state' in test_setting!")
            ValueError("#log# scenario_info.test_setting,goal is error!")

    def _get_planning_problem_initial_ego_state(self) -> EgoState:
        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(
                x=self._scenario_info.ego_info["x"],
                y=self._scenario_info.ego_info["y"],
                heading=self._scenario_info.ego_info["yaw_rad"],
            ),
            rear_axle_velocity_2d=StateVector2D(x=self._scenario_info.ego_info["v_mps"], y=0.0),
            rear_axle_acceleration_2d=StateVector2D(x=self._scenario_info.ego_info["acc_mpss"], y=0.0),
            tire_steering_angle=0.0,
            time_point=self._sim_time_origin.sim_time_point_origin,
            vehicle_parameters=self.ego_vehicle_parameters,
            is_in_auto_mode=True,
            angular_vel=self._scenario_info.ego_info["yawrate_radps"],
            angular_accel=0.0,
            tire_steering_rate=0.0,
        )

    def _extraction_info(self):
        with open(self._scenario_file.log_file_load_path, "r") as f:
            one_scenario = json.load(f)

        # 1) 获取ego车辆的目标区域,goal box
        self._scenario_info.test_setting["goal"] = one_scenario["goal"]

        # 2) 步长,最大时间,最大最小xy范围
        self._scenario_info._get_dt_maxt(one_scenario)

        # 3) 读取ego车初始信息
        self._scenario_info._init_vehicle_ego_info(one_scenario)

        for idex, value_traj_segment in enumerate(one_scenario["TrajSegmentInfo"]):
            # if value_traj_segment['TrajSetToken'] != "ego":
            num_vehicle = idex + 1
            # 4) 读取车辆长度与宽度等形状信息,录入scenario_info.背景车id从1号开始,ego车为0
            self._scenario_info.add_vehicle_shape(id=num_vehicle, t=-1, traj_info=value_traj_segment)
            # 5) 以下读取背景车相关信息,车辆编号从1号开始,轨迹信息记录在vehicle_traj中
            self._scenario_info.add_vehicle_traj(id=num_vehicle, t=-1, traj_info=value_traj_segment)

        return self._scenario_info

    def _get_minesim_scenario_meta_data(self) -> MineSimScenarioTrackedMetaData:
        # 遍历 self._scenario_info.vehicle_traj 的每一个车辆轨迹
        dynamic_obstacle_vehicles = convert_to_MineSimScenarioVehicleTraj_batch(
            vehicle_traj=self._scenario_info.vehicle_traj,
            sim_time_origin=self._sim_time_origin,
            dt=self._scenario_info.test_setting["dt"],
        )

        # 创建 MineSimScenarioTrackedMetaData 对象
        self._scenario_file.dt = self._scenario_info.test_setting["dt"]
        self.minesim_tracked_metadata = MineSimScenarioTrackedMetaData(
            scenario_file_info=self._scenario_file,
            dynamic_obstacle_vehicles=dynamic_obstacle_vehicles,
        )

        return self.minesim_tracked_metadata

    def extract_tracked_objects_within_entrintime_window(self) -> List[DetectionsTracks]:
        self._detections_tracks_frames.clear()
        # 收集所有车辆的时间步列表
        self.metadata_time_s_list = [
            time_step for vehicle in self.minesim_tracked_metadata.dynamic_obstacle_vehicles for time_step in vehicle.vehicle_time_s_list
        ]
        self.metadata_time_str_list = [f"{time_s:.1f}" for time_s in self.metadata_time_s_list]

        max_track_time_s = max(self.metadata_time_s_list) if self.metadata_time_s_list else 0.0

        # 0.0 ,0.1, ... , 18.8(188)
        for i in range(int(0.0 * 10), int(max_track_time_s * 10) + 1, int(self.database_interval * 10)):
            timestep_s_str = f"{i/10:.1f}"
            tracked_objects = []  # tracked_objects: List[TrackedObject] = []

            for vehicle_agent in self.minesim_tracked_metadata.dynamic_obstacle_vehicles:
                if timestep_s_str in vehicle_agent.vehicle_time_s_str_list:
                    index = vehicle_agent.vehicle_time_s_str_list.index(timestep_s_str)
                    agent_state = vehicle_agent.vehicle_agent_states[index].agent_state
                    tracked_objects.append(
                        Agent(
                            tracked_object_type=agent_state.tracked_object_type,
                            oriented_box=agent_state.box,
                            velocity=agent_state.velocity,
                            predictions=[],  # todo. to be filled in later
                            angular_velocity=agent_state.angular_velocity,
                            acceleration=agent_state.acceleration,
                            metadata=agent_state.metadata,
                        )
                    )
                pass
            # 创建当前时间步的 DetectionsTracks 对象
            detections_tracks_frame = DetectionsTracks(iteration_step=i, tracked_objects=TrackedObjects(tracked_objects=tracked_objects))

            self._detections_tracks_frames.append(detections_tracks_frame)

        return self._detections_tracks_frames

    def calculate_square_region(self):
        """确定最大的方形区域."""
        x_coords = (
            [self._scenario_info.ego_info["x"]]
            + self._scenario_info.test_setting["goal"]["x"]
            + [self._scenario_info.test_setting["x_min"], self._scenario_info.test_setting["x_max"]]
        )
        y_coords = (
            [self._scenario_info.ego_info["y"]]
            + self._scenario_info.test_setting["goal"]["y"]
            + [self._scenario_info.test_setting["y_min"], self._scenario_info.test_setting["y_max"]]
        )

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        max_range = max(x_max - x_min, y_max - y_min)
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        self.utm_local_range = (x_center - max_range / 2, y_center - max_range / 2, x_center + max_range / 2, y_center + max_range / 2)

        return self.utm_local_range

    def get_tracked_objects_at_iteration(self, iteration: int) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"
        return self._detections_tracks_frames[iteration]

    # def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> Generator[TimePoint, None, None]:
    #     """Inherited, see superclass."""
    #     for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
    #         yield TimePoint(lidar_pc.timestamp)

    # def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> Generator[TimePoint, None, None]:
    #     """Inherited, see superclass."""
    #     for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
    #         yield TimePoint(lidar_pc.timestamp)

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> Generator[EgoState, None, None]:
        """
        Find ego past trajectory
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon [s]: the desired horizon to the future
        :return: the past ego trajectory with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        past_ego_states = []
        if iteration < num_samples:
            num_samples_cpoy = num_samples - iteration
            # !返回历史的数据,如果不够配置参数,进行复制第一帧;
            for index in range(0, num_samples_cpoy):
                past_ego_states.append(self.initial_ego_state)

            for index_detection in range(0, iteration):
                ValueError("todo")
        else:
            # toto
            ValueError("todo")
            pass

        return past_ego_states

        # pass
        # num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        # indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)

    #     return cast(
    #         Generator[EgoState, None, None],
    #         get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=False),
    #     )

    # def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> Generator[EgoState, None, None]:
    #     """Inherited, see superclass."""
    #     num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
    #     indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)

    #     return cast(
    #         Generator[EgoState, None, None],
    #         get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=True),
    #     )

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """
        Find past detections.
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param num_samples: number of entries in the future.
        :param time_horizon [s]: the desired horizon to the future.
        :param future_trajectory_sampling: sampling parameters of agent future ground truth predictions if desired.
        :ret
        """
        past_tracked_objects = []
        if iteration < num_samples:
            num_samples_cpoy = num_samples - iteration
            # !返回历史的数据,如果不够配置参数,进行复制第一帧;
            for index in range(0, num_samples_cpoy):
                past_tracked_objects.append(
                    DetectionsTracks(
                        is_real_data=False,
                        iteration_step=iteration,
                        tracked_objects=self._detections_tracks_frames[0].tracked_objects,
                    )
                )

            for index_detection in range(0, iteration):
                past_tracked_objects.append(self._detections_tracks_frames[index_detection])
        else:
            # toto
            ValueError("todo")
            pass

        return past_tracked_objects

    # def get_future_tracked_objects(
    #     self,
    #     iteration: int,
    #     time_horizon: float,
    #     num_samples: Optional[int] = None,
    #     future_trajectory_sampling: Optional[TrajectorySampling] = None,
    # ) -> Generator[DetectionsTracks, None, None]:
    #     """Inherited, see superclass."""
    #     # TODO: This can be made even more efficient with a batch query
    #     for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
    #         yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))
