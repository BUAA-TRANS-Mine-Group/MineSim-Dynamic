from __future__ import annotations

from collections import defaultdict
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Union
from bokeh.models import ColumnDataSource
import numpy as np

from devkit.common.actor_state.tracked_objects_types import tracked_object_types
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.sim_engine.history.simulation_history import SimulationHistory
from devkit.sim_engine.observation_manager.observation_type import Observation
from devkit.sim_engine.observation_manager.observation_type import DetectionsTracks


@dataclass
class SimulationIterationTime:
    # 对应 Simulation Iteration Time ,从0开始,0.1时间间隔; Serial
    t_index: int
    t_str: str
    t: float


@dataclass
class SimulationIterationTimeSequence:
    sim_step_num: int
    t_index_s: List[int] = field(default_factory=list)  # 使用 default_factory 初始化空列表
    t_str_s: List[str] = field(default_factory=list)  # 使用 default_factory 初始化空列表
    t_s: List[float] = field(default_factory=list)  # 使用 default_factory 初始化空列表
    sim_iteration_s: List[SimulationIterationTime] = field(default_factory=list)  # 使用 default_factory 初始化空列表
    dt: float = 0.1

    def __post_init__(self):
        """
        Python dataclass初始化完成后，会自动执行__post_init__。
        在这里调用 _get_time_list，以便在实例化时自动生成相关时间序列数据。
        """
        self._get_time_list()

    def _get_time_list(self):
        for index in range(self.sim_step_num):
            now_t = index * self.dt
            now_t_str = str(round(now_t, 1))

            # 创建 SimulationIterationTime 实例并添加到 sim_iteration_s 列表
            self.sim_iteration_s.append(SimulationIterationTime(t_index=index, t_str=now_t_str, t=now_t))
            self.t_index_s.append(index)  # 添加索引到 t_index_s 列表
            self.t_str_s.append(now_t_str)  # 添加时间字符串到 t_str_s 列表
            self.t_s.append(now_t)  # 添加时间值到 t_s 列表


@dataclass
class EgoStatePlot:
    """A dataclass for ego state plot."""

    vehicle_parameters: VehicleParameters  # Ego vehicle parameters

    def update_plot(self, radius: float, frame_index: int) -> None:
        """
        Update the plot.
        :param radius: Figure radius.
        :param frame_index: Frame index.
        """
        while self.data_sources.get(frame_index, None) is None:
            self.condition.wait()

        data_sources = dict(self.data_sources[frame_index].data)
        center_x = data_sources["center_x"][0]
        center_y = data_sources["center_y"][0]

        if self.plot is None:
            self.plot = main_figure.multi_polygons(
                xs="xs",
                ys="ys",
                fill_color=simulation_tile_agent_style["ego"]["fill_color"],
                fill_alpha=simulation_tile_agent_style["ego"]["fill_alpha"],
                line_color=simulation_tile_agent_style["ego"]["line_color"],
                line_width=simulation_tile_agent_style["ego"]["line_width"],
                source=data_sources,
            )
        else:
            self.plot.data_source.data = data_sources

        if self.init_state:
            main_figure.x_range.start = center_x - radius / 2
            main_figure.x_range.end = center_x + radius / 2
            main_figure.y_range.start = center_y - radius / 2
            main_figure.y_range.end = center_y + radius / 2
            self.init_state = False
        else:
            x_radius = main_figure.x_range.end - main_figure.x_range.start
            y_radius = main_figure.y_range.end - main_figure.y_range.start
            main_figure.x_range.start = center_x - x_radius / 2
            main_figure.x_range.end = center_x + x_radius / 2
            main_figure.y_range.start = center_y - y_radius / 2
            main_figure.y_range.end = center_y + y_radius / 2

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update ego_pose state data sources.
        :param history: SimulationHistory time-series data.
        """
        for frame_index, sample in enumerate(history.data):
            ego_pose = sample.ego_state.car_footprint
            dynamic_car_state = sample.ego_state.dynamic_car_state
            ego_corners = ego_pose.all_corners()

            corner_xs = [corner.x for corner in ego_corners]
            corner_ys = [corner.y for corner in ego_corners]

            # Connect to the first point
            corner_xs.append(corner_xs[0])
            corner_ys.append(corner_ys[0])
            source = ColumnDataSource(
                dict(
                    center_x=[ego_pose.center.x],
                    center_y=[ego_pose.center.y],
                    velocity_x=[dynamic_car_state.rear_axle_velocity_2d.x],
                    velocity_y=[dynamic_car_state.rear_axle_velocity_2d.y],
                    speed=[dynamic_car_state.speed],
                    acceleration_x=[dynamic_car_state.rear_axle_acceleration_2d.x],
                    acceleration_y=[dynamic_car_state.rear_axle_acceleration_2d.y],
                    acceleration=[dynamic_car_state.acceleration],
                    heading=[ego_pose.center.heading],
                    steering_angle=[sample.ego_state.tire_steering_angle],
                    yaw_rate=[sample.ego_state.dynamic_car_state.angular_velocity],
                    xs=[[[corner_xs]]],
                    ys=[[[corner_ys]]],
                )
            )
            self.data_sources[frame_index] = source


# @dataclass
# class EgoStateTrajectoryPlot:
#     """A dataclass for ego state trajectory plot."""

#     data_sources: Dict[int, ColumnDataSource] = field(default_factory=dict)  # A dict of data sources for each frame
#     condition: Optional[threading.Condition] = None  # Threading condition

#     def __post_init__(self) -> None:
#         """Initialize threading condition."""
#         if not self.condition:
#             self.condition = threading.Condition(threading.Lock())

#     def update_plot(self, main_figure: Figure, frame_index: int) -> None:
#         """
#         Update the plot.
#         :param main_figure: The plotting figure.
#         :param frame_index: Frame index.
#         """
#         if not self.condition:
#             return

#         with self.condition:
#             while self.data_sources.get(frame_index, None) is None:
#                 self.condition.wait()

#             data_sources = dict(self.data_sources[frame_index].data)
#             if self.plot is None:
#                 self.plot = main_figure.line(
#                     x="xs",
#                     y="ys",
#                     line_color=simulation_tile_trajectory_style["ego"]["line_color"],
#                     line_width=simulation_tile_trajectory_style["ego"]["line_width"],
#                     line_alpha=simulation_tile_trajectory_style["ego"]["line_alpha"],
#                     source=data_sources,
#                 )
#             else:
#                 self.plot.data_source.data = data_sources

#     def update_data_sources(self, history: SimulationHistory) -> None:
#         """
#         Update ego_pose trajectory data sources.
#         :param history: SimulationHistory time-series data.
#         """
#         if not self.condition:
#             return

#         with self.condition:
#             for frame_index, sample in enumerate(history.data):
#                 trajectory = sample.trajectory.get_sampled_trajectory()

#                 x_coords = []
#                 y_coords = []
#                 for state in trajectory:
#                     x_coords.append(state.center.x)
#                     y_coords.append(state.center.y)

#                 source = ColumnDataSource(dict(xs=x_coords, ys=y_coords))
#                 self.data_sources[frame_index] = source
#                 self.condition.notify()


@dataclass
class AgentStatePlot:
    """A dataclass for agent state plot."""

    data_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # A dict of data for all agents Dict[trackid, Dict[time_str, Any]]
    sim_times: SimulationIterationTimeSequence = None
    times_link_track_ids_minesim: List[List[str]] = field(default_factory=list)  # track_ids for each frame

    def __post_init__(self):
        # 如果没有传 sim_times，强制报错
        if self.sim_times is None:
            raise ValueError("sim_times is necessary!")

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update agents data sources.
        :param history: SimulationHistory time-series data.
        """
        # 每次更新前先清空
        self.times_link_track_ids_minesim.clear()
        # 用 defaultdict(dict) 简化对多级 dict 的初始化
        frame_dict = defaultdict(dict)

        for frame_index, sample in enumerate(history.data):
            # 如果观测数据不是 DetectionsTracks，跳过
            if not isinstance(sample.observation, DetectionsTracks):
                continue

            track_ids_minesim = []
            for agent in sample.observation.tracked_objects.tracked_objects:
                track_id_str = str(agent.metadata.track_id_minesim)
                # 确保 sim_times 已初始化并且 frame_index 有效; 如果超出 sim_times 范围，就设为 'unknown'
                if self.sim_times and frame_index < len(self.sim_times.t_str_s):
                    t_str = self.sim_times.t_str_s[frame_index]
                else:
                    t_str = "unknown"
                # 若相同 (track_id_str, t_str) 不存在，才写入
                frame_dict[track_id_str].setdefault(t_str, agent)
                track_ids_minesim.append(track_id_str)

            self.times_link_track_ids_minesim.append(track_ids_minesim)

        # 最终写回 data_sources
        self.data_sources = {k: dict(v) for k, v in frame_dict.items()}

    def update_now_index(self, now_index: int):
        self.now_index = now_index
