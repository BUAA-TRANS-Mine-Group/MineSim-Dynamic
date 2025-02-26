import abc
import copy
from dataclasses import dataclass
import logging
import math
from typing import Dict, List, Optional, Tuple, Type


from shapely.geometry import Polygon
from shapely import affinity
import numpy as np


# Local library
# 由 onsite-mine 移植过来
from devkit.common.cost.cost_function import GoalSampledCostFunction
from devkit.common.geometry.cubic_spline import CubicSpline2D
from devkit.common.geometry.polynomial import QuarticPolynomial
from devkit.common.geometry.polynomial import QuinticPolynomial
from devkit.common.coordinate_system.frenet import FrenetState
from devkit.common.coordinate_system.frenet import State
from devkit.common.coordinate_system.frenet import GoalSampledFrenetTrajectory

from devkit.common.actor_state.commonroad_vehicle.vehicle import Vehicle
from devkit.common.actor_state.agent import Agent
from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.scene_object import SceneObject
from devkit.common.actor_state.state_representation import StateSE2
from devkit.common.actor_state.state_representation import StateVector2D
from devkit.common.actor_state.state_representation import TimePoint
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.common.geometry.transform import transform
from devkit.common.trajectory.interpolated_trajectory import InterpolatedTrajectory
from devkit.common.trajectory.abstract_trajectory import AbstractTrajectory
from devkit.metrics_tool.utils.expert_comparisons import principal_value

from devkit.sim_engine.environment_manager.collision_lookup import CollisionLookup
from devkit.sim_engine.environment_manager.collision_lookup import VehicleType
from devkit.sim_engine.observation_manager.observation_type import DetectionsTracks
from devkit.sim_engine.observation_manager.observation_type import Observation
from devkit.sim_engine.observation_manager.agent_update_policy.idm.utils import path_to_linestring
from devkit.sim_engine.observation_manager.agent_update_policy.idm.utils import create_path_from_se2

from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.sim_engine.planning.planner.abstract_planner import PlannerInitialization
from devkit.sim_engine.planning.planner.abstract_planner import PlannerInput
from devkit.sim_engine.planning.planner.route_planner.global_route_path_planner import GlobalRoutePathPlanner
from devkit.sim_engine.planning.planner.route_planner.global_route_path_planner import create_route_path_planner
from devkit.sim_engine.planning.planner.route_planner.utils.minesim_graph import create_minesim_graph
from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTasks
from devkit.sim_engine.simulation_time_controller.simulation_iteration import SimulationIteration

logger = logging.getLogger(__name__)


class Stats(object):
    """用于收集和处理与路径规划过程中的统计数据相关的信息.Statistical data;Statistics;
    这个类可以在自动驾驶或其他需要路径规划的应用中非常有用,用于评估规划算法的性能.
    """

    def __init__(self):
        """初始化统计数据,包括:
        迭代次数 (num_iter),生成的轨迹数 (num_trajs_generated),验证的轨迹数 (num_trajs_validated) 和碰撞检查次数 (num_collison_checks).
        """
        self.num_iter = 0
        self.num_trajs_generated = 0
        self.num_trajs_validated = 0
        self.num_collison_checks = 0
        # self.best_traj_costs = [] # float("inf")

    def __add__(self, other):
        """加法重载 (__add__):
            重载 + 操作符,允许将两个 Stats 对象的统计数据相加.
            当规划过程分为多个步骤或阶段时,这个方法可以用来累计整个过程的统计数据.
            加法操作将各个统计数据相加,并返回更新后的 Stats 对象.

        Args:
            other (Stats): 另一个Stats 对象.
        """
        self.num_iter += other.num_iter
        self.num_trajs_generated += other.num_trajs_generated
        self.num_trajs_validated += other.num_trajs_validated
        self.num_collison_checks += other.num_collison_checks
        # self.best_traj_costs.extend(other.best_traj_costs)
        return self

    def average(self, value: int):
        """用于计算统计数据的平均值.
            它将每个统计数值除以给定的 value 参数（例如,可能是迭代次数或试验次数）.
            这用于计算整个规划过程中每次迭代或每次试验的平均统计数据.

        Args:
            value (int): 迭代次数.


        """
        self.num_iter /= value
        self.num_trajs_generated /= value
        self.num_trajs_validated /= value
        self.num_collison_checks /= value
        return self


@dataclass
class FrenetOptimalPlannerSettings(abc.ABC):
    """Parameters related to the renetOptimalPlanner implementation."""

    num_width: int = (5,)  # road width方向的 sampling number
    num_speed: int = (5,)  # time sampling number 规划时间域的采样数量
    num_t: int = (5,)  # speed sampling number  速度采样数量

    max_road_width: Optional[float] = 1.4  # maximum road width [m]
    highest_speed: Optional[float] = 9.0  # highest sampling speed [m/s]
    lowest_speed: float = (0.0,)  # lowest sampling speed [m/s]
    min_planning_t: float = (5.0,)  # 终点状态采样,时间轴 min [s] ;最小规划8秒,最大规划10秒,因为预测轨迹给值？
    max_planning_t: float = (8.0,)  # 终点状态采样,时间轴 max [s]

    def __post_init__(self) -> None:
        assert self.num_width > 0, "num_width must be positive."
        assert self.num_speed > 0, "num_speed must be positive."
        assert self.num_t > 0, "num_t must be positive."
        assert (self.max_planning_t - self.min_planning_t) > 0, "num_t must be positive."
        # assert self.max_road_width > 0, "max_road_width must be positive."


class FrenetOptimalPlanner(AbstractPlanner):
    """
    Frenet Optimal Planner
    - Ref:
        - [1] Werling M, Ziegler J, Kammel S, et al. Optimal trajectory generation for dynamic street scenarios in
                a frenet frame[C]. IEEE, 2010: 987-993.
        - [2] Sun S, Chen J, Sun J, et al. FISS+: Efficient and focused trajectory generation and refinement using
            fast iterative search and sampling strategy[C]//2023 IEEE/RSJ International Conference on Intelligent
            Robots and Systems (IROS).
    - code: https://github.com/SS47816/fiss_plus_planner
    """

    # Inherited property, see superclass.
    requires_scenario: bool = True
    requires_init_ego_vehicle_parameters: bool = True

    def __init__(
        self,
        planner_settings: FrenetOptimalPlannerSettings,
        scenario: AbstractScenario,
        planned_trajectory_samples: int,
        planned_trajectory_sample_interval: float,
        truck_lateral_expansion_factor: float,
    ):
        self.settings = planner_settings
        self.scenario = scenario
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval
        self._planned_horizon = planned_trajectory_samples * planned_trajectory_sample_interval
        assert truck_lateral_expansion_factor > 0, "truck_lateral_expansion_factor 必须为正数"  # 参数校验
        self._truck_lateral_expansion_factor = truck_lateral_expansion_factor

        self.sample_time_min = self._planned_horizon
        self.sample_time_max = self._planned_horizon + self.settings.max_planning_t - self.settings.min_planning_t

        if self.requires_init_ego_vehicle_parameters:
            self.ego_vehicle: VehicleParameters = scenario.ego_vehicle_parameters
            self.settings.highest_speed = self.ego_vehicle.constraints["max_longitudinal_velocity"]

        if self.settings.max_road_width is None:
            max_road_width = self.ego_vehicle.width * 4.0
            self.settings.max_road_width = max_road_width
        else:
            max_road_width = self.settings.max_road_width

        self.ego_vehicle_commonroad: Vehicle = self._set_commonroad_vehicle_parameters(ego_vehicle=self.ego_vehicle)
        self.cost_function = GoalSampledCostFunction("WX1", vehicle_info=self.ego_vehicle_commonroad, max_road_width=max_road_width)
        self.collision_lookup: CollisionLookup = None  # 道路边界碰撞检测工具 TODO.refine strategy
        self._route_path_planner: GlobalRoutePathPlanner = None
        self.cubic_spline: CubicSpline2D = None  # 三次样条曲线用于记录参考路径;

        self.ego_state: State = None  # current_ego_state
        self.ego_state_nuplan: EgoState = None
        self.ego_frenet_state: FrenetState = FrenetState()
        self.now_t: float = None
        self.now_t_str: str = None
        self.now_t_step: int = None
        self.observation_detection: DetectionsTracks = None

        self.best_traj: GoalSampledFrenetTrajectory = None
        self.all_trajs_before_check: List[GoalSampledFrenetTrajectory] = []  # 记录最后保留下来的所有轨迹,碰撞检测前
        self.all_trajs: List[GoalSampledFrenetTrajectory] = [] # 记录最后保留下来的所有轨迹,碰撞检测后

        self.dt = scenario.database_interval
        self.current_iteration: SimulationIteration = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        # ========== search ego_route_path ==========
        self.scenario_name = initialization.scenario_name
        self._map_api = initialization.map_api
        self._initialize_search_ego_route_path(
            initial_ego_state=initialization.initial_ego_state, goal_task=initialization.planning_problem_goal_task
        )
        logger.info("#log# Initialized a FrenetOptimalPlanner: initialize ego_route_planner and search ego_route_path.")

        # ========== create collision_lookup ==========
        if self.collision_lookup is None:
            if self.scenario._ego_vehicle_parameters.vehicle_name == "XG90G":
                vehicle_type = VehicleType.MineTruck_XG90G
            elif self.scenario._ego_vehicle_parameters.vehicle_name == "NTE200":
                vehicle_type = VehicleType.MineTruck_NTE200
            else:
                ValueError("vehicle_name")
                logging.error(f"vehicle_name={self.scenario._ego_vehicle_parameters.vehicle_name} is error!")
            self.collision_lookup = CollisionLookup(type=vehicle_type)

        # ========== create other FOP planner initialization ==========
        # self.ego_state = self._get_ego_states(ego_state_nuplan=self.scenario.initial_ego_state)

        self._initialized = True

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass. - method core"""
        self.ego_state_nuplan, self.observation_detection = current_input.history.current_state

        # ========== Ego current state ==========
        self.ego_state = self._get_ego_states(ego_state_nuplan=self.ego_state_nuplan)
        self.ego_frenet_state.update_frenetstate_from_state(state=self.ego_state, polyline=self._route_path_planner.refline_smooth)

        # ========== _update_now_time ==========
        self.current_iteration = current_input.iteration
        self.now_t = self.dt * current_input.iteration.index
        self.now_t_str = f"{self.now_t:.1f}"
        self.now_t_step = current_input.iteration.index

        # ========== plan traj using sampling method (FOP) ==========
        self.plan_trajectory(frenet_state=self.ego_frenet_state, observation_detection=self.observation_detection)

        return self._get_planned_trajectory(ego_state_nuplan=self.ego_state_nuplan, best_traj=self.best_traj)

    def _get_planned_trajectory(self, ego_state_nuplan: EgoState, best_traj: GoalSampledFrenetTrajectory) -> InterpolatedTrajectory:
        """将FOP轨迹结果 重新采样为 nuplan框架结果"""
        vehicle_parameters = ego_state_nuplan.car_footprint.vehicle_parameters

        # Initialize planned trajectory with current state
        current_time_point = ego_state_nuplan.time_point
        planned_trajectory: List[EgoState] = []

        # 生成时间索引列表
        time_indices = [i * self._planned_trajectory_sample_interval for i in range(self._planned_trajectory_samples)]
        # 生成整数时间索引列表
        int_time_indices = [int(i * self._planned_trajectory_sample_interval * 10) for i in range(self._planned_trajectory_samples)]
        int_time_indices.append(79)
        # Integer Time Indices: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]

        # Propagate planned trajectory for set number of samples
        for index in int_time_indices:
            # Propagate IDM state w.r.t. selected leading agent time_us=self._sim_time_origin.sim_time_point_origin.time_us + int(iteration * 1e5)
            ego_state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x=self.best_traj.x[index], y=self.best_traj.y[index], heading=self.best_traj.yaw[index]),
                rear_axle_velocity_2d=StateVector2D(x=self.best_traj.s_d[index], y=self.best_traj.d_d[index]),
                rear_axle_acceleration_2d=StateVector2D(x=self.best_traj.s_dd[index], y=self.best_traj.d_dd[index]),
                tire_steering_angle=0.0,
                time_point=TimePoint(time_us=current_time_point.time_us + int(index * 1e5)),
                vehicle_parameters=self.ego_vehicle,
                is_in_auto_mode=True,
                angular_vel=0.0,
                angular_accel=0.0,
            )
            planned_trajectory.append(ego_state)

        return InterpolatedTrajectory(planned_trajectory)

    def _set_commonroad_vehicle_parameters(self, ego_vehicle: VehicleParameters) -> Vehicle:
        if ego_vehicle.vehicle_name == "XG90G":
            longitudinal_jerk_max = 10.0
            lateral_accel_max = 8.0
        elif ego_vehicle.vehicle_name == "NTE200":
            longitudinal_jerk_max = 8.8
            lateral_accel_max = 6.8

        return Vehicle(
            vehicle_l=ego_vehicle.length,
            vehicle_w=ego_vehicle.width,
            vehicle_h=ego_vehicle.height,
            safety_factor=1.2,
            longitudinal_v_max=ego_vehicle.constraints["max_longitudinal_velocity"],
            longitudinal_a_max=ego_vehicle.constraints["max_longitudinal_acceleration"],
            longitudinal_jerk_max=longitudinal_jerk_max,
            lateral_accel_max=ego_vehicle.constraints["max_lateral_acceleration"],
            lateral_jerk_max=lateral_accel_max,
            centripetal_accel_max=ego_vehicle.constraints["max_centripetal_acceleration"],
            shape=ego_vehicle.shape,
            constraints=ego_vehicle.constraints,
        )

    def _initialize_search_ego_route_path(self, initial_ego_state: EgoState, goal_task: PlanningProblemGoalTasks):
        """Initializes the ego route path from mine map;"""
        # ========== search route path ==========
        assert self._map_api, "_map_api has not yet been initialized. Please call the initialize() function first!"
        # minesim_graph = create_minesim_graph(map_name=self._map_api.map_name, semantic_map=self._map_api.semantic_map)
        if self._route_path_planner is None:
            self._route_path_planner = create_route_path_planner(
                map_name=self._map_api.map_name,
                semantic_map=self._map_api.semantic_map,
                minesim_graph=create_minesim_graph(map_name=self._map_api.map_name, semantic_map=self._map_api.semantic_map),
                is_initial_path_stitched=True,
            )

        # 搜索路径
        self._route_path_planner.get_ego_global_route_path(
            start_pose_statese2=StateSE2(
                x=initial_ego_state.rear_axle.x,
                y=initial_ego_state.rear_axle.y,
                heading=initial_ego_state.rear_axle.heading,
            ),
            goal_task=goal_task,
            scenario_name=self.scenario_name,
        )
        # 检查 refline_smooth 是否存在并且维度正确
        if self._route_path_planner.refline_smooth is None or self._route_path_planner.refline_smooth.shape[1] < 3:
            logging.error("#log# refline_smooth is not properly generated or lacks required columns.")
            raise ValueError("Error: refline_smooth data is missing or incomplete.")

        # ========== nuplan 框架的 参考路径表示 ==========
        # 使用 refline_smooth 生成离散路径
        discrete_path = [StateSE2(x=waypoint[0], y=waypoint[1], heading=waypoint[2]) for waypoint in self._route_path_planner.refline_smooth]
        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)

        # ========== FOP | FISS++ 算法 参考路径表示：cubic_spline， frenet，三次样条曲线插值 ==========
        self.cubic_spline = self._route_path_planner.cubic_spline

    def _get_ego_states(self, ego_state_nuplan: EgoState) -> State:
        """transform EgoState to State"""
        return State(
            t=None,
            x=ego_state_nuplan.rear_axle.x,
            y=ego_state_nuplan.rear_axle.y,
            yaw=ego_state_nuplan.rear_axle.heading,
            v=ego_state_nuplan.dynamic_car_state.rear_axle_velocity_2d.x,
            a=ego_state_nuplan.dynamic_car_state.rear_axle_acceleration_2d,
        )

    def _get_ego_frenet_state(self, ego_states: State = None) -> FrenetState:
        self.ego_frenet_state.from_state(ego_states, self.ref_ego_lane_pts)
        return self.ego_frenet_state

    def sampling_frenet_trajectories(self, frenet_state: FrenetState) -> list:
        """横向,纵向,规划时长;采样的核心函数.

        Args:
            frenet_state (FrenetState): 自车当前的frene状态

        Returns:
            list: 采样完成的所有轨迹;
        """
        frenet_trajectories = []
        traj_per_timestep = []

        sampling_width = self.settings.max_road_width - self.ego_vehicle.width
        #! lateral sampling
        for di in np.linspace(-sampling_width / 2, sampling_width / 2, self.settings.num_width):

            #! time sampling
            for Ti in np.linspace(self.sample_time_min, self.sample_time_max, self.settings.num_t):
                aa = abs(Ti % 0.1)
                if aa > 1e-3 and aa < 0.09:  # epsilon = 1e-10  # 设置一个很小的容差值
                    raise Exception("##log##:error.")
                ft = GoalSampledFrenetTrajectory()
                # xs, vxs, axs, xe, vxe, axe, time):
                lat_qp = QuinticPolynomial(xs=frenet_state.d, vxs=frenet_state.d_d, axs=frenet_state.d_dd, xe=di, vxe=0.0, axe=0.0, time=Ti)
                ft.t = [t for t in np.arange(0.0, Ti, self.dt)]
                ft.d = [lat_qp.calc_point(t) for t in ft.t]
                ft.d_d = [lat_qp.calc_first_derivative(t) for t in ft.t]
                ft.d_dd = [lat_qp.calc_second_derivative(t) for t in ft.t]
                ft.d_ddd = [lat_qp.calc_third_derivative(t) for t in ft.t]

                #! longitudinal sampling
                for tv in np.linspace(self.settings.lowest_speed, self.settings.highest_speed, self.settings.num_speed):
                    tft = copy.deepcopy(ft)

                    lon_qp = QuarticPolynomial(xs=frenet_state.s, vxs=frenet_state.s_d, axs=frenet_state.s_dd, vxe=tv, axe=0.0, time=Ti)
                    tft.s = [lon_qp.calc_point(t) for t in ft.t]
                    tft.s_d = [lon_qp.calc_first_derivative(t) for t in ft.t]
                    tft.s_dd = [lon_qp.calc_second_derivative(t) for t in ft.t]
                    tft.s_ddd = [lon_qp.calc_third_derivative(t) for t in ft.t]

                    # Compute the final cost
                    tft.cost_total = self.cost_function.cost_total(
                        traj=tft, target_speed=self.settings.highest_speed, t_max=self.sample_time_max, t_min=self.sample_time_min
                    )
                    frenet_trajectories.append(tft)
                    traj_per_timestep.append(tft)
        self.all_trajs_before_check.append(traj_per_timestep)

        return frenet_trajectories

    def calc_global_paths(self, ftlist: list[GoalSampledFrenetTrajectory]) -> list:
        """将一系列 GoalSampledFrenetTrajectory 对象（代表在 Frenet 坐标系中的路径）转换成全局坐标系中的路径.
        具体来说,它会计算每个 GoalSampledFrenetTrajectory 上的全局 x, y 坐标,偏航角,曲率等参数.

        Args:
            ftlist (list[GoalSampledFrenetTrajectory]): GoalSampledFrenetTrajectory 的列表.其实每次只传入一个值

        Returns:
            list:ftlist, 返回本身的地址, 将一个列表作为参数传递给函数时,您实际上是传递了这个列表的引用（或地址）
        """
        passed_ftlist = []
        for ft in ftlist:
            # Calculate global positions for each point in the  Frenet trajectory ft
            for i in range(len(ft.s)):
                ix, iy = self.cubic_spline.calc_position(ft.s[i])  # 沿参考曲线的距离 s 计算曲线上的 (x, y) 坐标
                if ix is None:  # !超过了参考路径终点s距离
                    break
                # Calculate global yaw (orientation) from Frenet s coordinate
                i_yaw = self.cubic_spline.calc_yaw(ft.s[i])  # 沿参考曲线的距离 s 计算曲线上的 yaw
                di = ft.d[i]
                # Calculate the final x, y coordinates considering lateral offset (d)
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                ft.x.append(fx)
                ft.y.append(fy)

            if len(ft.x) >= 2:
                #  Convert lists to numpy arrays for easier mathematical operations
                # calc yaw and ds
                ft.x = np.array(ft.x)
                ft.y = np.array(ft.y)
                # # Calculate differences between consecutive x and y coordinates
                x_d = np.diff(ft.x)
                y_d = np.diff(ft.y)
                # Calculate yaw angles and their differences
                # ! 涉及角度的计算都考虑了角度的周期性:使用 np.arctan2(y_d, x_d) 函数计算偏航角时,结果的范围[-PI,PI),使用弧度作为单位.
                ft.yaw = np.arctan2(y_d, x_d)
                ft.ds = np.hypot(x_d, y_d)
                ft.yaw = np.append(ft.yaw, ft.yaw[-1])
                # Calculate curvature (c), its first derivative (c_d), and second derivative (c_dd)
                dt = self.dt
                # ! 计算曲率
                # 通过偏航角的差分除以路径长度的差分计算曲率.
                # 这里使用 np.diff 函数来计算偏航角 yaw 的差分,即每段路径的偏航角变化量,然后除以相应的路径长度变化量 ft.ds.
                ft.c = np.divide(np.diff(ft.yaw), ft.ds)  # ! 确保 yaw 为-pi 到 pi.
                ft.c_d = np.divide(np.diff(ft.c), dt)
                ft.c_dd = np.divide(np.diff(ft.c_d), dt)

                # Append the trajectory with calculated global parameters to the list
                passed_ftlist.append(ft)
        return ftlist

    def check_constraints(self, trajs: list[GoalSampledFrenetTrajectory]) -> list:
        """对轨迹进行约束检查.

        1）Max curvature check.
        2）最大速度检查（Max Speed Check）：确保轨迹上的速度在任何点上都不超过车辆的最大速度限制.
            这是通过检查轨迹中的每个点上的纵向速度 s_d 是否超过 self.ego_vehicle_commonroad.max_speed 来完成的.
        3）最大加速度检查（Max Acceleration Check）：确保轨迹上的加速度不超过车辆的最大加速度限制.
            这是通过检查轨迹中每个点上的纵向加速度 s_dd 是否超过 self.ego_vehicle_commonroad.max_accel 来完成的.

        Args:
            trajs (list[GoalSampledFrenetTrajectory]): GoalSampledFrenetTrajectory 的列表.其实每次只传入一个值

        Returns:
            list: 函数返回一个通过所有检查的轨迹列表.
            这是通过在 passed 列表中记录符合要求的轨迹的索引,然后从原始 trajs 列表中提取这些轨迹来实现的.
            如果轨迹不满足某项检查,它将被忽略,不会被添加到返回的列表中.
        """
        passed = []
        for i, traj in enumerate(trajs):
            # 1）Max curvature check
            # if any([abs(c) > self.ego_vehicle_commonroad.max_curvature for c in traj.c]):
            #     continue
            # if any([abs(c_d) > self.ego_vehicle_commonroad.max_kappa_d for c_d in traj.c_d]):
            #     continue
            # if any([abs(c_dd) > self.ego_vehicle_commonroad.max_kappa_dd for c_dd in traj.c_dd]):
            #     continue
            # 2）Max speed check,因为是在速度空间采样,所以不会超速
            # if any([v > self.ego_vehicle_commonroad.max_speed for v in traj.s_d]):
            #     continue
            # 3）Max accel check
            if any([abs(a) > self.ego_vehicle_commonroad.max_accel for a in traj.s_dd]):
                continue
            # 4）max jerk check;
            if any([abs(jerk) > self.ego_vehicle_commonroad.max_jerk for jerk in traj.s_ddd]):
                continue
            # 5) max_centripetal_accel check #!有bug,将所有的转弯速度快的都剪掉了
            # if any([abs(curvature * v**2) > self.ego_vehicle_commonroad.max_centripetal_accel for curvature, v in zip(traj.c, traj.s_d)]):
            #     continue

            # todo
            traj.constraint_passed = True
            passed.append(i)  # 通过所有检查的轨迹 保存其索引
        return [trajs[i] for i in passed]

    def construct_polygon(self, polygon: Polygon, x: float, y: float, yaw: float) -> Polygon:
        """根据给定的位置和方向参数来构造多边形.
        位置和方向参数:笛卡尔坐标系下的(x,y,yaw)

        Args:
            polygon (Polygon): shapely.geometry.polygon.Polygon ,
            x (float): 位置和方向参数 ,
            y (float): 位置和方向参数
            yaw (float): 位置和方向参数,偏航角,rad  use_radians.
            # ! yaw在旋转函数中一般取[-pi, pi]范围：这种取值范围可以确保旋转的连续性,因为yaw = -pi和yaw = pi被认为是等效的（相差360度）.
            # 注：[-pi, pi] ,[0, 2*pi],[-8*pi, 8*pi]范围均可, affinity 旋转函数均可以正确处理

        Returns:
            Polygon: 平移旋转后的多边形  shapely.geometry.polygon.Polygon
        """
        polygon_translated = affinity.translate(geom=polygon, xoff=x, yoff=y)
        polygon_rotated = affinity.rotate(geom=polygon_translated, angle=yaw, use_radians=True)

        return polygon_rotated

    def get_vehicle_polygon(self, l, w):
        # footprint coordinates (in clockwise direction)  足迹坐标(顺时针方向)
        corners: list[tuple[float, float]] = [
            (l / 2, w / 2),  # front left corner's coordinates in box_center frame [m]
            (l / 2, -w / 2),  # front right corner's coordinates in box_center frame [m]
            (-l / 2, -w / 2),  # rear right corner's coordinates in box_center frame [m]
            (-l / 2, w / 2),  # rear left corner's coordinates in box_center frame [m]
            (l / 2, w / 2),  # front left corner's coordinates in box_center frame [m] (to enclose the polygon)
        ]
        vehicle_polygon: Polygon = Polygon(corners)

        return vehicle_polygon

    def has_collision(self, traj: GoalSampledFrenetTrajectory, observation_detection: DetectionsTracks, check_resolution: int = 2) -> tuple:
        """检查给定轨迹是否与任何障碍物发生碰撞

        Args:
            traj (GoalSampledFrenetTrajectory): 单条轨迹
            obstacles (list): 障碍物列表
            check_resolution (int, optional): 检查分辨率. Defaults to 1. 每几步检测一次,每0.1s一次有点频繁,所以可以设置 2 step.

        Returns:
            tuple: 返回一个布尔值,指示是否发生碰撞,以及检查了多少个多边形.
        """

        num_polys = 0
        if len(observation_detection.tracked_objects.tracked_objects) <= 0:
            return False, 0  # 没有障碍物
        obstacles: Dict = self.scenario.scenario_info.vehicle_traj

        # !only for replay test
        final_time_step = int(self.scenario.scenario_info.test_setting["max_t"] / self.scenario.database_interval)  # 100 step = 10.0s
        t_step_max = min(len(traj.x), final_time_step - self.now_t_step)  # 规划时间步 ,该场景的最大时间步, 取最小 self.now_t_step
        for i in range(t_step_max):
            if i % check_resolution == 0:
                # construct a polygon for the ego ego_vehicle at time step i   # 在第 i 步构造自车的多边形
                try:
                    ego_polygon = self.construct_polygon(polygon=self.ego_vehicle_commonroad.polygon, x=traj.x[i], y=traj.y[i], yaw=traj.yaw[i])
                except:
                    print(f"Failed to create Polygon for t={i} x={traj.x[i]}, y={traj.y[i]}, yaw={traj.y[i]}")
                    return True, num_polys  # 创建多边形失败
                else:
                    # construct a polygon for the obstacle at time step i   # 在第 i 步构造每个障碍物的多边形
                    time_s = (i + self.now_t_step) * self.dt
                    str_time = f"{time_s:.1f}"
                    for key, obstacle in obstacles.items():
                        if str_time in obstacle.keys():
                            state = obstacle[str_time]  # state['y']
                            box_polygon = self.get_vehicle_polygon(l=obstacle["shape"]["length"], w=obstacle["shape"]["width"])
                            obstacle_polygon = self.construct_polygon(polygon=box_polygon, x=state["x"], y=state["y"], yaw=state["yaw_rad"])
                            num_polys += 1
                            if ego_polygon.intersects(
                                obstacle_polygon
                            ):  # Returns True if geometries intersect, else FalsePython library,两个几何相交判断
                                # plot_collision(ego_polygon, obstacle_polygon, t_step)
                                return True, num_polys  # 如果发生碰撞

        return False, num_polys

    def check_collisions(self, trajs: list, observation_detection: DetectionsTracks) -> list[GoalSampledFrenetTrajectory]:
        """接收一系列轨迹,障碍物列表和当前时间步,返回经过碰撞检测的轨迹列表.
        Returns:
            list: 返回 通过了碰撞检测的轨迹列表
        """
        passed = []

        for i, traj in enumerate(trajs):
            # Collision check  # 检查轨迹和障碍物之间是否发生碰撞
            collision, num_polys = self.has_collision(traj=traj, observation_detection=observation_detection, check_resolution=2)
            if collision:
                continue

            # 检查轨迹合道路边界之间是否发生碰撞
            collision = self.has_collision_with_boundary(traj=traj, check_resolution=20)
            if collision:
                continue

            traj.collision_passed = True
            passed.append(i)

        return [trajs[i] for i in passed]

    def plan_trajectory(self, frenet_state: FrenetState, observation_detection: DetectionsTracks) -> GoalSampledFrenetTrajectory:
        """frenet 规划方法的核心函数.

        Args:
            frenet_state (FrenetState): 自车当前的frene状态
            obstacles (list): 障碍物.
            observation (_type_): 场景的一些状态数据.

        Returns:
            GoalSampledFrenetTrajectory: 规划结果.
        """
        ftlist = self.sampling_frenet_trajectories(frenet_state=frenet_state)  # frenet path list
        ftlist = self.calc_global_paths(ftlist)
        ftlist = self.check_constraints(ftlist)
        # print(f"##log## {len(ftlist)} trajectories passed constraint check")

        ftlist = self.check_collisions(trajs=ftlist, observation_detection=observation_detection)
        # print(f"##log## {len(ftlist)} trajectories passed collision check")
        self.all_trajs.append(ftlist)

        # find minimum cost path
        min_cost = float("inf")
        for ft in ftlist:
            if min_cost >= ft.cost_total:
                min_cost = ft.cost_total
                self.best_traj = ft

        return self.best_traj

    def has_collision_with_boundary(self, traj: GoalSampledFrenetTrajectory, check_resolution: int = 10) -> bool:
        """检查给定轨迹是否与道路边界发生碰撞

        Args:
            traj (GoalSampledFrenetTrajectory): 单条轨迹
            obstacles (list): 障碍物列表
            time_step_now (int, optional): 当前时间步. Defaults to 0.
            check_resolution (int, optional): 检查分辨率. Defaults to 1. 每几步检测一次,每0.1s一次有点频繁,所以可以设置 2 step.

        Returns:
            tuple: 返回一个布尔值,指示是否发生碰撞,以及检查了多少个多边形.
        """
        final_time_step = int(self.scenario.scenario_info.test_setting["max_t"] / self.scenario.database_interval)  # 100 step = 10.0s
        t_step_max = min(len(traj.x), final_time_step - self.now_t_step)  # 规划时间步 ,该场景的最大时间步, 取最小

        local_x_range = self.scenario.map_api.raster_bit_map.bitmap_info["bitmap_mask_PNG"]["UTM_info"]["local_x_range"]
        local_y_range = self.scenario.map_api.raster_bit_map.bitmap_info["bitmap_mask_PNG"]["UTM_info"]["local_y_range"]

        for i in range(t_step_max):

            if i != t_step_max - 1 and i % check_resolution != 0:
                continue

            if self.collision_lookup.collision_detection(
                x=traj.x[i] - local_x_range[0],
                y=traj.y[i] - local_y_range[0],
                yaw=traj.yaw[i],
                image=self.scenario.map_api.raster_bit_map.image_ndarray,
            ):
                return True

        return False
