from collections import deque
from typing import Dict, List, Optional, Tuple
from typing import Any, Tuple, Type
import sys

sys.path.append("/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/MineSim-Dynamic-Dev")
from devkit.common.actor_state.state_representation import StateSE2, StateVector2D
from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.state_representation import TimePoint

from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphEdgeRefPathMapObject
from devkit.sim_engine.map_manager.minesim_map.minesim_graph_search_edge_node import GraphTopologyNodeMapObject
from devkit.sim_engine.map_manager.minesim_map.minesim_map_factory import get_maps_api

from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTasks
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTask
from devkit.scenario_builder.minesim_scenario_json.minesim_planning_problem_data_type import PlanningProblemGoalTaskFinalPose
from devkit.common.actor_state.vehicle_parameters import get_mine_truck_parameters
from devkit.sim_engine.planning.planner.route_planner.utils.breadth_first_search import BreadthFirstSearch
from devkit.sim_engine.planning.planner.route_planner.utils.graph_search_utils import calculate_start_refpath_info
from devkit.sim_engine.planning.planner.route_planner.utils.graph_search_utils import calculate_end_refpath_info
from devkit.sim_engine.map_manager.minesim_map.minesim_map_factory import get_graph_edges_and_nodes
from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader
from devkit.sim_engine.planning.planner.route_planner.utils.minesim_graph import MineSimGraph
from devkit.sim_engine.planning.planner.route_planner.utils.minesim_graph import create_minesim_graph

# ============================== 测试代码 ==============================
map_root = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/maps"
map_name = "guangdong_dapai"

mine_map_api = get_maps_api(map_root=map_root, map_name=map_name)

from devkit.common.actor_state.tracked_objects_types import TrackedObjectType


name = TrackedObjectType.BICYCLE
# scenario.initial_ego_stat
# runners[0].scenario._scenario_info.ego_info=
# {'shape': {'vehicle_type': 'MineTruck_XG90G', 'length': 9, 'width': 4, 'height': 3.5, 'min_turn_radius': 12, 'wheel_base': 5.15, 'locationPoint2Head': 6.5, 'locationPoint2Rear': 2.5},
# 'x': 1570.55, 'y': 701.404, 'v_mps': 4.5, 'acc_mpss': 0, 'yaw_rad': 6.01, 'yawrate_radps': 0}
initial_ego_state = EgoState.build_from_rear_axle(
    rear_axle_pose=StateSE2(x=1570.55, y=701.404, heading=6.01),
    rear_axle_velocity_2d=StateVector2D(x=4.5, y=0.0),
    rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
    tire_steering_angle=0.0,
    time_point=TimePoint(time_us=0),
    vehicle_parameters=get_mine_truck_parameters(mine_name="guangdong_dapai"),
    is_in_auto_mode=True,
    angular_vel=0.0,
    angular_accel=0.0,
    tire_steering_rate=0.0,
)

# scenario._scenario_info.test_setting['goal'] = {'x': [1688.06, 1686.56, 1697.06, 1700.05], 'y': [654.902, 633.905, 631.906, 655.402]}
# goal_task=PlanningProblemGoalTaskFinalPose(scenario_name="test-1",goal_range_polygon=1,final_pose=StateSE2(x=,y=,heading=))
goal_task = PlanningProblemGoalTask(
    scenario_name="test-1", goal_range_xs=[1688.06, 1686.56, 1697.06, 1700.05], goal_range_ys=[654.902, 633.905, 631.906, 655.402]
)
a = 1


minesim_graph = create_minesim_graph(map_name=map_name, semantic_map=mine_map_api.semantic_map)
a = 1
