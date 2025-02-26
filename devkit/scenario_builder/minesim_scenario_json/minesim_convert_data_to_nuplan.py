from functools import lru_cache
from math import cos, sin
from typing import Optional, Set, Dict, List

import numpy as np

from devkit.common.actor_state.agent_temporal_state import AgentTemporalState
from devkit.common.actor_state.agent import Agent
from devkit.common.actor_state.oriented_box import in_collision
from devkit.common.actor_state.scene_object import SceneObject
from devkit.common.actor_state.scene_object import SceneObjectMetadata
from devkit.common.actor_state.state_representation import TimeDuration
from devkit.common.actor_state.state_representation import TimePoint
from devkit.common.actor_state.state_representation import Point2D
from devkit.common.actor_state.state_representation import StateSE2
from devkit.common.actor_state.state_representation import ProgressStateSE2
from devkit.common.actor_state.state_representation import TemporalStateSE2
from devkit.common.actor_state.state_representation import StateVector2D
from devkit.common.actor_state.static_object import SceneObject
from devkit.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from devkit.common.actor_state.tracked_objects_types import TrackedObjectType
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.common.actor_state.oriented_box import OrientedBox
from devkit.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from devkit.common.geometry.compute import principal_value
from devkit.common.geometry.transform import translate_longitudinally

from devkit.common.actor_state.agent_state import AgentState

from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import ScenarioFileBaseInfo
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import MineSimVehicleAgentState
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import MineSimScenarioVehicleTraj
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import MineSimScenarioTrackedMetaData
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import SimTimePointOrigin
from devkit.configuration.vehicle_conf import vehicle_conf


# 缓存相同的车辆形状转换，避免重复计算
@lru_cache(maxsize=50)
def get_center_shift(location_point_to_head: float, location_point_to_rear: float) -> float:
    """
    计算从后轴中心到车辆中心的距离，并缓存结果以减少重复计算。
    """
    return (location_point_to_head - location_point_to_rear) / 2.0


def convert_to_OrientedBox(vehicle_shape: dict, mine_vehicle_state: dict) -> OrientedBox:
    """
    # NOTE 一次计算，效率不高.
    将MineSim obstacle 车辆的状态信息和外形尺寸转换为 OrientedBox 表示。

    :param vehicle_shape: 包含车辆尺寸信息的字典，车辆的外形轮廓信息，包括长度、宽度、高度等。
        e.g. vehicle_shape={'vehicle_type': 'PickupSuv_BX5', 'length': 4.49, 'width': 1.877, 'height': 1.675, 'locationPoint2Head': 2.245, 'locationPoint2Rear': 2.245}
    :param mine_vehicle_state: 包含车辆状态信息的字典，包括位置、朝向等，参考点是车辆后轴。
        e.g. mine_vehicle_state = {'x': 1643.87201, 'y': 693.58076, 'yaw_rad': 4.47652, 'v_mps': 4.98716, 'yawrate_radps': -0.05318, 'acc_mpss': -0.36987}
    :return: 表示车辆的 OrientedBox 对象，二维平面位置和姿态。
    """
    # 从 mine_vehicle_state 中获取车辆的位置和朝向
    x = mine_vehicle_state["x"]
    y = mine_vehicle_state["y"]
    heading = principal_value(mine_vehicle_state["yaw_rad"], -np.pi)

    # 从 vehicle_shape 中获取车辆的几何信息
    length = vehicle_shape["length"]  # 车辆总长度
    width = vehicle_shape["width"]  # 车辆宽度
    height = vehicle_shape["height"]  # 车辆高度
    location_point_to_head = vehicle_shape["locationPoint2Head"]  # 后轴中心到车头的距离
    location_point_to_rear = vehicle_shape["locationPoint2Rear"]  # 后轴中心到车尾的距离

    # 获取从后轴中心到车辆中心的距离（缓存计算）
    location_point_to_center = get_center_shift(location_point_to_head, location_point_to_rear)

    # 构建后轴中心位置的 StateSE2 对象
    rear_axle_pose = StateSE2(x=x, y=y, heading=heading)

    # 将后轴中心的位置信息转换为车辆中心的位置信息（沿航向方向平移到车辆中心）
    center_pose = translate_longitudinally(rear_axle_pose, location_point_to_center)

    # 使用车辆的中心位置和几何信息创建 OrientedBox
    oriented_box = OrientedBox(center=center_pose, length=length, width=width, height=height)

    return oriented_box


def convert_to_oriented_box_batch(vehicle_shapes: list, mine_vehicle_states: list) -> list:
    """
    批量将MineSim obstacle 车辆的状态信息和外形尺寸转换为 OrientedBox 表示。

    :param vehicle_shapes: 每个车辆的尺寸信息列表。
    :param mine_vehicle_states: 每个车辆的状态信息列表。
    :return: OrientedBox 对象列表。
    """
    oriented_boxes = []
    for vehicle_shape, mine_vehicle_state in zip(vehicle_shapes, mine_vehicle_states):
        # 从 mine_vehicle_state 中获取车辆的位置和朝向
        x = mine_vehicle_state["x"]
        y = mine_vehicle_state["y"]
        heading = principal_value(mine_vehicle_state["yaw_rad"], -np.pi)

        # 从 vehicle_shape 中获取车辆的几何信息
        length = vehicle_shape["length"]
        width = vehicle_shape["width"]
        height = vehicle_shape["height"]
        location_point_to_head = vehicle_shape["locationPoint2Head"]
        location_point_to_rear = vehicle_shape["locationPoint2Rear"]

        # 获取从后轴中心到车辆中心的距离（缓存计算）
        location_point_to_center = get_center_shift(location_point_to_head, location_point_to_rear)

        # 构建后轴中心位置的 StateSE2 对象
        rear_axle_pose = StateSE2(x=x, y=y, heading=heading)

        # 将后轴中心的位置信息转换为车辆中心的位置信息（沿航向方向平移到车辆中心）
        center_pose = translate_longitudinally(rear_axle_pose, location_point_to_center)

        # 使用车辆的中心位置和几何信息创建 OrientedBox
        oriented_box = OrientedBox(center=center_pose, length=length, width=width, height=height)
        oriented_boxes.append(oriented_box)

    return oriented_boxes


def convert_to_TrackedObjectType(vehicle_shape: dict) -> OrientedBox:
    if vehicle_shape["width"] <= 2.5:
        tracked_object_type = TrackedObjectType.CAR
    elif vehicle_shape["width"] <= 3.8 and vehicle_shape["width"] > 2.5:
        tracked_object_type = TrackedObjectType.WIDE_BODY_TRUCK
    elif vehicle_shape["width"] >= 2.5:
        tracked_object_type = TrackedObjectType.MINE_TRUCK
    else:
        tracked_object_type = TrackedObjectType.VEHICLE

    return tracked_object_type


def convert_to_MineSimScenarioVehicleTraj_batch(vehicle_traj: dict, dt: float, sim_time_origin: SimTimePointOrigin) -> list:
    """
    批量将MineSim obstacle 车辆的状态信息和外形尺寸转换为 OrientedBox 表示。

    :param self._scenario_info.vehicle_traj: 每个场景中 原json提取信息的 车辆轨迹。
    sim_time_origin = SimTimePointOrigin
    """
    vehicle_traj_convert_list = []
    for vehicle_id, vehicle_data in vehicle_traj.items():
        # 获取车辆的形状数据
        vehicle_shape = vehicle_data.get("shape", {})
        tracked_object_type = convert_to_TrackedObjectType(vehicle_data["shape"])

        # 遍历每一个时间步，忽略 'shape' 键
        vehicle_agent_state_list = list()
        vehicle_time_s_list = list()
        vehicle_time_s_str_list = list()
        for time_step_str, metadata_state in vehicle_data.items():
            if time_step_str == "shape" or time_step_str == -1 or time_step_str == "-1":
                continue
            # 将时间步保存到列表
            time_s = float(time_step_str)
            vehicle_time_s_list.append(time_s)
            vehicle_time_s_str_list.append(time_step_str)

            oriented_box = convert_to_OrientedBox(vehicle_shape=vehicle_shape, mine_vehicle_state=metadata_state)

            agent_state = AgentState(
                tracked_object_type=tracked_object_type,
                oriented_box=oriented_box,
                velocity=StateVector2D(
                    x=metadata_state["v_mps"],  # 纵向速度,只能为正
                    y=0.0,  # todo 待检查: 侧向速度为0.0是否合适?
                ),
                metadata=SceneObjectMetadata(
                    timestamp_us=sim_time_origin.sim_time_point_origin.time_us + int(time_s * 1e6),
                    track_id_minesim=vehicle_id,
                    token=None,
                    track_id=None,
                    track_token=None,
                    category_name=None,
                ),
                angular_velocity=metadata_state["yawrate_radps"],
                acceleration=StateVector2D(
                    x=metadata_state["acc_mpss"],  # 纵向加速度
                    y=0.0,  # todo 待检查: 侧向加速度为0.0是否合适?
                ),
            )
            vehicle_agent_state = MineSimVehicleAgentState(agent_state=agent_state, metadata_state=metadata_state)

            vehicle_agent_state_list.append(vehicle_agent_state)

            # 构建 MineSimScenarioVehicleTraj 对象

            vehicle_traj_convert = MineSimScenarioVehicleTraj(
                vehicle_id=vehicle_id,
                vehicle_shape=vehicle_shape,
                vehicle_time_s_list=vehicle_time_s_list,
                vehicle_time_s_str_list=vehicle_time_s_str_list,
                dt=dt,
                vehicle_agent_states=vehicle_agent_state_list,
            )

        vehicle_traj_convert_list.append(vehicle_traj_convert)

    return vehicle_traj_convert_list


# def setup_ego_vehicle_parameters(ego_info: dict):
    #
    # if ego_info["shape"]["vehicle_type"] == "MineTruck_XG90G":
    #     vehicle_name = "XG90G"
    #     vehicle_config = vehicle_conf[vehicle_name]
    # elif ego_info["shape"]["vehicle_type"] == "MineTruck_NTE200":
    #     vehicle_name = "NTE200"
    #     vehicle_config = vehicle_conf[vehicle_name]
    # else:
    #     pass

    # ego_vehicle = VehicleParameters(
    #     width=vehicle_config["shape"]["width"],
    #     front_length=vehicle_config["shape"]["locationPoint2Head"],
    #     rear_length=vehicle_config["shape"]["locationPoint2Rear"],
    #     cog_position_from_rear_axle=get_center_shift(
    #         location_point_to_head=vehicle_config["shape"]["locationPoint2Head"],
    #         location_point_to_rear=vehicle_config["shape"]["locationPoint2Rear"],
    #     ),
    #     wheel_base=vehicle_config["shape"]["wheelbase"],
    #     vehicle_name=vehicle_name,
    #     vehicle_type=vehicle_config["vehicle_type"],
    #     height=vehicle_config["shape"]["height"],
    #     shape=vehicle_config["shape"],
    #     constraints=vehicle_config["constraints"],
    # )
    # return ego_vehicle
