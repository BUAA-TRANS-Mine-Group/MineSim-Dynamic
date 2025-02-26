from __future__ import annotations

from enum import Enum
from typing import Set


class TrackedObjectType(Enum):
    """Enum of classification types for TrackedObject.
    - 示例:
      - TrackedObjectType.VEHICLE.value  # 返回 0
      - TrackedObjectType.VEHICLE.fullname  # 返回 "vehicle"

    矿区自动驾驶系统检测到的类别通常包括以下几种:
        - **车辆 (VEHICLE) **: 指的是道路上行驶的各种机动车，如轿车、矿车等。包括：大型电动轮矿卡 ， 宽体自卸车，
        - **行人 (PEDESTRIAN) **: 指的是在路上行走的人。
        - **自行车 (BICYCLE) **: 指的是骑自行车的人或自行车本身。
        - **交通锥 (TRAFFIC_CONE) **: 通常用于道路施工或事故现场，起到警示和引导交通的作用。
        - **路障 (BARRIER) **: 用于阻止或限制车辆、行人通行的设施，可以是固定的也可以是临时的。
        - **限速标志 (CZONE_SIGN) **: 指的是特定区域 (如施工区、学校区等) 的限速标志。
        - **通用物体 (GENERIC_OBJECT) **: 指的是不属于上述特定类别但可被检测到的其他物体。
        - **自我车辆 (EGO) **: 指的是装有自动驾驶系统的车辆本身。

        - **矿区的小汽车 (car) **: 指的是矿区出现的小汽车。
        - **矿区的矿卡 (mine_truck) **: 指的是矿区出现的大型电动轮矿卡。
        - **矿区的宽体车 (wide_body_truck) **: 指的是矿区出现的宽体自卸车。
        - **矿区的大石块 (ROCK) **: 指的是矿区出现在行驶道路 附近的的大石块。
        - **矿区的静止车辆 (static_vehicle) **: 指的是矿区出现在行驶道路 附近的静止的车辆；经常出现在道路边,停靠不能用于闭环仿真；
    这些类别帮助自动驾驶系统识别和理解周围环境，从而做出安全的驾驶决策。
    """

    VEHICLE = 0, "vehicle"
    PEDESTRIAN = 1, "pedestrian"
    BICYCLE = 2, "bicycle"
    TRAFFIC_CONE = 3, "traffic_cone"  # 交通锥桶
    BARRIER = 4, "barrier"  # 栅栏； 障碍物；路障
    CZONE_SIGN = 5, "czone_sign"
    GENERIC_OBJECT = 6, "generic_object"
    EGO = 7, "ego"
    CAR = 8, "car"  #
    MINE_TRUCK = 9, "mine_truck"
    WIDE_BODY_TRUCK = 10, "wide_body_truck"
    ROCK = 11, "rock"
    STATIC_VEHICLE = 12, "static_vehicle"

    def __int__(self) -> int:
        """用于将枚举成员转换为整数
        示例: int(TrackedObjectType.PEDESTRIAN)  # 返回 1
        Convert an element to int
        :return: int
        """
        return self.value  # type: ignore

    def __new__(cls, value: int, name: str) -> TrackedObjectType:
        """自定义枚举成员的创建方式。
        Create new element
        :param value: its value
        :param name: its name
        """
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name  # type: ignore
        return member

    def __eq__(self, other: object) -> bool:
        """自定义等值判断方法。检查两个枚举实例是否具有相同的 `name` 和 `value`。
        示例:
            TrackedObjectType.VEHICLE == TrackedObjectType.VEHICLE  # 返回 True
            TrackedObjectType.VEHICLE == TrackedObjectType.PEDESTRIAN  # 返回 False
        Equality checking
        :return: int
        """
        # Cannot check with isisntance, as some code imports this in a different way
        try:
            return self.name == other.name and self.value == other.value  # type: ignore
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash
        - 生成唯一的哈希值，以便将枚举对象用作字典的键或存储在集合中。
        - 示例:
            obj_dict = {TrackedObjectType.EGO: "Ego vehicle"}
            obj_dict[TrackedObjectType.EGO]  # 返回 "Ego vehicle"
        """
        return hash((self.name, self.value))


# VEHICLE = 0, 'vehicle'
# PEDESTRIAN = 1, 'pedestrian'
# BICYCLE = 2, 'bicycle'
# TRAFFIC_CONE = 3, 'traffic_cone'
# BARRIER = 4, 'barrier'
# CZONE_SIGN = 5, 'czone_sign'
# GENERIC_OBJECT = 6, 'generic_object'
# EGO = 7, 'ego'
# CAR = 8, 'car'
# MINE_TRUCK = 9, 'mine_truck'
# WIDE_BODY_TRUCK = 10, 'wide_body_truck'
# ROCK = 11, 'rock'
# TATIC_VEHICLE = 12, "static_vehicle"

tracked_object_types = {
    "vehicles": TrackedObjectType.VEHICLE,
    "pedestrians": TrackedObjectType.PEDESTRIAN,
    "bicycles": TrackedObjectType.BICYCLE,
    "traffic_cone": TrackedObjectType.TRAFFIC_CONE,
    "barrier": TrackedObjectType.BARRIER,
    "czone_sign": TrackedObjectType.CZONE_SIGN,
    "genericobjects": TrackedObjectType.GENERIC_OBJECT,
    "ego": TrackedObjectType.EGO,
    "car": TrackedObjectType.CAR,
    "mine_truck": TrackedObjectType.MINE_TRUCK,
    "wide_body_truck": TrackedObjectType.WIDE_BODY_TRUCK,
    "rock": TrackedObjectType.ROCK,
    "static_vehicle": TrackedObjectType.STATIC_VEHICLE,
}


# 场景中所有的动态 agent dynamic: DYNAMIC_AGENT_TYPES
AGENT_TYPES: Set[TrackedObjectType] = {
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
    TrackedObjectType.EGO,
    TrackedObjectType.CAR,
    TrackedObjectType.MINE_TRUCK,
    TrackedObjectType.WIDE_BODY_TRUCK,
}

# 可以被 闭环仿真的 smart agent #!目前不支持 BICYCLE and PEDESTRIAN
SMART_AGENT_TYPES: Set[TrackedObjectType] = {
    TrackedObjectType.VEHICLE,
    TrackedObjectType.CAR,
    TrackedObjectType.MINE_TRUCK,
    TrackedObjectType.WIDE_BODY_TRUCK,
}

# 不能被用于闭环仿真, 只能回放，因此他通常为 静态的 object
STATIC_OBJECT_TYPES: Set[TrackedObjectType] = {
    TrackedObjectType.CZONE_SIGN,
    TrackedObjectType.BARRIER,
    TrackedObjectType.TRAFFIC_CONE,
    TrackedObjectType.GENERIC_OBJECT,
    TrackedObjectType.ROCK,
    TrackedObjectType.STATIC_VEHICLE,
}
