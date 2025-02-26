from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Type


@dataclass
class VehicleParameters:
    """
    Class holding parameters of a vehicle (使用 @dataclass 自动生成 __init__ 和 __repr__，简化代码)

    - max_acceleration: 3.0  # [m/s^2] Absolute value threshold on acceleration input.
    - max_steering_angle: 1.047197  # [rad] Absolute value threshold on steering angle.
    - max_steering_angle_rate: 0.5  # [rad/s] Absolute value threshold on steering rate input.
    """

    width: float  # [m] Box width
    front_length: float  # [m] 后轴到前保险杠距离 [m] distance between rear axle and front bumper
    rear_length: float  # [m] 后轴到后保险杠距离  [m] distance between rear axle and rear bumper
    cog_position_from_rear_axle: float  # [m] COG 到后轴距离 [m] distance between rear axle and center of gravity (cog)
    wheel_base: float  # [m] 轴距 [m] wheel base of the vehicle
    vehicle_name: str  # 车辆型号  name of the vehicle; ["XG90G","NTE200"]
    vehicle_type: str  # 车辆类型 type of the vehicle; ["XG90G":"wide-body-dump-truck","NTE200":"electric-drive-mining-truck"]
    height: Optional[float] = None  # [m] 车高
    shape: Dict[str, float] = None  # 几何形状参数
    constraints: Dict[str, float] = None  # 运动学约束
    ilqr_tracker_constraints: Dict[str, float] = None  # iLQR 跟踪约束

    def __post_init__(self):
        """数据一致性检查"""
        self.shape = self.shape or {}
        self.constraints = self.constraints or {}
        self.ilqr_tracker_constraints = self.ilqr_tracker_constraints or {}
        self.length = self.front_length + self.rear_length
        # 确保关键参数存在
        assert "locationPoint2Head" in self.shape, "shape 必须包含 locationPoint2Head"
        assert "locationPoint2Rear" in self.shape, "shape 必须包含 locationPoint2Rear"

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def rear_axle_to_center(self) -> float:
        """
        :return: [m] distance between rear axle and center of vehicle
        """
        return abs(self.half_length - self.rear_length)

    @property
    def length_cog_to_front_axle(self) -> float:
        """COG 到前轴的距离
        :return: [m] distance between cog and front axle
        """
        return self.wheel_base - self.cog_position_from_rear_axle

    def __hash__(self) -> int:
        """哈希计算 (用于对象唯一性判断)
        :return: hash vehicle parameters
        """
        return hash(
            (
                self.vehicle_name,
                self.vehicle_type,
                self.width,
                self.front_length,
                self.rear_length,
                self.cog_position_from_rear_axle,
                self.wheel_base,
                self.height,
            )
        )

    def __str__(self) -> str:
        """字符串表示 (用于调试和日志)
        :return: string for this class
        """
        return (
            f"VehicleParameters(vehicle_name={self.vehicle_name}, "
            f"vehicle_type={self.vehicle_type}, width={self.width}, "
            f"front_length={self.front_length}, rear_length={self.rear_length}, "
            f"cog_position_from_rear_axle={self.cog_position_from_rear_axle}, "
            f"wheel_base={self.wheel_base}, height={self.height}, "
            f"locationPoint2Head={self.shape.get('locationPoint2Head')}, "
            f"locationPoint2Rear={self.shape.get('locationPoint2Rear')})"
        )

    def __reduce__(self) -> Tuple[Type[VehicleParameters], Tuple[Any, ...]]:
        """
        ! 自定义 反序列化时如何重建对象。
        在绝大多数情况下，你并不需要为每个类显式地定义 __reduce__ ,只有在以下情况时，才考虑自定义 __reduce__：
        1) 类中的初始化逻辑或属性设置在默认 pickle 机制下不能被正确还原。
        2) 你需要精确控制序列化/反序列化的过程，包括哪些属性被序列化、以何种方式进行序列化等。
        3) 你使用了C 扩展类型或特殊对象，默认的 pickle 机制可能无法处理。
        :return: tuple of class and its constructor parameters, this is used to pickle the class
        """
        return self.__class__, (
            self.width,
            self.front_length,
            self.rear_length,
            self.cog_position_from_rear_axle,
            self.wheel_base,
            self.vehicle_name,
            self.vehicle_type,
            self.height,
            self.shape,
            self.constraints,
            self.ilqr_tracker_constraints,
        )


@lru_cache(maxsize=2)
def get_mine_truck_parameters(mine_name: str) -> VehicleParameters:
    """获取矿卡参数 (支持缓存优化)
    - mine_name= "guangdong_dapai";map_name = "guangdong_dapai";ego vehicle_name= ="XG90G";
    - mine_name= "jiangxi_jiangtong";map_name = "jiangxi_jiangtong";ego vehicle_name= ="NTE200"

    :return VehicleParameters containing parameters of mine_truck Vehicle
    """
    if mine_name == "guangdong_dapai":
        return VehicleParameters(
            vehicle_name="XG90G",
            vehicle_type="wide-body-dump-truck",
            width=4.0,
            front_length=6.5,
            rear_length=2.5,
            wheel_base=5.3,
            cog_position_from_rear_axle=(6.5 + 2.5) * 0.5 - 2.5,
            height=3.5,
            shape={
                "length": 9.0,
                "width": 4.0,
                "locationPoint2Head": 6.5,
                "locationPoint2Rear": 2.5,
                "wheel_base": 5.3,
                "height": 3.5,
            },
            # 车辆自身的约束参数;可以不与 算法(控制,规划)的约束参数相等,通常 算法(控制,规划)的约束参数适当小一些
            constraints={
                "min_steering_angle": -1.8,  # rad
                "max_steering_angle": 1.8,
                "max_steering_velocity": 0.3,  # rad/s
                "max_longitudinal_velocity": 16.7,  # m/s ; 16.7 (60 Km/h) 12.5 (45 Km/h)
                "min_longitudinal_acceleration": -15.5,  # m/s^2
                "max_longitudinal_acceleration": 16.8,
                "max_centripetal_acceleration": 1.5,
                "max_lateral_acceleration": 1.2,
                "min_turning_radius": 5.44,  # m,R = L / sin(max_steering_angle)
            },
            ilqr_tracker_constraints={
                "max_acceleration": 5.2,  # rad
                "max_steering_angle": 1.22,
                "max_steering_angle_rate": 0.26,
            },
        )
    elif mine_name == "jiangxi_jiangtong":
        return VehicleParameters(
            vehicle_name="NTE200",
            vehicle_type="electric-drive-mining-truck",
            width=6.7,
            front_length=9.2,
            rear_length=3.8,
            wheel_base=5.3,
            cog_position_from_rear_axle=(9.2 + 3.8) * 0.5 - 3.8,  # 几何中心到 后桥的位置
            height=6.9,
            shape={
                "length": 13.0,
                "width": 6.7,
                "locationPoint2Head": 9.2,
                "locationPoint2Rear": 3.8,
                "wheel_base": 9.6,
                "height": 6.9,
            },
            constraints={
                "min_steering_angle": -1.5,  # rad
                "max_steering_angle": 1.5,
                "max_steering_velocity": 0.2,
                "max_longitudinal_velocity": 12.5,  # 12.5 (45 Km/h)
                "min_longitudinal_acceleration": -8.9,
                "max_longitudinal_acceleration": 10.2,
                "max_centripetal_acceleration": 1.5,
                "max_lateral_acceleration": 1.2,
                "min_turning_radius": 9.62,  # m
            },
            ilqr_tracker_constraints={
                "max_acceleration": 4.0,  # rad
                "max_steering_angle": 1.13,
                "max_steering_angle_rate": 0.18,
            },
        )
    else:
        raise ValueError(f"Unsupported mine_name: {mine_name}")


def get_vehicle_waypoint_speed_limit_data(vehicle_width: float) -> tuple[float, float, float]:
    """获取车辆-waypoints 计算路径限速的参数（露天矿区场景专用）

    1. **横向加速度 (max_lateral_accel)**：
    - 随车辆尺寸增大而降低（重心高度↑ → 侧翻风险↑）
    - 参考值：
        - 普通车辆：通常2.5-3.0 m/s²
        - 重型矿卡：实测数据建议 ≤1.5 m/s²

    2. **速度限制**：
        - 最高速度 (max_speed_limit)：
            - 直道限制：按矿区道路等级（主干道/支线）
            - 弯道限制：通过 `v = sqrt(a_max * R)` 动态计算（R为转弯半径）; a_max实际是向心加速度;
        - 最低速度 (min_speed_limit)：防止拥堵并保持设备稳定性

    3. **取值**：
        - 三种车辆类型是小车、中型车（宽体自卸车）和大型电动矿用卡车。露天矿区的道路条件可能比较复杂，弯道较多，所以车辆的横向加速度限制很重要，以避免侧滑或翻车。
        - 同时，矿用卡车通常载重大，速度相对较低，而小车可能更灵活，速度较高。
        - 参数设计依据：
            | 车辆类型 （width）| 横向加速度 | 最高速度 | 最低速度 | 对应现实场景                     |
            |-----------------|------------|----------|----------|-----------------------------|
            | 轻型车辆（≤3m）   | 2.5 m/s²   | 10 m/s   | 5 m/s    | 矿区巡查车快速机动             |
            | 宽体矿卡（≤5m）   | 1.5 m/s²   | 7 m/s    | 4 m/s    | 90吨级矿卡重载运输             |
            | 巨型电驱矿卡（≤9m）| 1.0 m/s²   | 5 m/s    | 3 m/s    | 200吨级电动轮矿卡坡道行驶       |

    Return: (max_lateral_accel, max_speed_limit, min_speed_limit) 单位：m/s², m/s, m/s
    """
    if vehicle_width <= 3.0:  # 轻型车辆（皮卡/巡查车）
        # 典型车型：丰田Hilux（车宽1.85m），允许较高机动性
        return (2.5, 13.89, 4.15)  # 横向加速度2.5m/s²，最高50km/h，最低15km/h

    elif vehicle_width <= 5.0:  # 宽体矿卡（如XG90G）
        # 车宽4.0m，载重90吨，重心较高需保守控制
        return (1.5, 11.67, 2.78)  # 横向加速度1.5m/s²，最高42km/h，最低10km/h ；16.7 (60 Km/h) 12.5 (45 Km/h) *0.7

    elif vehicle_width <= 9.0:  # 超大型电驱矿卡（如NTE200）
        # 车宽6.7m，载重200吨，需严格限制动态性能
        return (1.0, 10.11, 2.22)  # 横向加速度1.0m/s²，最高18km/h，最低8km/h ； 16.7 (60 Km/h) 12.5 (45 Km/h)*0.7

    else:  # 异常尺寸处理
        raise ValueError(f"Unsupported vehicle width {vehicle_width}m (max 9.0m)")


if __name__ == "__main__":

    test_Vehicle_Parameters = get_mine_truck_parameters()
    pass
