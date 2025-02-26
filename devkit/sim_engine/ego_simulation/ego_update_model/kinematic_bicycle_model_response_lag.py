import numpy as np

from devkit.common.actor_state.dynamic_car_state import DynamicCarState
from devkit.common.actor_state.ego_state import EgoState
from devkit.common.actor_state.state_representation import StateVector2D, TimePoint
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.sim_engine.ego_simulation.ego_update_model.kinematic_bicycle_model import KinematicBicycleModel


class KinematicBicycleModelResponseLag(KinematicBicycleModel):
    """
    扩展 KinematicBicycleModel，添加响应延迟模型。
    A class describing the kinematic motion model where the rear axle is the point of reference.
    "models": [
            "KBM",  # Kinematic Bicycle Model (KBM)
            "KBM-wRL",  # Bicycle Model with Response Lag (KBM-wRL)
            "KBM-wRLwRS",  # Bicycle Model with Response Lag and Road Slope (KBM-wRLwRS)
        ],
        NOTE 只在 _update_commands 方法中处理响应延迟，然后调用父类的 propagate_state 进行状态传播。
    """

    def __init__(
        self,
        vehicle: VehicleParameters,
        max_steering_angle: float = None,
        accel_time_constant: float = None,
        steering_angle_time_constant: float = None,
    ):
        super().__init__(vehicle, max_steering_angle)
        """
        初始化响应延迟模型。
        Construct KinematicBicycleModel with Response Lag.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s  加速度低通滤波器时间常数
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s  转向器低通滤波器时间常数
        NOTE: kinematic_bicycle_model_response_lag 的 时间常数都设置为0，该模型即转化为 kinematic_bicycle_model。
        """
        # todo 宽体车 和大型矿车的值应该是不一样的,待修改;
        if accel_time_constant is None:
            self._accel_time_constant = 0.4
        else:
            self._accel_time_constant = accel_time_constant

        if steering_angle_time_constant is None:
            self._steering_angle_time_constant = 0.12
        else:
            self._steering_angle_time_constant = steering_angle_time_constant

    def _update_commands(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """
        This function applies some first order control delay/a low pass filter to acceleration/steering.
        NOTE ： 模拟了车辆底盘的控制器，将【期望加速度、期望转向角】 转化为 底盘控制输入-- 带延迟相应的 【纵向加速度，转向速率】。

        :param state: Ego state
        :param ideal_dynamic_state: The desired dynamic state for propagation
        :param sampling_time: The time duration to propagate for
        :return: propagating_state including updated dynamic_state
        """
        # 从 sampling_time 中提取的时间步长（秒），表示在这个时间段内需要更新的状态。
        # e.g. 0.1s
        dt_control = sampling_time.time_s
        # 前车辆的后轴加速度。steering_angle: 当前车辆的转向角度。
        accel = state.dynamic_car_state.rear_axle_acceleration_2d.x
        steering_angle = state.tire_steering_angle

        # 获取理想加速度和转向角
        ideal_accel_x = ideal_dynamic_state.rear_axle_acceleration_2d.x
        ideal_steering_angle = dt_control * ideal_dynamic_state.tire_steering_rate + steering_angle

        # 应用低通滤波器（基于一阶控制延迟），计算得到更新后的后轴加速度。
        updated_accel_x = dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel

        # 同样应用低通滤波器，计算得到W更新后的转向角。
        updated_steering_angle = (
            dt_control / (dt_control + self._steering_angle_time_constant) * (ideal_steering_angle - steering_angle) + steering_angle
        )
        # 通过更新后的转向角与原始转向角的差值，计算更新后的转向率。
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control

        dynamic_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(updated_accel_x, 0),
            tire_steering_rate=updated_steering_rate,
        )
        propagating_state = EgoState(
            car_footprint=state.car_footprint,
            dynamic_car_state=dynamic_state,
            tire_steering_angle=state.tire_steering_angle,
            is_in_auto_mode=True,
            time_point=state.time_point,
        )
        return propagating_state


if __name__ == "__main__":
    pass
