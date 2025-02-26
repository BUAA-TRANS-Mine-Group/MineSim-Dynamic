import numpy as np

from devkit.common.actor_state.dynamic_car_state import DynamicCarState
from devkit.common.actor_state.ego_state import EgoState, EgoStateDot
from devkit.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from devkit.common.actor_state.vehicle_parameters import VehicleParameters
from devkit.common.geometry.compute import principal_value
from devkit.sim_engine.ego_simulation.ego_update_model.abstract_ego_state_update_model import AbstractEgoStateUpdateModel
from devkit.sim_engine.ego_simulation.ego_update_model.forward_integrate import forward_integrate

# MineSim-Dynamic-Dev/devkit/sim_engine/ego_update_model/abstract_ego_state_update_model.py


class KinematicBicycleModel(AbstractEgoStateUpdateModel):
    """
    A class describing the kinematic motion model where the rear axle is the point of reference.
    "models": [
            "KBM",  # Kinematic Bicycle Model (KBM)
            "KBM-wRL",  # Bicycle Model with Response Lag (KBM-wRL)
            "KBM-wRLwRS",  # Bicycle Model with Response Lag and Road Slope (KBM-wRLwRS)
        ],
    """

    def __init__(
        self,
        vehicle: VehicleParameters,
        max_steering_angle: float = None,
    ):
        """
        Construct KinematicBicycleModel.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.

        """
        self._vehicle = vehicle
        if max_steering_angle is None:
            self._max_steering_angle = np.pi / 3
        else:
            self._max_steering_angle = max_steering_angle

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """Inherited, see super class."""
        # [Chinese] 车辆的状态导数
        longitudinal_speed = state.dynamic_car_state.rear_axle_velocity_2d.x
        x_dot = longitudinal_speed * np.cos(state.rear_axle.heading)
        y_dot = longitudinal_speed * np.sin(state.rear_axle.heading)
        yaw_dot = longitudinal_speed * np.tan(state.tire_steering_angle) / self._vehicle.wheel_base

        return EgoStateDot.build_from_rear_axle(
            rear_axle_pose=StateSE2(x=x_dot, y=y_dot, heading=yaw_dot),
            rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_acceleration_2d,
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
            tire_steering_angle=state.dynamic_car_state.tire_steering_rate,
            time_point=state.time_point,
            is_in_auto_mode=True,
            vehicle_parameters=self._vehicle,
        )

    def _update_commands(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """
        This function applies some first order control delay/a low pass filter to acceleration/steering.

        :param state: Ego state
        :param ideal_dynamic_state: The desired dynamic state for propagation
        :param sampling_time: The time duration to propagate for
        :return: propagating_state including updated dynamic_state
        """
        # 从 sampling_time 中提取的时间步长delta_t （秒），表示在这个时间段内需要更新的状态。# e.g. 0.1s
        dt_control = sampling_time.time_s
        # 前车辆的后轴加速度。steering_angle: 当前车辆的转向角度。
        accel = state.dynamic_car_state.rear_axle_acceleration_2d.x
        steering_angle = state.tire_steering_angle

        # 获取理想加速度和转向角
        ideal_accel_x = ideal_dynamic_state.rear_axle_acceleration_2d.x
        ideal_steering_angle = dt_control * ideal_dynamic_state.tire_steering_rate + steering_angle

        # 不进行 低通滤波器（基于一阶控制延迟），直接赋值得到更新后的后轴加速度。
        updated_accel_x = ideal_accel_x

        # 不进行 低通滤波器（基于一阶控制延迟），直接赋值得到更新后的转向角。
        updated_steering_angle = ideal_steering_angle
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

    def propagate_state(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """状态传播逻辑。
        0.更新底盘底层控制器控制指令。
        1.计算状态导数 (get_state_dot)。
        2.通过欧拉积分更新状态（位置、速度、转向角等）。
        3.返回更新后的 EgoState。
        """
        # 0.更新底盘底层控制器控制指令。
        propagating_state = self._update_commands(state, ideal_dynamic_state, sampling_time)

        # Compute state derivatives
        state_dot = self.get_state_dot(propagating_state)

        # Integrate position and heading 2.通过欧拉积分更新状态（位置、速度、转向角等）。
        # x_{\text{next}} = x + \dot{x} \cdot \Delta t
        next_x = forward_integrate(propagating_state.rear_axle.x, state_dot.rear_axle.x, sampling_time)
        # y_{\text{next}} = y + \dot{y} \cdot \Delta t
        next_y = forward_integrate(propagating_state.rear_axle.y, state_dot.rear_axle.y, sampling_time)
        # \dot{\theta} = \frac{v_x \cdot \tan(\delta)}{L}
        next_heading = forward_integrate(propagating_state.rear_axle.heading, state_dot.rear_axle.heading, sampling_time)
        #! Wrap angle between [-pi, pi]
        next_heading = principal_value(next_heading)

        # Compute rear axle velocity in car frame
        # v_next = v + a * delat_T
        next_point_velocity_x = forward_integrate(
            propagating_state.dynamic_car_state.rear_axle_velocity_2d.x,
            state_dot.dynamic_car_state.rear_axle_velocity_2d.x,
            sampling_time,
        )
        next_point_velocity_y = 0.0  #! Lateral velocity is always zero in kinematic bicycle model

        # Integrate steering angle and clip to bounds
        # steering_angel_next = steering_angel + tire_steering_rate * delat_T
        next_point_tire_steering_angle = np.clip(
            forward_integrate(propagating_state.tire_steering_angle, state_dot.tire_steering_angle, sampling_time),
            -self._max_steering_angle,
            self._max_steering_angle,
        )

        # Compute angular velocity
        next_point_angular_velocity = next_point_velocity_x * np.tan(next_point_tire_steering_angle) / self._vehicle.wheel_base

        rear_axle_accel = [
            state_dot.dynamic_car_state.rear_axle_velocity_2d.x,
            state_dot.dynamic_car_state.rear_axle_velocity_2d.y,
        ]
        angular_accel = (next_point_angular_velocity - state.dynamic_car_state.angular_velocity) / sampling_time.time_s

        # 3.返回更新后的 EgoState。
        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(next_x, next_y, next_heading),  # 【 右手坐标系，x-纵向 向前为正，y-横向 向左为正】
            rear_axle_velocity_2d=StateVector2D(next_point_velocity_x, next_point_velocity_y),
            rear_axle_acceleration_2d=StateVector2D(rear_axle_accel[0], rear_axle_accel[1]),
            tire_steering_angle=float(next_point_tire_steering_angle),
            time_point=propagating_state.time_point + sampling_time,
            vehicle_parameters=self._vehicle,
            is_in_auto_mode=True,
            angular_vel=next_point_angular_velocity,
            angular_accel=angular_accel,
            tire_steering_rate=state_dot.tire_steering_angle,
        )


if __name__ == "__main__":
    pass
