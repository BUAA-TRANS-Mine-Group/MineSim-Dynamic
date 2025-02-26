from __future__ import annotations

import math
from functools import cached_property
from typing import Tuple

import numpy as np
import numpy.typing as npt

from devkit.common.actor_state.state_representation import StateVector2D


def get_velocity_shifted(displacement: StateVector2D, ref_velocity: StateVector2D, ref_angular_vel: float) -> StateVector2D:
    """
    Computes the velocity at a query point on the same planar rigid body as a reference point.
    :param displacement: [m] The displacement vector from the reference to the query point
    :param ref_velocity: [m/s] The velocity vector at the reference point
    :param ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
    :return: [m/s] The velocity vector at the given displacement.

    计算同一平面刚体上相对于参考点的查询点的速度（考虑刚体运动学）。
    :param displacement: [米] 从参考点到查询点的位移向量 (Δx, Δy)
    :param ref_velocity: [米/秒] 参考点的速度向量 (vx, vy)
    :param ref_angular_vel: [弧度/秒] 刚体绕垂直轴的角速度
    :return: [米/秒] 查询点的速度向量
    """
    # From cross product of velocity transfer formula in 2D
    # 根据刚体速度传递公式，计算因旋转引起的速度变化项（二维叉乘等效）
    velocity_shift_term: npt.NDArray[np.float64] = np.array([-displacement.y * ref_angular_vel, displacement.x * ref_angular_vel])
    # 总速度 = 参考点速度 + 旋转引起的速度项
    return StateVector2D(*(ref_velocity.array + velocity_shift_term))


def get_acceleration_shifted(
    displacement: StateVector2D, ref_accel: StateVector2D, ref_angular_vel: float, ref_angular_accel: float
) -> StateVector2D:
    """
    Computes the acceleration at a query point on the same planar rigid body as a reference point.
    :param displacement: [m] The displacement vector from the reference to the query point
    :param ref_accel: [m/s^2] The acceleration vector at the reference point
    :param ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
    :param ref_angular_accel: [rad/s^2] The angular acceleration of the body around the vertical axis
    :return: [m/s^2] The acceleration vector at the given displacement.

    计算同一平面刚体上相对于参考点的查询点的加速度（考虑刚体运动学）。度
    :param displacement: [m] 参考点到查询点的位移矢量
    :param ref_accel: [m/s^2] 参考点的加速度矢量
    :param ref_angular_vel: [rad/s] 刚体绕垂直轴的角速度
    :param ref_angular_accel: [rad/s^2] 刚体绕垂直轴的角加速度
    :return: [m/s^2] 查询点的加速度矢量
    """
    # 向心加速度项：方向指向旋转中心，计算公式为 ω² * displacement
    centripetal_acceleration_term = displacement.array * ref_angular_vel**2  # (Δx*ω², Δy*ω²)
    # 角加速度引起的切向加速度项：方向由角加速度方向决定，计算公式为 α * displacement
    angular_acceleration_term = displacement.array * ref_angular_accel  # (Δx*α, Δy*α)
    # 总加速度 = 参考点加速度 + 向心加速度项 + 切向加速度项
    return StateVector2D(*(ref_accel.array + centripetal_acceleration_term + angular_acceleration_term))


def _get_beta(steering_angle: float, wheel_base: float) -> float:
    """
    Computes beta, the angle from rear axle to COG at instantaneous center of rotation
    :param [rad] steering_angle: steering angle of the car
    :param [m] wheel_base: distance between the axles
    :return: [rad] Value of beta

    计算后轮轴到车辆几何中心的瞬时转动中心角度 beta。
    :param steering_angle: [rad] 车轮前轮转向角
    :param wheel_base: [m] 前后车轴的距离
    :return: [rad] beta 值
    """
    beta = math.atan2(math.tan(steering_angle), wheel_base)  # β = arctan(tan(δ)/L)
    return beta


def _projected_velocities_from_cog(beta: float, cog_speed: float) -> Tuple[float, float]:
    """
    Computes the projected velocities at the rear axle using the Bicycle kinematic model using COG data
    :param beta: [rad] the angle from rear axle to COG at instantaneous center of rotation
    :param cog_speed: [m/s] Magnitude of velocity vector at COG
    :return: Tuple with longitudinal and lateral velocities [m/s] at the rear axle

    使用自行车模型从车辆几何中心 (COG) 计算投影至后轮轴的速度： 纵向和横向速度（假设无侧滑）。
    :param beta: [rad] 瞬时转动中心角度
    :param cog_speed: [m/s] 几何中心的速度大小
    :return: 后轮轴的纵向和横向速度 [m/s]
    """
    # This gives COG longitudinal, which is the same as rear axle 纵向速度：沿车辆前进方向的分量（假设与质心速度方向一致）
    rear_axle_forward_velocity = math.cos(beta) * cog_speed  # [m/s]
    # Lateral velocity is zero, by model assumption  # 横向速度：自行车模型假设为0（无侧滑）
    rear_axle_lateral_velocity = 0

    return rear_axle_forward_velocity, rear_axle_lateral_velocity


def _angular_velocity_from_cog(cog_speed: float, length_rear_axle_to_cog: float, beta: float, steering_angle: float) -> float:
    """
    Computes the angular velocity using the Bicycle kinematic model using COG data.
    :param cog_speed: [m/s] Magnitude of velocity vector at COG
    :param length_rear_axle_to_cog: [m] Distance from rear axle to COG
    :param beta: [rad] angle from rear axle to COG at instantaneous center of rotation
    :param steering_angle: [rad] of the car

    使用自行车模型从几何中心 (COG) 计算角速度。
    :param cog_speed: [m/s] 几何中心的速度
    :param length_rear_axle_to_cog: [m] 后轮轴到几何中心的距离
    :param beta: [rad] 瞬时转动中心角度
    :param steering_angle: [rad] 车轮转向角
    :return: 角速度 [rad/s]
    """
    return (cog_speed / length_rear_axle_to_cog) * math.cos(beta) * math.tan(steering_angle)


def _project_accelerations_from_cog(
    rear_axle_longitudinal_velocity: float, angular_velocity: float, cog_acceleration: float, beta: float
) -> Tuple[float, float]:
    """
    Computes the projected accelerations at the rear axle using the Bicycle kinematic model using COG data
    :param rear_axle_longitudinal_velocity: [m/s] Longitudinal component of velocity vector at COG
    :param angular_velocity: [rad/s] Angular velocity at COG
    :param cog_acceleration: [m/s^2] Magnitude of acceleration vector at COG
    :param beta: [rad] ]the angle from rear axle to COG at instantaneous center of rotation
    :return: Tuple with longitudinal and lateral velocities [m/s] at the rear axle

    使用自行车模型从几何中心 (COG) 计算投影至后轮轴的加速度： 纵向和横向加速度。
    :param rear_axle_longitudinal_velocity: [m/s] 后轴纵向速度
    :param angular_velocity: [rad/s] 角速度
    :param cog_acceleration: [m/s^2] 几何中心的加速度
    :param beta: [rad] 瞬时转动中心角度
    :return: 后轮轴的纵向和横向加速度 [m/s^2]
    """
    # Rigid body assumption, can project from COG 纵向加速度：质心加速度在车辆纵向的分量 # a_long = a_cog * cos(β)
    rear_axle_longitudinal_acceleration = math.cos(beta) * cog_acceleration  # [m/s^2]
    # Centripetal accel is a=v^2 / R and angular_velocity = v / R 横向加速度：由向心加速度引起，公式为  a_lat = v_long * ω
    rear_axle_lateral_acceleration = rear_axle_longitudinal_velocity * angular_velocity  # [m/s^2]
    return rear_axle_longitudinal_acceleration, rear_axle_lateral_acceleration


class DynamicCarState:
    """Contains the various dynamic attributes of ego.
    处理车的动态状态，包括后轴速度、加速度、角速度、角加速度等信息，并基于此来推算车辆几何中心的速度和加速度
    # 一些 getter 和缓存属性方法提供了对后轴、车辆几何中心速度、加速度等的访问
    """

    def __init__(
        self,
        rear_axle_to_center_dist: float,
        rear_axle_velocity_2d: StateVector2D,
        rear_axle_acceleration_2d: StateVector2D,
        angular_velocity: float = 0.0,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ):
        """
        :param rear_axle_to_center_dist:[m]  Distance (positive) from rear axle to the geometrical center of ego
        :param rear_axle_velocity_2d: [m/s]Velocity vector at the rear axle
        :param rear_axle_acceleration_2d: [m/s^2] Acceleration vector at the rear axle
        :param angular_velocity: [rad/s] Angular velocity of ego
        :param angular_acceleration: [rad/s^2] Angular acceleration of ego
        :param tire_steering_rate: [rad/s] Tire steering rate of ego

        ：param rear_axle_to_center_dist:[m]后桥到自我几何中心的距离（正）, 一般后轴中心在几何中心后方；
        ：param rear_axle_velocity_2d: [m/s] 后轴的速度矢量，以 2D 矢量形式表示，包括纵向和横向速度分量。【 右手坐标系，x-纵向 向前为正，y-横向 向左为正】
        ：param rear_axle_acceleration_2d: [m/s^2]后轴处的加速度矢量 ，表示纵向和横向的加速度。【 右手坐标系，x-纵向 向前为正，y-横向 向左为正】
        ：param angular_velocity: [rad/s] 车辆的角速度，表示车辆绕垂直轴的旋转速度，单位是弧度每秒（rad/s）。
        ：param angular_acceleration: [rad/s^2] 车辆的角加速度，表示角速度的变化率，单位是弧度每平方秒（rad/s²）。
        ：param tire_steering_rate: [rad/s] 轮胎的转向速率，表示车轮转向的速率，单位是弧度每秒（rad/s）。
        """
        self._rear_axle_to_center_dist = rear_axle_to_center_dist
        self._angular_velocity = angular_velocity
        self._angular_acceleration = angular_acceleration
        self._rear_axle_velocity_2d = rear_axle_velocity_2d
        self._rear_axle_acceleration_2d = rear_axle_acceleration_2d
        self._tire_steering_rate = tire_steering_rate

    @property
    def rear_axle_velocity_2d(self) -> StateVector2D:
        """
        Returns the vectorial velocity at the middle of the rear axle.
        后轴的速度矢量，以 2D 矢量形式表示，包括纵向和横向速度分量。【 右手坐标系，x-纵向 向前为正，y-横向 向左为正】
        :return: StateVector2D Containing the velocity at the rear axle
        """
        return self._rear_axle_velocity_2d

    @property
    def rear_axle_acceleration_2d(self) -> StateVector2D:
        """
        Returns the vectorial acceleration at the middle of the rear axle.
        :return: StateVector2D Containing the acceleration at the rear axle
        """
        return self._rear_axle_acceleration_2d

    @cached_property  # 缓存计算结果，避免重复计算
    def center_velocity_2d(self) -> StateVector2D:
        """
        Returns the vectorial velocity at the geometrical center of Ego.
        根据后轴速度和角速度推算出车辆几何中心的速度，纵向和横向速度分量。【 右手坐标系，x-纵向 向前为正，y-横向 向左为正】
        :return: StateVector2D Containing the velocity at the geometrical center of Ego
        """
        displacement = StateVector2D(self._rear_axle_to_center_dist, 0.0)  # 位移向量 (Δx, 0)
        return get_velocity_shifted(displacement, self.rear_axle_velocity_2d, self.angular_velocity)

    @cached_property
    def center_acceleration_2d(self) -> StateVector2D:
        """
        Returns the vectorial acceleration at the geometrical center of Ego.
        根据后轴加速度和角加速度推算出几何中心的加速度。
        :return: StateVector2D Containing the acceleration at the geometrical center of Ego
        """
        displacement = StateVector2D(self._rear_axle_to_center_dist, 0.0)
        return get_acceleration_shifted(displacement, self.rear_axle_acceleration_2d, self.angular_velocity, self.angular_acceleration)

    @property
    def angular_velocity(self) -> float:
        """
        Getter for the angular velocity of ego.
        :return: [rad/s] Angular velocity
        """
        return self._angular_velocity

    @property
    def angular_acceleration(self) -> float:
        """
        Getter for the angular acceleration of ego.
        :return: [rad/s^2] Angular acceleration
        """
        return self._angular_acceleration

    @property
    def tire_steering_rate(self) -> float:
        """
        Getter for the tire steering rate of ego.
        :return: [rad/s] Tire steering rate
        """
        return self._tire_steering_rate

    @cached_property
    def speed(self) -> float:
        """
        返回车辆的标量速度，即后轴中心的速度的模。
        Magnitude of the speed of the center of ego.
        :return: [m/s] 1D speed
        """
        return float(self._rear_axle_velocity_2d.magnitude())

    @cached_property
    def acceleration(self) -> float:
        """
        返回车辆的标量加速度，即后轴中心的加速度的模。
        Magnitude of the acceleration of the center of ego.
        :return: [m/s^2] 1D acceleration
        """
        return float(self._rear_axle_acceleration_2d.magnitude())

    def __eq__(self, other: object) -> bool:
        """
        Compare two instances whether they are numerically close
        :param other: object
        :return: true if the classes are almost equal
        判断两个DynamicCarState实例是否近似相等。
        :param other: 另一个对象
        :return: 如果所有属性近似相等则返回True，否则返回False
        """
        if not isinstance(other, DynamicCarState):
            # Return NotImplemented in case the classes do not match
            return NotImplemented

        return (
            self.rear_axle_velocity_2d == other.rear_axle_velocity_2d
            and self.rear_axle_acceleration_2d == other.rear_axle_acceleration_2d
            and math.isclose(self._angular_acceleration, other._angular_acceleration)
            and math.isclose(self._angular_velocity, other._angular_velocity)
            and math.isclose(self._rear_axle_to_center_dist, other._rear_axle_to_center_dist)
            and math.isclose(self._tire_steering_rate, other._tire_steering_rate)
        )

    def __repr__(self) -> str:
        """Repr magic method 返回实例的字符串表示，用于调试。"""
        return (
            f"Rear Axle| velocity: {self.rear_axle_velocity_2d}, acceleration: {self.rear_axle_acceleration_2d}\n"
            f"Center   | velocity: {self.center_velocity_2d}, acceleration: {self.center_acceleration_2d}\n"
            f"angular velocity: {self.angular_velocity}, angular acceleration: {self._angular_acceleration}\n"
            f"rear_axle_to_center_dist: {self._rear_axle_to_center_dist} \n"
            f"_tire_steering_rate: {self._tire_steering_rate} \n"
        )

    @staticmethod
    def build_from_rear_axle(
        rear_axle_to_center_dist: float,
        rear_axle_velocity_2d: StateVector2D,
        rear_axle_acceleration_2d: StateVector2D,
        angular_velocity: float = 0.0,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ) -> DynamicCarState:
        """
        Construct ego state from rear axle parameters
        :param rear_axle_to_center_dist: [m] distance between center and rear axle
        :param rear_axle_velocity_2d: [m/s] velocity at rear axle
        :param rear_axle_acceleration_2d: [m/s^2] acceleration at rear axle
        :param angular_velocity: [rad/s] angular velocity
        :param angular_acceleration: [rad/s^2] angular acceleration
        :param tire_steering_rate: [rad/s] tire steering_rate
        :return: constructed DynamicCarState of ego.

        通过后轴参数构造DynamicCarState实例。
        :param rear_axle_to_center_dist: [米] 后轴到中心的距离
        :param rear_axle_velocity_2d: [米/秒] 后轴速度向量
        :param rear_axle_acceleration_2d: [米/秒²] 后轴加速度向量
        :param angular_velocity: [弧度/秒] 角速度
        :param angular_acceleration: [弧度/秒²] 角加速度
        :param tire_steering_rate: [弧度/秒] 转向速率
        :return: DynamicCarState实例
        """
        return DynamicCarState(
            rear_axle_to_center_dist=rear_axle_to_center_dist,
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            tire_steering_rate=tire_steering_rate,
        )

    @staticmethod
    def build_from_cog(
        wheel_base: float,
        rear_axle_to_center_dist: float,
        cog_speed: float,
        cog_acceleration: float,
        steering_angle: float,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ) -> DynamicCarState:
        """
        Construct ego state from rear axle parameters
        :param wheel_base: distance between axles [m]
        :param rear_axle_to_center_dist: distance between center and rear axle [m]
        :param cog_speed: magnitude of speed COG [m/s]
        :param cog_acceleration: magnitude of acceleration at COG [m/s^s]
        :param steering_angle: steering angle at tire [rad]
        :param angular_acceleration: angular acceleration
        :param tire_steering_rate: tire steering rate
        :return: constructed DynamicCarState of ego.

        通过质心参数构造DynamicCarState实例（基于自行车模型）。
        :param wheel_base: [米] 前后轴距
        :param rear_axle_to_center_dist: [米] 后轴到几何中心的距离
        :param cog_speed: [米/秒] 质心速度大小
        :param cog_acceleration: [米/秒²] 质心加速度大小
        :param steering_angle: [弧度] 前轮转向角
        :param angular_acceleration: [弧度/秒²] 角加速度
        :param tire_steering_rate: [弧度/秒] 转向速率
        :return: DynamicCarState实例
        """
        # under kinematic state assumption: compute additionally needed states
        beta = _get_beta(steering_angle, wheel_base)

        rear_axle_longitudinal_velocity, rear_axle_lateral_velocity = _projected_velocities_from_cog(beta, cog_speed)

        angular_velocity = _angular_velocity_from_cog(cog_speed, wheel_base, beta, steering_angle)

        # under kinematic state assumption: compute additionally needed states 计算beta角
        beta = _get_beta(steering_angle, wheel_base)
        # 计算后轴纵向和横向速度
        rear_axle_longitudinal_velocity, rear_axle_lateral_velocity = _projected_velocities_from_cog(beta, cog_speed)
        # 计算角速度
        angular_velocity = _angular_velocity_from_cog(cog_speed, wheel_base, beta, steering_angle)
        # compute acceleration at rear axle given the kinematic assumptions
        # 计算后轴纵向和横向加速度
        longitudinal_acceleration, lateral_acceleration = _project_accelerations_from_cog(
            rear_axle_longitudinal_velocity, angular_velocity, cog_acceleration, beta
        )

        return DynamicCarState(
            rear_axle_to_center_dist=rear_axle_to_center_dist,
            rear_axle_velocity_2d=StateVector2D(rear_axle_longitudinal_velocity, rear_axle_lateral_velocity),
            rear_axle_acceleration_2d=StateVector2D(longitudinal_acceleration, lateral_acceleration),
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            tire_steering_rate=tire_steering_rate,
        )
