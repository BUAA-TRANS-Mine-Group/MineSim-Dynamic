import numpy as np
from shapely.geometry import Polygon


class Vehicle(object):
    """class that stores vehicle parameters
    Used internally by the Planner class
    """

    def __init__(
        self,
        vehicle_l,
        vehicle_w,
        vehicle_h: float = 1.5,
        safety_factor: float = 1.0,
        longitudinal_v_max: float = 24.0,
        longitudinal_a_max: float = 3.0,
        longitudinal_jerk_max: float = 8.0,
        lateral_accel_max: float = 0.5,
        lateral_jerk_max: float = 6.0,
        centripetal_accel_max: float = 1.0,
        shape: dict = None,
        constraints: dict = None,
    ):
        # 原文件的参数
        self.shape = shape
        self.constraints = constraints

        # vehicle dimensions
        self.l: float = vehicle_l * safety_factor
        self.w: float = vehicle_w * safety_factor
        self.h: float = vehicle_h * safety_factor
        self.bbox_size: np.ndarray = np.array([self.l, self.w, self.h])  # bounding box size [m]

        # footprint coordinates (in clockwise direction)  足迹坐标(顺时针方向)
        self.corners: list[tuple[float, float]] = [
            (self.l / 2, self.w / 2),  # front left corner's coordinates in box_center frame [m]
            (self.l / 2, -self.w / 2),  # front right corner's coordinates in box_center frame [m]
            (-self.l / 2, -self.w / 2),  # rear right corner's coordinates in box_center frame [m]
            (-self.l / 2, self.w / 2),  # rear left corner's coordinates in box_center frame [m]
            (self.l / 2, self.w / 2),  # front left corner's coordinates in box_center frame [m] (to enclose the polygon)
        ]
        self.polygon: Polygon = Polygon(self.corners)

        # kinematic parameters
        # self.a = vehicle_params.a                                               # distance from base_link to the CoG [m]
        # self.b = vehicle_params.b                                               # distance from the CoG to front_link [m]
        # self.L = self.a + self.b                                                # wheelbase distance [m]
        # self.T_f = vehicle_params.T_f                                           # front track w [m]
        # self.T_f = vehicle_params.T_r                                           # rear track w [m]
        self.max_speed = longitudinal_v_max  # maximum speed [m/s]
        self.max_accel = longitudinal_a_max  # maximum acceleration [m/ss]
        self.max_deccel = -longitudinal_a_max  # maximum decceleration [m/ss]
        self.max_jerk = longitudinal_jerk_max
        # self.max_steering_angle = vehicle_params.steering.max                   # maximum steering angle [rad]
        # self.max_steering_rate = vehicle_params.steering.v_max                  # maximum steering rate [rad/s]
        # self.max_curvature = np.sin(self.max_steering_angle)/self.L             # maximum curvature [1/m]
        # self.max_kappa_d = vehicle_params.steering.kappa_dot_max                # maximum curvature change rate [1/m]
        # self.max_kappa_dd = vehicle_params.steering.kappa_dot_dot_max           # maximum curvature change rate rate [1/m]

        self.max_lateral_accel = lateral_accel_max  # 横向加减速度阈值
        self.max_lateral_jerk = lateral_jerk_max  # 侧向加加速度阈值
        self.max_centripetal_accel = centripetal_accel_max  # 最大的 转弯向心加速度

        # !向心加速度的计算公式为：
        # \[ a_c = \frac{v^2}{r} \]
        # 其中：
        # - \( a_c \) 是向心加速度（单位通常是米每平方秒,即 m/s^2）
        # - \( v \) 是你在转弯时的线速度（单位通常是米每秒,即 m/s）
        # - \( r \) 是转弯的半径（单位通常是米,即 m）

        # 在自动驾驶汽车领域,横向加速度和转弯向心加速度的区别主要体现在以下几个方面：

        ## 定义与物理含义：横向加速度主要描述的是汽车在水平方向上,垂直于行驶方向（即侧向）的加速度变化.这种加速度变化通常是由于车辆的转向,侧滑或道路曲率等因素引起的.
        ## 而转弯向心加速度,又称为法向加速度或向心加速度,是描述车辆在转弯时,由于向心力作用而产生的加速度,它始终指向转弯的圆心（或曲率中心）,与车辆的行驶方向垂直.

        ## 作用与影响：横向加速度对自动驾驶汽车的稳定性和操控性具有重要影响.在高速行驶或紧急避障等情况下,横向加速度的准确测量和控制对保证车辆安全和稳定至关重要.
        ## 而转弯向心加速度则主要影响车辆在转弯过程中的动力学性能,包括轮胎与地面的摩擦力,车辆的侧倾和翻滚等.

        ## 控制策略：在自动驾驶汽车的控制系统中,对横向加速度和转弯向心加速度的控制策略也有所不同.
        ## 对于横向加速度,控制系统需要通过调整转向角度,车速等参数来实现对车辆侧向运动的精确控制.
        ## 而对于转弯向心加速度,控制系统则需要根据车辆的转弯半径,车速等信息,计算出合适的向心力大小和方向,以保证车辆在转弯过程中的稳定性和安全性.
