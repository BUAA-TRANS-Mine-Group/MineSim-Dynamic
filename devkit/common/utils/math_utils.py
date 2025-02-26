# import math
import numpy as np
from scipy.spatial.transform import Rotation


def mps2kph(x):
    return x * 3.6


def kph2mps(x):
    return x / 3.6


def euler_to_quat(roll, pitch, yaw, degrees=False):
    """将欧拉角(roll , pitch , yaw)转换为四元数.欧拉角表示三维旋转 ,而四元数提供了一种更稳定的方式来表示旋转."""
    rot = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=degrees)
    return rot.as_quat()


def quate_to_euler(quat, degrees=False):
    """将四元数转换回欧拉角."""
    rot = Rotation.from_quat(quat)
    return rot.as_euler("xyz", degrees=degrees)


def yaw_to_quat(yaw, degrees=False):
    """只根据偏航角(yaw)创建四元数. 这在2D情况下(例如车辆在平面上的旋转)很有用."""
    rot = Rotation.from_euler("z", yaw, degrees=degrees)
    return rot.as_quat()


def quate_to_yaw(quat, degrees=False):
    """从四元数中提取偏航角."""
    rot = Rotation.from_quat(quat)
    euler = rot.as_euler("zyx", degrees=degrees)
    return euler[0]


def unify_angle_range(angle):
    """
    将输入角度(可以是单个数值或 NumPy 数组)标准化到 [-pi , pi] 的范围内.

    参数:
        angle (float or np.ndarray): 输入角度,以弧度为单位.可以是单个数值或 NumPy 数组.

    返回:
        np.ndarray: 标准化后的角度,范围在 [-pi , pi]内.如果输入是单个数值,则返回单个标准化角度数值.

    说明:
        这个函数使用 NumPy 的模运算和向量化操作来确保任何输入的角度值都被转换到一个标准的连续范围内.
        这对于角度的比较,差值计算等操作非常有用,并且能够高效处理单个数值或整个数组.
        优点:
    > 直观性和一致性：这个范围内的角度表示与数学和物理中常见的角度表示方式保持一致,使得角度的加减法运算直观且易于理解.
    > 简化计算：当涉及到角度差的计算时,使用 [-pi , pi] 范围可以避免额外的逻辑来处理角度跨越0点的情况.这简化了算法的实现,并可能减少编程错误.
    > 向量运算兼容性：在处理与方向有关的向量运算时,标准化的角度范围能确保计算结果的一致性和准确性.
    """
    # new_angle = angle
    # while new_angle > np.pi:
    #     new_angle -= 2 * np.pi
    # while new_angle < -np.pi:
    #     new_angle += 2 * np.pi
    # return new_angle
    normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return normalized_angle


# def clamp(value , lower_bound , upper_bound):
# """np.clip() 用于限制值的范围."""
#     return max(min(value , upper_bound) , lower_bound)
# np.clip(value , lower_bound , upper_bound)

# def distance(x1 , y1 , x2 , y2):
# """np.hypot() 用于计算两点之间的距离 ,"""
#     return ((x2 - x1) ** 2+(y2 - y1) ** 2) ** 0.5
# np.hypot()

# def pose_distance(a , b):
#     return distance(a.position.x , a.position.y , b.position.x , b.position.y)

# def magnitude(x , y , z):
#     return (x**2 + y**2 + z**2)**0.5

# def isLegal(x):
#     """检查一个数值是否是合法的(不是无穷大也不是 NaN).
#         这在处理实数和浮点数时很有用 ,尤其是在从传感器或计算过程中获取数据时.
#     """
#     return False if (math.isinf(x) or math.isnan(x)) else True
