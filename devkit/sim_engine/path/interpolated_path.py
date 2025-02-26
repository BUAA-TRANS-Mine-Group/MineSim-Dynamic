import logging
import math
from typing import List

import numpy as np
import scipy.interpolate as sp_interp
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from shapely import Point

from devkit.common.actor_state.state_representation import Point2D
from devkit.common.actor_state.state_representation import StateSE2
from devkit.common.actor_state.state_representation import ProgressStateSE2
from devkit.common.geometry.compute import AngularInterpolator
from devkit.sim_engine.path.path import AbstractPath

logger = logging.getLogger(__name__)


def _get_heading(pt1: Point, pt2: Point) -> float:
    """
    Computes the angle between the line formed by two points and the x-axis.
    :param pt1: origin point.
    :param pt2: end point.
    :return: [rad] resulting angle.
    """
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.atan2(y_diff, x_diff)


class InterpolatedPath(AbstractPath):
    """A path that is interpolated from a list of points."""

    def __init__(self, path: List[ProgressStateSE2]):
        """
        Constructor of InterpolatedPath. 插值路径的构造函数。
        - 修改为3次样条曲线进行插值，来平滑路径； nuplan 原先为 线性插值，不方便计算曲率；

        :param path: List of states creating a path.
            The path has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
            用于创建路径的 ProgressStateSE2 列表。要求路径点至少包含两个，否则会抛出异常。
        """
        assert len(path) > 1, "Path has to has more than 1 element!"
        self._path = path

        # 初始化样条插值用于曲率计算：重新整理路径点为插值数， 用于插值 Re-arrange to arrays for interpolation
        progress = [point.progress for point in self._path]
        x = [p.x for p in self._path]
        y = [p.y for p in self._path]

        # 确保样条阶数不超过数据点数量允许的范围；默认使用3次样条曲线
        k = min(3, len(progress) - 1)
        self._spline_x = UnivariateSpline(x=progress, y=x, k=k, s=0)
        self._spline_y = UnivariateSpline(x=progress, y=y, k=k, s=0)

        # !创建线性插值函数；使用线性插值; 存储线性插值状态（进度、x、y） ; 默认 kind = "linear"
        self._function_interp_linear = sp_interp.interp1d(x=progress, y=np.array([[p.progress, p.x, p.y] for p in path]), axis=0, kind="linear")
        # 创建角度插值器;存储角度插值状态（heading）
        self._angular_interpolator = AngularInterpolator(states=progress, angular_states=np.array([[p.heading] for p in path]))

        # 限速相关属性
        self._speed_limits: List[float] = None
        self._speed_interp: sp_interp.interp1d = None
        self._curvatures: List[float] = None
        self._curvatures_interp: List[float] = None

    def get_start_progress(self) -> float:
        """Inherited, see superclass. 获取路径起点的进度值。"""
        return self._path[0].progress  # type: ignore

    def get_end_progress(self) -> float:
        """Inherited, see superclass.获取路径终点的进度值。"""
        return self._path[-1].progress  # type: ignore

    def get_state_at_progress(self, progress: float) -> ProgressStateSE2:
        """Inherited, see superclass.
        返回给定进度处的路径点状态。

        :param progress: 查询的进度值。
        :return: ProgressStateSE2 对象。
        """
        self._assert_progress(progress)  # 检查进度值是否在范围内

        # 通过插值获取线性和角度状态
        linear_states = list(self._function_interp_linear(progress))
        angular_states = list(self._angular_interpolator.interpolate(progress))

        # 将插值结果反序列化为 ProgressStateSE2
        return ProgressStateSE2.deserialize(linear_states + angular_states)

    def get_sampled_path(self) -> List[ProgressStateSE2]:
        """Inherited, see superclass. 返回路径中的采样点列表。"""
        return self._path

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """Calculates the nearest arc length for a given point.
        计算指定点到路径上最近点的弧长（进度）。

        :param point: 输入的位置点。
        :return: 距离最近的路径点的进度值。
        """
        distances = [np.hypot(point.x - waypoint.x, point.y - waypoint.y) for waypoint in self._path]
        nearest_index = np.argmin(distances)
        return self._path[nearest_index].progress  # 返回最近点的进度值

    def get_nearest_pose_from_position(self, point: Point2D) -> tuple[StateSE2, float]:
        """Finds the nearest pose on the path based on a given position.
        根据输入的位置点，在路径上找到最近的姿态（位置 + 航向）。

        :param point: 输入的位置点。
        :return: 路径上最接近的 `StateSE2` 对象。 + arc_length progress 返回最近点的进度值
        """
        # 获取最近点的弧长（进度）
        arc_length = self.get_nearest_arc_length_from_position(point)

        # Get interpolated positions at arc length and offset for heading estimation 在最近弧长处获取路径状态
        state1 = self.get_state_at_progress(arc_length)
        heading_offset = 0.5  # Distance offset to estimate heading 设置用于计算航向的偏移距离
        # 获取偏移位置处的状态，用于计算航向
        state2 = self.get_state_at_progress(min(arc_length + heading_offset, self.get_end_progress()))

        # Handle edge case if state1 and state2 are the same (e.g., end of path)  处理特殊情况：如果 state1 和 state2 是同一点（例如路径末端）
        if state1.x == state2.x and state1.y == state2.y:
            state2 = self.get_state_at_progress(max(arc_length - heading_offset, self.get_start_progress()))  # 尝试向前取点计算航向

        # Calculate heading between two points
        heading = _get_heading(Point(state1.x, state1.y), Point(state2.x, state2.y))

        return StateSE2(state1.x, state1.y, heading), arc_length

    def _assert_progress(self, progress: float) -> None:
        """Check if queried progress is within bounds 检查给定的进度值是否在路径范围内。"""
        start_progress = self.get_start_progress()
        end_progress = self.get_end_progress()
        assert start_progress <= progress <= end_progress, f"Progress exceeds path! " f"{start_progress} <= {progress} <= {end_progress}"

    def compute_speed_limits(self, max_lateral_accel: float, max_speed: float = 25.0, min_speed: float = 2.5) -> None:
        """
        根据路径曲率和车辆参数计算限速值。 新增曲率平滑处理；

        Args:
            max_lateral_accel (float): 最大允许横向加速度（m/s²）
            max_speed (float): 车辆限速的最大速度（m/s）; 矿区乘用车最大速度 90Km/h = 25m/s;
            min_speed (float): 车辆限速的最小速度（m/s）; 可以按照 最大速度 * 0.1 取值弯道最小限速


        1. **曲率计算**：
            - 使用 `UnivariateSpline` 对路径点进行插值，计算一阶和二阶导数。
            - 根据导数计算曲率，公式为：
                $$ \kappa = \frac{|dx/ds \cdot d^2y/ds^2 - dy/ds \cdot d^2x/ds^2|}{(dx/ds^2 + dy/ds^2)^{1.5}} $$

        2. **限速逻辑**：
            - 横向加速度约束：通过 `v = sqrt(a_max / κ)` 计算曲率相关限速。
            - 限速范围约束：限制速度在 `[min_speed, max_speed]` 之间。
            - 直道处理：曲率接近零时直接采用最大速度。

        3. **插值优化**：
            - 使用线性插值生成限速函数 `_speed_interp`，允许在路径范围内外插值。
            - 插值填充值设置为路径起点和终点的限速值，避免不合理外推。

        4. **曲率平滑流程**：
            - **原始计算**：保留原始曲率计算逻辑
            - **Savitzky-Golay滤波**：
                - 自动调整窗口长度（5点或数据长度）
                - 保证窗口长度为奇数
                - 使用二次多项式拟合
            - **非负处理**：`np.maximum(smoothed_curvatures, 0.0)`
        """
        raw_curvatures = []
        for point in self._path:
            s = point.progress
            dx_ds = self._spline_x.derivative(1)(s)
            dy_ds = self._spline_y.derivative(1)(s)
            d2x_ds2 = self._spline_x.derivative(2)(s)
            d2y_ds2 = self._spline_y.derivative(2)(s)

            # 曲率公式
            denominator = (dx_ds**2 + dy_ds**2) ** 1.5
            if denominator < 1e-6:
                curvature = 0.0
            else:
                curvature = abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / denominator
            raw_curvatures.append(curvature)

        # Savitzky-Golay滤波平滑
        try:
            window_length = min(5, len(raw_curvatures))  # 窗口长度取5或数据长度
            window_length = window_length if window_length % 2 == 1 else window_length - 1  # 保证奇数
            smoothed_curvatures = savgol_filter(raw_curvatures, window_length=window_length, polyorder=2, mode="nearest")  # 二次多项式拟合
            # 确保曲率非负
            curvatures = np.maximum(smoothed_curvatures, 0.0).tolist()
        except ValueError as e:
            logger.warning(f"#log# 曲率平滑失败，使用原始曲率: {str(e)}")
            curvatures = raw_curvatures
        self._raw_curvatures = raw_curvatures  # 新增原始曲率存储
        self._curvatures = curvatures  # 平滑后曲率

        # 计算限速
        speed_limits = []
        for curvature in curvatures:
            if curvature < 1e-6:  # stright road
                speed = max_speed
            else:  # curve road
                speed = math.sqrt(max_lateral_accel / curvature)
                speed = max(min_speed, min(speed, max_speed))
            speed_limits.append(speed)

        # 存储限速并创建插值器
        self._speed_limits = speed_limits
        progress_list = [p.progress for p in self._path]
        self._speed_interp = sp_interp.interp1d(
            x=progress_list, y=speed_limits, kind="linear", bounds_error=False, fill_value=(speed_limits[0], speed_limits[-1])
        )

    def get_curvature_at_progress(self, progress: float, use_smoothed: bool = True) -> float:
        """新增曲率查询接口"""
        if use_smoothed:
            data = self._curvatures
        else:
            data = self._raw_curvatures

        # self._curvatures_interp = sp_interp.interp1d(
        #     x=progress_list, y=curvatures, kind="linear", bounds_error=False, fill_value=(curvatures[0], curvatures[-1])
        # )
        return float(
            sp_interp.interp1d([p.progress for p in self._path], data, kind="linear", bounds_error=False, fill_value=(data[0], data[-1]))(progress)
        )

    def get_curvature_list(self, use_smoothed: bool = True) -> List[float]:
        """获取曲率数据列表"""
        if use_smoothed:
            return self._curvatures
        return self._raw_curvatures

    def get_speed_limit_at_progress(self, progress: float) -> float:
        """
        返回给定进度处的限速值。

        Args:
            progress (float): 路径进度值。
        Returns:
            float: 限速值（m/s）。
        """
        if self._speed_interp is None:
            raise RuntimeError("Speed limits not computed. Call compute_speed_limits first.")
        return float(self._speed_interp(progress))
