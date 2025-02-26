import math
from typing import List, Union
import abc
import shapely
from devkit.common.actor_state.state_representation import StateSE2


class PlanningProblemBase(abc.ABC):
    """
    基础规划问题类，包含一个场景名称。
    """

    def __init__(self, scenario_name: str):
        """
        初始化一个规划问题基础类实例。

        Args:
            scenario_name (str): 场景名称。
        """
        self.scenario_name = scenario_name


class PlanningProblemGoalTask(PlanningProblemBase):
    """
    - Goal task 1 : 车辆定位点通过目标多边形;

    Attributes:
        goal_range_polygon: 目标范围的多边形表示，定义了车辆定位点的区域。
    """

    def __init__(self, scenario_name: str, goal_range_xs: List[float], goal_range_ys: List[float]):
        """
        scenario_name (str): 场景名称。
        goal_range_xs (List[float]): 目标范围的 x 坐标。
        goal_range_ys (List[float]): 目标范围的 y 坐标。
        """
        super().__init__(scenario_name=scenario_name)

        if len(goal_range_xs) != len(goal_range_ys):
            raise ValueError("The length of goal_range_xs must be equal to the length of goal_range_ys.")

        self.goal_range_polygon = shapely.geometry.Polygon(zip(goal_range_xs, goal_range_ys))

    @property
    def get_polygon_center_point(self) -> shapely.geometry.Point:
        """
        获取目标多边形的中心点。

        Returns:
            shapely.geometry.Point: 目标多边形的中心点坐标。
        """
        return self.goal_range_polygon.centroid

    def is_point_inside_polygon(self, check_point: shapely.geometry.Point) -> bool:
        """
        检查给定点是否在目标多边形内。

        Args:
            check_point (shapely.geometry.Point): 需要检查的点。

        Returns:
            bool: 如果点在目标多边形内，返回 True；否则返回 False。
        """
        return self.goal_range_polygon.contains(check_point)


class PlanningProblemGoalTaskFinalPose(PlanningProblemGoalTask):
    """
    - Goal task 1 : 车辆定位点通过目标多边形;
    - Goal task 2 : 要求车辆通过目标多边形过程， 车头朝向与道路朝向相同；

    Attributes:
        final_pose: 车辆的最终姿态，包含位置和朝向信息。
    """

    def __init__(self, scenario_name: str, goal_range_xs: List[float], goal_range_ys: List[float], final_pose: StateSE2):
        """
        scenario_name (str): 场景名称。
        goal_range_xs (List[float]): 目标范围的 x 坐标。
        goal_range_ys (List[float]): 目标范围的 y 坐标。
        final_pose (StateSE2): 车辆的最终姿态，包含位置和朝向。
        """
        super().__init__(scenario_name=scenario_name, goal_range_xs=goal_range_xs, goal_range_ys=goal_range_ys)
        self.final_pose = final_pose


# 定义规划问题任务的联合类型，可以是目标任务或者最终定位目标任务。
PlanningProblemGoalTasks = Union[PlanningProblemGoalTask, PlanningProblemGoalTaskFinalPose]
