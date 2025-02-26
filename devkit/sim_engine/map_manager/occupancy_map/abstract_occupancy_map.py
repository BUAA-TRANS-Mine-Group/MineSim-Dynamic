from __future__ import annotations

import abc
from typing import List, Tuple, Union

from shapely.geometry import LineString, Polygon

Geometry = Union[Polygon, LineString]


class OccupancyMap(abc.ABC):
    """
    A class for handling spatial relationships between geometries. The two main functionalities are
    1. collision checking
    2. querying nearest geometry

    用于处理几何图形之间的空间关系的类。两个主要功能是
    1. 碰撞检查
    2. 查询最近的几何
    """

    @abc.abstractmethod
    def get_nearest_entry_to(self, geometry_id: str) -> Tuple[str, Geometry, float]:
        """
        Returns the row who's geometry is the nearest to the queried one
        :param geometry_id: id of the queried geometry
        :return: nearest geometry, corresponding ID, and distance to nearest geometry
        @raises AssertionError if the occupancy does not contain geometry_id

        返回其几何图形最接近查询的行
        :param geometry_id：查询到的几何体的 ID
        ：return：最近的几何体、相应的 ID 和到最近几何体的距离
        @raises AssertionError 如果占用不包含 geometry_id
        """
        pass

    @abc.abstractmethod
    def intersects(self, geometry: Geometry) -> OccupancyMap:
        """
        Returns a new occupancy map containing all geometries that intersects the given geometry
        :param geometry: geometry to check for intersection

        返回一个新的 occupancy map，其中包含与给定几何相交的所有几何
        :p aram geometry：要检查交集的几何体
        """
        pass

    @abc.abstractmethod
    def insert(self, geometry_id: str, geometry: Geometry) -> None:
        """
        Inserts a geometry to the occupancy map
        :param geometry_id: id of the geometry
        :param geometry: geometry to be inserted

        将几何图形插入到占用地图
        :p aram geometry_id：几何体的 ID
        :p ARAM 几何体：要插入的几何体
        """
        pass

    @abc.abstractmethod
    def get(self, geometry_id: str) -> Geometry:
        """
        Gets the geometry with the corresponding geometry_id
        :param geometry_id: the id corresponding to the geometry

        获取具有相应 geometry_id 的几何图形
        :p aram geometry_id：几何体对应的 ID
        """
        pass

    @abc.abstractmethod
    def set(self, geometry_id: str, geometry: Geometry) -> None:
        """
        Set a specific geometry with a new one
        :param geometry_id: the id corresponding to the geometry
        :param geometry: the new geometry to set
        """
        pass

    @abc.abstractmethod
    def get_all_ids(self) -> List[str]:
        """
        Return ids of all geometries in the occupancy map
        :return: all ids as a list of strings
        """

    @abc.abstractmethod
    def get_all_geometries(self) -> List[Geometry]:
        """
        Return all geometries in the occupancy map
        :return: all geometries as a list of Geometry
        """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        :return: the number of entries in occupancy map
        """
        pass

    @abc.abstractmethod
    def is_empty(self) -> bool:
        """
        :return: true if the occupancy map is empty
        """
        pass

    @abc.abstractmethod
    def contains(self, geometry_id: str) -> bool:
        """
        :return: true if a geometry with the given id exists in the occupancy map
        ：return： true，如果占用地图中存在具有给定 ID 的几何图形
        """
        pass

    @abc.abstractmethod
    def remove(self, geometry_ids: List[str]) -> None:
        """
        Removes the geometries with the corresponding geometry_ids
        :param geometry_ids: the ids corresponding to the geometries
        """
        pass

    def __len__(self) -> int:
        """Support len() as returning the number of entries in the map."""
        return self.size
