from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

from shapely.ops import nearest_points
from shapely.strtree import STRtree

from devkit.common.actor_state.scene_object import SceneObject
from devkit.sim_engine.map_manager.occupancy_map.abstract_occupancy_map import OccupancyMap
from devkit.sim_engine.map_manager.occupancy_map.abstract_occupancy_map import Geometry

# ignoring shapely RuntimeWarning: invalid value encountered in line_locate_point
warnings.filterwarnings(action="ignore", message="#log# (.|\n)*invalid value encountered in line_locate_point(.|\n)*")
GeometryMap = Dict[str, Geometry]


class STRTreeOccupancyMap(OccupancyMap):
    """
    OccupancyMap using an SR-tree to support efficient get-nearest queries.

    ### Nuplan 中的作用：
        记录 occupancy_map_radius 范围内所有的 感知检测 agent 的 box 几何，并且基于 from shapely.strtree import STRtree 高效查询；
    ### STRtree：
    - `STRtree` 主要用于以下场景：
        - 空间查询（Spatial Queries）：例如，查找与某个几何对象相交的所有对象。
        - 最近邻查找（Nearest Neighbor Search）：查找距离给定点最近的几何对象。
        - 批量查找（Bulk Queries）：同时对多个查询对象进行空间检索。
    - **3. STRtree 主要方法**
    | 方法                      | 描述                                   | 示例                                        |
    | ------------------------ | -------------------------------------- | ------------------------------------------- |
    | `query(geometry)`        | 查找与给定几何对象相交的对象                | `tree.query(Point(2,2))`                    |
    | `nearest(geometry)`      | 查找距离给定几何对象最近的对象              | `tree.nearest(Point(6,6))`                  |
    | `geometries`             | 返回索引中存储的所有几何对象                | `all_geoms = tree.geometries`               |
    | `query_bulk(geometries)` | 批量查询多个几何对象，并返回匹配的索引        | `tree.query_bulk([Point(2,2), Point(5,5)])` |
    """

    def __init__(self, geom_map: GeometryMap):
        """
        Constructor of STRTreeOccupancyMap.
        :param geom_map: underlying geometries for occupancy map.
        """
        self._geom_map: GeometryMap = geom_map

    def get_nearest_entry_to(self, geometry_id: str) -> Tuple[str, Geometry, float]:
        """Inherited, see superclass."""
        assert self.contains(geometry_id), "This occupancy map does not contain given geometry id"

        strtree, index_by_id = self._build_strtree(ignore_id=geometry_id)
        nearest_index = strtree.nearest(self.get(geometry_id=geometry_id))
        nearest = strtree.geometries.take(nearest_index)
        p1, p2 = nearest_points(self.get(geometry_id), nearest)
        return index_by_id[id(nearest)], nearest, p1.distance(p2)

    def intersects(self, geometry: Geometry) -> OccupancyMap:
        """Inherited, see superclass."""
        strtree, index_by_id = self._build_strtree()
        indices = strtree.query(geometry)
        new_occ_map = STRTreeOccupancyMap(
            geom_map={index_by_id[id(geom)]: geom for geom in strtree.geometries.take(indices) if geom.intersects(geometry)}
        )

        return new_occ_map
        # return STRTreeOccupancyMap({index_by_id[id(geom)]: geom for geom in strtree.geometries.take(indices) if geom.intersects(`1``)})

    def insert(self, geometry_id: str, geometry: Geometry) -> None:
        """Inherited, see superclass."""
        self._geom_map[geometry_id] = geometry

    def get(self, geometry_id: str) -> Geometry:
        """Inherited, see superclass."""
        return self._geom_map[geometry_id]

    def set(self, geometry_id: str, geometry: Geometry) -> None:
        """Inherited, see superclass."""
        self._geom_map[geometry_id] = geometry

    def get_all_ids(self) -> List[str]:
        """Inherited, see superclass."""
        return list(self._geom_map.keys())

    def get_all_geometries(self) -> List[Geometry]:
        """Inherited, see superclass."""
        return list(self._geom_map.values())

    @property
    def size(self) -> int:
        """Inherited, see superclass."""
        return len(self._geom_map)

    def is_empty(self) -> bool:
        """Inherited, see superclass."""
        return not self._geom_map

    def contains(self, geometry_id: str) -> bool:
        """Inherited, see superclass."""
        return geometry_id in self._geom_map

    def remove(self, geometry_ids: List[str]) -> None:
        """Remove geometries from the occupancy map by ids."""
        for id in geometry_ids:
            assert id in self._geom_map, "Geometry does not exist in occupancy map"
            self._geom_map.pop(id)

    def _get_other_geometries(self, ignore_id: str) -> GeometryMap:
        """返回除 ignore_id 指定的几何之外的所有几何
        Returns all geometries as except for one specified by ignore_id

        :param ignore_id: the key corresponding to the geometry to be skipped
        :return: GeometryMap
        """
        return {geom_id: geom for geom_id, geom in self._geom_map.items() if geom_id not in ignore_id}

    def _build_strtree(self, ignore_id: Optional[str] = None) -> Tuple[STRtree, Dict[int, str]]:
        """
        Constructs an STRTree from the geometries stored in the geometry map. Additionally, returns a index-id
        mapping to the original keys of the geometries. Has the option to build a tree omitting on geometry

        :param ignore_id: the key corresponding to the geometry to be skipped
        :return: STRTree containing the values of _geom_map, index mapping to the original keys

        根据几何图形中存储的几何图形构建 STRTree。此外，还会返回几何图形原始键的索引 ID。可选择不在几何体上构建树
        :param ignore_id：要跳过的几何体对应的键值
        :return： 包含 _geom_map 值的 STRTree，索引映射到原始键
        {geom_id:geom}; e.g. {"object-1": 车辆Box polygon}

        """
        if ignore_id is not None:
            temp_geom_map = self._get_other_geometries(ignore_id)
        else:
            temp_geom_map = self._geom_map

        strtree = STRtree(list(temp_geom_map.values()))
        index_by_id = {id(geom): geom_id for geom_id, geom in temp_geom_map.items()}

        return strtree, index_by_id


class STRTreeOccupancyMapFactory:
    """
    Factory for STRTreeOccupancyMap.
    """

    @staticmethod
    def get_from_boxes(scene_objects: List[SceneObject]) -> OccupancyMap:
        """
        Builds an STRTreeOccupancyMap from a list of SceneObject. The underlying dictionary will have the format
          key    : value
        return {geom_id: geom for geom_id, geom in self._geom_map.items() if ge
          token1 : [Polygon, LineString]
          token2 : [Polygon, LineString]
        The polygon is derived from the corners of each SceneObject
        :param scene_objects: list of SceneObject to be converted
        :return: STRTreeOccupancyMap
        """
        return STRTreeOccupancyMap(
            {scene_object.track_token: scene_object.box.geometry for scene_object in scene_objects if scene_object.track_token is not None}
        )

    @staticmethod
    def get_from_geometry(geometries: List[Geometry], geometry_ids: Optional[List[str]] = None) -> OccupancyMap:
        """
        Builds an STRTreeOccupancyMap from a list of Geometry. The underlying dictionary will have the format
          key    : value
          token1 : [Polygon, LineString]
          token2 : [Polygon, LineString]]
        :param geometries: list of [Polygon, LineString]
        :param geometry_ids: list of corresponding ids
        :return: STRTreeOccupancyMap
        """
        if geometry_ids is None:
            return STRTreeOccupancyMap({str(geom_id): geom for geom_id, geom in enumerate(geometries)})

        return STRTreeOccupancyMap({str(geom_id): geom for geom_id, geom in zip(geometry_ids, geometries)})
