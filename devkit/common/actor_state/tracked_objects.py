from __future__ import annotations

from functools import cached_property
from typing import Dict, Iterable, List, Optional, Tuple, Union

from devkit.common.actor_state.agent import Agent
from devkit.common.actor_state.agent_temporal_state import AgentTemporalState
from devkit.common.actor_state.oriented_box import OrientedBox
from devkit.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from devkit.common.actor_state.static_object import StaticObject
from devkit.common.actor_state.tracked_objects_types import TrackedObjectType
from devkit.common.actor_state.tracked_objects_types import AGENT_TYPES
from devkit.common.actor_state.tracked_objects_types import SMART_AGENT_TYPES
from devkit.common.actor_state.tracked_objects_types import STATIC_OBJECT_TYPES

# 定义一个联合类型，表示可能的跟踪对象
TrackedObject = Union[Agent, StaticObject, SceneObject, AgentTemporalState]


class TrackedObjects:
    """表示被跟踪对象的类，一个SceneObjects的集合
    Class representing tracked objects, a collection of SceneObjects"""

    def __init__(self, tracked_objects: Optional[List[TrackedObject]] = None):
        """初始化TrackedObjects实例

        :param tracked_objects: 被跟踪对象的列表
        :param tracked_objects: List of tracked objects
        """
        tracked_objects = tracked_objects if tracked_objects is not None else []
        # 按照对象类型的值对被跟踪对象进行排序 ; 使用 key=lambda agent: agent.tracked_object_type.value 按照对象类型排序
        self.tracked_objects = sorted(tracked_objects, key=lambda agent: agent.tracked_object_type.value)

    def __iter__(self) -> Iterable[TrackedObject]:
        """当迭代此对象时，返回被跟踪对象的迭代器。
        When iterating return the tracked objects."""
        return iter(self.tracked_objects)

    # @classmethod
    # def from_oriented_boxes(cls, boxes: List[OrientedBox]) -> TrackedObjects:
    #     """When iterating return the tracked objects."""
    #     scene_objects = [
    #         SceneObject(
    #             TrackedObjectType.GENERIC_OBJECT,
    #             box,
    #             SceneObjectMetadata(timestamp_us=i, token=str(i), track_token=None, track_id=None),
    #         )
    #         for i, box in enumerate(boxes)
    #     ]
    #     return TrackedObjects(scene_objects)

    @cached_property
    def _ranges_per_type(self) -> Dict[TrackedObjectType, Tuple[int, int]]:
        """返回每种TrackedObjectType在tracked_objects列表中的起始和结束索引范围。
        该范围会被缓存以供后续调用使用。

        :return: 一个字典，键为TrackedObjectType，值为(start_index, end_index)的元组
        Returns the start and end index of the range of agents for each agent type
        in the list of agents (sorted by agent type). The ranges are cached for subsequent calls.
        """
        ranges_per_type: Dict[TrackedObjectType, Tuple[int, int]] = {}

        if self.tracked_objects:
            # 初始化第一个对象的类型
            last_agent_type = self.tracked_objects[0].tracked_object_type
            start_range = 0
            end_range = len(self.tracked_objects)

            for idx, agent in enumerate(self.tracked_objects):
                # 如果当前对象的类型与上一个不同，则记录上一个类型的范围
                if agent.tracked_object_type is not last_agent_type:
                    ranges_per_type[last_agent_type] = (start_range, idx)
                    start_range = idx
                    last_agent_type = agent.tracked_object_type
            # 记录最后一种类型的范围
            ranges_per_type[last_agent_type] = (start_range, end_range)

            # 对于未出现的TrackedObjectType，设置其范围为(end_range, end_range)
            ranges_per_type.update({agent_type: (end_range, end_range) for agent_type in TrackedObjectType if agent_type not in ranges_per_type})

        return ranges_per_type

    def get_tracked_objects_of_type(self, tracked_object_type: TrackedObjectType) -> List[TrackedObject]:
        """获取指定类型的被跟踪对象子列表。

        :param tracked_object_type: 要查询的TrackedObjectType
        :return: 指定类型的被跟踪对象列表。如果类型无效，则返回空列表。

        Gets the sublist of agents of a particular TrackedObjectType
        :param tracked_object_type: The query TrackedObjectType
        :return: List of the present agents of the query type. Throws an error if the key is invalid.
        """
        if tracked_object_type in self._ranges_per_type:
            start_idx, end_idx = self._ranges_per_type[tracked_object_type]
            return self.tracked_objects[start_idx:end_idx]

        else:
            # 如果查询的类型在_ranges_per_type中不存在，则返回空列表 There are no objects of the queried type
            return []

    def get_agents(self) -> List[Agent]:
        """获取所有类型为Agent的被跟踪对象。

        :return: Agent对象的列表
        Getter for the tracked objects which are Agents
        :return: list of Agents
        """
        agents = []
        for agent_type in AGENT_TYPES:
            agents.extend(self.get_tracked_objects_of_type(agent_type))
        return agents
    
    def get_dynamic_smart_agents(self) -> List[Agent]:
        """获取所有类型为smart Agent的被跟踪对象。

        :return: Smart Agent 对象的列表
        Getter for the tracked objects which are Agents
        :return: list of SmartAgents
        """
        smart_agents = []
        for agent_type in SMART_AGENT_TYPES:
            smart_agents.extend(self.get_tracked_objects_of_type(agent_type))
        return smart_agents
    

    def get_static_objects(self) -> List[StaticObject]:
        """获取所有类型为StaticObject的被跟踪对象。

        :return: StaticObject对象的列表
        Getter for the tracked objects which are StaticObjects
        :return: list of StaticObjects
        """
        static_objects = []
        for static_object_type in STATIC_OBJECT_TYPES:
            static_objects.extend(self.get_tracked_objects_of_type(static_object_type))
        return static_objects

    def __len__(self) -> int:
        """获取被跟踪对象的总数量。

        :return: 被跟踪对象的数量
        :return: The number of tracked objects in the class
        """
        return len(self.tracked_objects)

    def get_tracked_objects_of_types(self, tracked_object_types: List[TrackedObjectType]) -> List[TrackedObject]:
        """获取多个指定类型的被跟踪对象子列表。

        :param tracked_object_types: 要查询的TrackedObjectType列表
        :return: 指定类型的被跟踪对象列表。如果某类型无效，则忽略该类型。

        Gets the sublist of agents of particular TrackedObjectTypes
        :param tracked_object_types: The query TrackedObjectTypes
        :return: List of the present agents of the query types. Throws an error if the key is invalid.
        """
        open_loop_tracked_objects = []
        for _type in tracked_object_types:
            open_loop_tracked_objects.extend(self.get_tracked_objects_of_type(_type))

        return open_loop_tracked_objects
