from __future__ import annotations

from typing import List, Optional

from devkit.common.actor_state.agent_state import AgentState
from devkit.common.actor_state.agent_temporal_state import AgentTemporalState
from devkit.common.actor_state.oriented_box import OrientedBox
from devkit.common.actor_state.scene_object import SceneObjectMetadata
from devkit.common.actor_state.state_representation import StateVector2D, TimePoint
from devkit.common.actor_state.tracked_objects_types import TrackedObjectType
from devkit.common.trajectory.predicted_trajectory import PredictedTrajectory


class Agent(AgentTemporalState, AgentState):
    """
    AgentState with future and past trajectory.
    """

    def __init__(
        self,
        tracked_object_type: TrackedObjectType,
        oriented_box: OrientedBox,
        velocity: StateVector2D,
        metadata: SceneObjectMetadata,
        angular_velocity: Optional[float] = None,
        acceleration: Optional[StateVector2D] = None,
        predictions: Optional[List[PredictedTrajectory]] = None,
        past_trajectory: Optional[PredictedTrajectory] = None,
    ):
        """
        场景中Agent的表示（车辆、行人、骑自行车的人和通用对象）。

        Representation of an Agent in the scene (Vehicles, Pedestrians, Bicyclists and GenericObjects).
        :param tracked_object_type: Type of the current agent.
        :param oriented_box: Geometrical representation of the Agent.
        :param velocity: Velocity (vectorial) of Agent. 车辆几何中心的速度，纵向和横向速度分量。【 右手坐标系，x-纵向 向前为正，y-横向 向左为正】
        :param metadata: Agent's metadata.
        :param angular_velocity: The scalar angular velocity of the agent, if available.
        :param predictions: Optional list of (possibly multiple) predicted trajectories.
        :param past_trajectory: Optional past trajectory of this agent.
        """
        AgentTemporalState.__init__(
            self,
            initial_time_stamp=TimePoint(metadata.timestamp_us),
            predictions=predictions,
            past_trajectory=past_trajectory,
        )
        AgentState.__init__(
            self,
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            metadata=metadata,
            velocity=velocity,
            angular_velocity=angular_velocity,
            acceleration=acceleration,  # 车辆几何中心的加速度，纵向和横向加速度分量。【 右手坐标系，x-纵向 向前为正，y-横向 向左为正】
        )

    @classmethod
    def from_agent_state(cls, agent: AgentState) -> Agent:
        """
        Create Agent from AgentState.
        :param agent: input single agent state.
        :return: Agent with None for future and past trajectory.
        """
        return cls(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=agent.box,
            velocity=agent.velocity,
            metadata=agent.metadata,
            angular_velocity=agent.angular_velocity,
            predictions=None,
            past_trajectory=None,
        )
