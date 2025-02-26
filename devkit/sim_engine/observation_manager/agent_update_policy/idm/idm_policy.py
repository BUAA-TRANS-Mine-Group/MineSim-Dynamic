from math import sqrt
from typing import Any, List

import numpy as np
from scipy.integrate import odeint, solve_ivp

from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_states import IDMAgentState
from devkit.sim_engine.observation_manager.agent_update_policy.idm.idm_states import IDMLeadAgentState


class IDMPolicy:
    """
    An agent policy that describes the agent's behaviour w.r.t to a lead agent. The policy only controls the longitudinal states (progress, velocity) of the agent.
    This longitudinal states are used to propagate the agent along a given path.

    IDM（智能驾驶员模型）策略实现类
    - 核心功能：根据前车状态和自车参数，计算自车的纵向运动控制指令
    - 模型特点：综合考虑自由行驶和跟车场景，平衡行驶效率和安全性

    - 这个代码实现了一个基于智能驾驶模型（IDM）的策略，主要用于模拟车辆在自动驾驶仿真中的纵向行为。
      IDM（Intelligent Driver Model）是一种用于描述跟车行为的数学模型，能够模拟驾驶员在自由交通和跟车情境下的加速和减速行为。

    - IDMPolicy类提供了三种不同的方法来求解描述车辆纵向运动的微分方程，分别是前向欧拉法、`odeint`以及`solve_ivp`。
      这些方法可以在不同精度和性能要求下灵活选择，以模拟自动驾驶中车辆的跟车行为。

    - 车辆的 速度 和 加速度 都是沿着道路纵向的标量量，符号的意义是：
    """

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
    ):
        """
        Constructor for IDMPolicy

        :param target_velocity: Desired velocity in free traffic [m/s]
        :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
        :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
        :param accel_max: maximum acceleration [m/s^2]
        :param decel_max: maximum deceleration (positive value) [m/s^2]

        - `target_velocity`: 在自由交通下期望的速度。
        - `min_gap_to_lead_agent`: 与前车的最小安全距离。
        - `headway_time`: 与前车的期望时间间隔。
        - `accel_max`: 最大加速度。
        - `decel_max`: 最大减速度。

        初始化IDM策略参数

        参数说明：
        - target_velocity: 自由流期望速度（m/s）
          车辆在无前车干扰时希望保持的最高速度
          示例：高速公路场景可设为33.3m/s（约120km/h）

        - min_gap_to_lead_agent: 最小安全距离（m）
          静止状态下与前车保持的最小物理间距
          示例：典型值2-5米，取决于车辆制动性能

        - headway_time: 安全时距（秒）
          反映驾驶员跟车时的时间裕度，保持与前车的时间间隔
          示例：1.5秒时距意味着以120km/h行驶时保持50米间距

        - accel_max: 最大加速度（m/s²）
          车辆能达到的最大加速能力
          示例：普通轿车约1-3 m/s²，跑车可达4 m/s²

        - decel_max: 最大减速度（m/s²，取正值）
          紧急制动时的最大减速能力
          示例：普通制动约3-5 m/s²，紧急制动可达8-10 m/s²
        """
        # 参数校验
        assert target_velocity > 0, "目标速度必须为正数"
        assert min_gap_to_lead_agent >= 0, "最小安全距离不能为负"
        assert headway_time > 0, "安全时距必须为正数"
        assert accel_max > 0, "最大加速度必须为正数"
        assert decel_max > 0, "最大减速度必须为正数（输入正值）"

        # 参数绑定
        self._target_velocity = target_velocity
        self._min_gap_to_lead_agent = min_gap_to_lead_agent
        self._headway_time = headway_time
        self._accel_max = accel_max
        self._decel_max = decel_max

    @property
    def idm_params(self) -> List[float]:
        """Returns the policy parameters as a list; 获取IDM参数列表 [目标速度, 最小间距, 时距, 最大加速度, 最大减速度"""
        return [
            self._target_velocity,
            self._min_gap_to_lead_agent,
            self._headway_time,
            self._accel_max,
            self._decel_max,
        ]

    @property
    def target_velocity(self) -> float:
        """
        The policy's desired velocity in free traffic [m/s] 获取/设置目标速度（属性方式访问）
        :return: target velocity
        """
        return self._target_velocity

    @target_velocity.setter
    def target_velocity(self, target_velocity: float) -> None:
        """
        Sets the policy's desired velocity in free traffic [m/s] ; 设置目标速度时的参数校验
        """
        self._target_velocity = target_velocity
        assert target_velocity > 0, f"The target velocity must be greater than 0! {target_velocity} > 0"

    @property
    def headway_time(self) -> float:
        """获取期望的时间间隔。
        The policy's minimum possible time to the vehicle in front [s]
        :return: Desired time headway
        """
        return self._headway_time

    @property
    def decel_max(self) -> float:
        """获取最大减速度
        The policy's maximum deceleration (positive value) [m/s^2]
        :return: Maximum deceleration
        """
        return self._decel_max

    @staticmethod
    def idm_model(time_points: List[float], state_variables: List[float], lead_agent: List[float], params: List[float]) -> List[Any]:
        """
        IDM模型核心微分方程定义
        输入参数：
        - time_points: 时间点序列（积分用，此处未直接使用）
        - state_variables: 自车状态 [当前位置(m), 当前速度(m/s)]
        - lead_agent: 前车状态 [前车位置(m), 前车速度(m/s), 前车半长(m)]
        - params: IDM参数 [目标速度, 最小间距, 时距, 最大加速度, 最大减速度]

        输出：
        - 状态变化率列表 [速度变化, 加速度变化]

        数学模型：
        dx/dt = v  # 位置变化率即速度
        dv/dt = a * [1 - (v/v0)^δ - (s*/s)^2]  # 加速度由三部分组成：
          1. 自由行驶项：趋向目标速度的加速度
          2. 速度差项：与前车速度差异的影响
          3. 间距项：与前车间距的影响
        其中：
          s* = s0 + v*T + (v*Δv)/(2*sqrt(a*b))  # 期望安全距离
          s = x_lead - x_agent - vehicle_length  # 实际间距

        Defines the differential equations for IDM.

        :param state_variables: vector of the state variables:
                  state_variables = [x_agent: progress,
                                     v_agent: velocity]
        :param time_points: time A sequence of time points for which to solve for the state variables
        :param lead_agent: vector of the state variables for the lead vehicle:
                  lead_agent = [x_lead: progress,
                                v_lead: velocity,
                                l_r_lead: half length of the leading vehicle]
        :param params:vector of the parameters:
                  params = [target_velocity: desired velocity in free traffic,
                            min_gap_to_lead_agent: minimum relative distance to lead vehicle,
                            headway_time: desired time headway. The minimum possible time to the vehicle in front,
                            accel_max: maximum acceleration,
                            decel_max: maximum deceleration (positive value)]

        :return: system of differential equations
        """
        # state variables 解包输入参数
        x_agent, v_agent = state_variables  # 自车位置和速度
        x_lead, v_lead, l_r_lead = lead_agent  # 前车位置、速度、半长

        # parameters target_velocity, s0, T, a, b = params  # IDM参数
        target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max = params
        acceleration_exponent = 4  # Usually set to 4

        # 计算期望安全距离（IDM核心公式）
        # delta_v = v_agent - v_lead  # 速度差（自车-前车）
        # s_star = s0 + v_agent*T + (v_agent*delta_v)/(2*sqrt(a*b))
        s_star = min_gap_to_lead_agent + v_agent * headway_time + (v_agent * (v_agent - v_lead)) / (2 * sqrt(accel_max * decel_max))

        # 计算实际间距（考虑前车长度）
        # actual_gap = x_lead - x_agent - 2*l_r_lead  # 间距=前车位置-自车位置-前车全长
        # s_alpha = max(actual_gap, 1e-3)  # 防止除以零，保持最小间距
        s_alpha = max(x_lead - x_agent - l_r_lead, min_gap_to_lead_agent, 1e-3)  # clamp to avoid zero division  # 防止除以零，保持最小间距

        # differential equations
        # 计算加速度（IDM核心公式）
        free_road_term = (v_agent / target_velocity) ** acceleration_exponent  # 自由行驶项（δ=4）
        interaction_term = (s_star / s_alpha) ** 2  # 跟车交互项
        acceleration = accel_max * (1 - free_road_term - interaction_term)

        # 限制加速度范围 [-b, a]
        acceleration_clamped = np.clip(acceleration, -decel_max, accel_max)

        return [v_agent, acceleration_clamped]

    def solve_forward_euler_idm_policy(self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float) -> IDMAgentState:
        """
        Solves Solves an initial value problem for a system of ODEs using forward euler.
        This has the benefit of being differentiable

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :return: solution to the differential equations

        前向欧拉法求解IDM微分方程
        特点：
        - 显式积分方法，计算简单快速
        - 适合实时仿真场景
        - 精度较低，时间步长需足够小

        参数：
        - agent: 自车当前状态（位置、速度）
        - lead_agent: 前车状态（位置、速度、长度）
        - sampling_time: 积分时间步长（秒）

        返回：
        - 新的自车状态（更新后的位置、速度）
        """
        # params = self.idm_params# 获取模型参数
        # x_dot, v_agent_dot = self.idm_model([], agent.to_array(), lead_agent.to_array(), params)# 计算当前时刻的导数
        # return IDMAgentState(
        #     agent.progress + sampling_time * x_dot,
        #     agent.velocity + sampling_time * min(max(-self._decel_max, v_agent_dot), self._accel_max),
        # )

        # 获取模型参数
        params = self.idm_params

        # 计算当前时刻的导数
        _, acceleration = self.idm_model([], agent.to_array(), lead_agent.to_array(), params)

        # 前向欧拉积分
        new_velocity = agent.velocity + sampling_time * acceleration
        new_velocity = max(new_velocity, 0)  # 速度不能为负

        # 更新位置（使用平均速度）
        new_progress = agent.progress + sampling_time * (agent.velocity + new_velocity) / 2

        return IDMAgentState(progress=new_progress, velocity=new_velocity)

    def solve_odeint_idm_policy(
        self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float, solve_points: int = 10
    ) -> IDMAgentState:
        """
        Solves an initial value problem for a system of ODEs using scipy odeint

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :param solve_points: number of points for temporal resolution
        :return: solution to the differential equations

        使用SciPy的`odeint`函数（基于LSODA算法）求解IDM微分方程。该方法提供了更高精度的数值解，适用于更复杂的情境。它同样返回更新后的智能体状态。

        特点：
        - 基于LSODA算法，自动切换刚性/非刚性解法
        - 中等精度，适合离线分析
        - 支持多步积分，提高精度

        参数：
        - solve_points: 积分区间内的采样点数（影响精度）
        """
        t = np.linspace(0, sampling_time, solve_points)  # 构造时间序列

        # 调用odeint求解
        solution = odeint(
            self.idm_model,
            agent.to_array(),
            t,
            args=(
                lead_agent.to_array(),
                self.idm_params,
            ),
            tfirst=True,
        )

        # return the last solution 取最终状态
        return IDMAgentState(progress=solution[-1][0], velocity=solution[-1][1])

    def solve_ivp_idm_policy(self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float) -> IDMAgentState:
        """
        Solves an initial value problem for a system of ODEs using scipy RK45

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :return: solution to the differential equations

        使用SciPy的`solve_ivp`函数（默认使用Runge-Kutta 45阶方法）求解IDM微分方程。这个方法适合处理刚性问题，并且在时间积分上有更好的性能表现。它返回更新后的智能体状态。

        特点：
        - 自适应步长的Runge-Kutta方法（4-5阶）
        - 高精度，适合科研场景
        - 计算成本较高

        返回：
        - 积分结束后的最终状态
        """
        # 设置时间区间和初始条件
        t = (0, sampling_time)

        # 调用solve_ivp求解
        solution = solve_ivp(
            self.idm_model,
            t,
            agent.to_array(),
            args=(
                lead_agent.to_array(),
                self.idm_params,
            ),
            method="RK45",
        )

        # return the last solution 提取最终状态
        return IDMAgentState(progress=solution.y[0][-1], velocity=solution.y[1][-1])
