from devkit.common.actor_state.state_representation import TimePoint


def forward_integrate(init: float, delta: float, sampling_time: TimePoint) -> float:
    """
    前向欧拉积分
    Performs a simple euler integration.
    :param init: Initial state
    :param delta: The rate of chance of the state.
    :param sampling_time: The time duration to propagate for.
    :return: The result of integration
    """
    return float(init + delta * sampling_time.time_s)


# 这个函数 `forward_integrate` 实现了前向欧拉积分法，用于计算在给定初始状态、变化率和时间步长下的状态更新。前向欧拉积分法的公式如下：

# \[
# \text{new\_state} = \text{init} + \Delta \times \Delta t
# \]

# 其中：
# - \(\text{init}\) 是初始状态。
# - \(\Delta\) 是状态的变化率。
# - \(\Delta t\) 是时间步长，即 `sampling_time.time_s`。

# 这个公式表示，在时间步长 \(\Delta t\) 内，状态的变化量 \(\Delta \times \Delta t\) 被加到初始状态上，从而得到更新后的状态。
