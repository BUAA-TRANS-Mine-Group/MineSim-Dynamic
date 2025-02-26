from __future__ import annotations

import logging
import time
from typing import Any, Callable, List

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.sim_engine.runner.abstract_runner import AbstractRunner
from devkit.sim_engine.runner.runner_report import RunnerReport
from devkit.sim_engine.environment_manager.environment_simulation import EnvironmentSimulation

logger = logging.getLogger(__name__)


def for_each(fn: Callable[[Any], Any], items: List[Any]) -> None:
    """
    这个函数接收一个函数 fn 和一个列表 items，然后对列表中的每个元素调用 fn 函数。
    这个模式通常用于对一组对象执行统一的操作。
    Call function on every item in items
    :param fn: function to be called fn(item)
    :param items: list of items
    """
    for item in items:  # 逐个对列表中的 item 调用 fn
        fn(item)


class SimulationRunner(AbstractRunner):
    """用于管理和执行多个模拟任务。
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: EnvironmentSimulation, planner: AbstractPlanner):
        """
        Initialize the simulations manager
        :param simulation: EnvironmentSimulation which will be executed
        :param planner: to be used to compute the desired ego's trajectory
        """
        self._simulation = simulation  # 保存要执行的仿真对象
        self._planner = planner  # 保存要使用的规划器

    def _initialize(self) -> None:
        """
        用于初始化模拟任务和规划器。调用了一系列的回调函数，确保在初始化开始和结束时执行特定的操作。
        Initialize the planner
        调用 _initialize 方法，初始化规划器和模拟。
        该方法会触发一系列回调函数，包括 on_initialization_start 和 on_initialization_end，
        并通过调用 Simulation.initialize() 来重置模拟状态、初始化历史缓冲区、加载初始观测数据等。
        """
        # Execute specific callback
        self._simulation.callback.on_initialization_start(setup=self._simulation.setup, planner=self.planner)

        # Initialize Planner
        self.planner.initialize(initialization=self._simulation.initialize())

        # Execute specific callback
        self._simulation.callback.on_initialization_end(setup=self._simulation.setup, planner=self.planner)

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: Planner used by the SimulationRunner
        """
        return self._planner

    @property
    def simulation(self) -> EnvironmentSimulation:
        """
        :return: EnvironmentSimulation used by the SimulationRunner
        """
        return self._simulation

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario relative to the simulation.
        """
        return self.simulation.scenario

    def run(self) -> RunnerReport:
        """
        核心，它管理了模拟任务的整个执行流程。
        具体步骤包括初始化规划器、迭代执行模拟任务的每一步操作，
        并在模拟结束后生成并返回一个 RunnerReport 对象。
        整个过程中使用了一些回调函数来执行特定的操作，同时利用 time.perf_counter 来测量执行时间。

        Run through all simulations. The steps of execution follow:
         - Initialize all planners
         - Step through simulations until there no running simulation
        :return: List of SimulationReports containing the results of each simulation
        """
        start_time = time.perf_counter()

        # Initialize reports for all the simulations that will run
        report = RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulation.scenario.scenario_name,
            planner_name=self.planner.name(),
            log_name=self._simulation.scenario.log_name,
        )

        # Execute specific callback
        self.simulation.callback.on_simulation_start(self.simulation.setup)

        # Initialize all simulations
        self._initialize()

        # ! 6.进入模拟执行循环
        while self.simulation.is_simulation_running():
            # Execute specific callback # 6.1 步骤开始回调
            self.simulation.callback.on_step_start(setup=self.simulation.setup, planner=self.planner)

            # Perform step 执行 # 6.2 获取规划器输入
            planner_input = self._simulation.get_planner_input()
            # logger.debug("#log# EnvironmentSimulation iterations: %s" % planner_input.iteration.index)

            # Execute specific callback # 6.3 规划器开始回调
            self._simulation.callback.on_planner_start(setup=self.simulation.setup, planner=self.planner)

            # Plan path based on all planner's inputs # 6.4 计算规划器的轨迹
            trajectory = self.planner.compute_trajectory(current_input=planner_input)

            # Propagate simulation based on planner trajectory
            # 6.5 规划器结束回调
            self._simulation.callback.on_planner_end(setup=self.simulation.setup, planner=self.planner, trajectory=trajectory)
            # 6.6 传播模拟状态(更新车辆位置、历史、观测等)
            self.simulation.propagate(trajectory=trajectory)

            # 6.7 步骤结束回调  # Execute specific callback
            self.simulation.callback.on_step_end(setup=self.simulation.setup, planner=self.planner, sample=self.simulation.history.last())

            # 7. 检查模拟是否结束 # Store reports for simulations which just finished running
            current_time = time.perf_counter()
            if not self.simulation.is_simulation_running():
                report.end_time = current_time

            logger.info(
                f"#log# scenario_name: {self.scenario.scenario_name}, EnvironmentSimulation iteration index = {planner_input.iteration.index}, END!"
            )

        # !8. 调用回调函数：模拟结束;执行多个  self.simulation.callback中绑定的回调函数; # Execute specific callback
        self.simulation.callback.on_simulation_end(setup=self.simulation.setup, planner=self.planner, history=self.simulation.history)

        # 生成规划器报告，将其存入 RunnerReport
        planner_report = self.planner.generate_planner_report()
        report.planner_report = planner_report

        logger.info("#log# scenario_name: {self.scenario.scenario_name},  EnvironmentSimulation iteration (ALL) END!--------------------------\n\n")
        return report
