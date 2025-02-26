# Python library
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Third-party library
from hydra.utils import instantiate
from omegaconf import DictConfig

# Local library, `minesim.devkit`, ` minesim.devkit` dir is necessary
dir_current_file = os.path.dirname(__file__)
sys.path.append(dir_current_file)
dir_parent_1 = os.path.dirname(dir_current_file)
sys.path.append(dir_parent_1)
dir_parent_2 = os.path.dirname(dir_parent_1)
sys.path.append(dir_parent_2)
dir_parent_3 = os.path.dirname(dir_parent_2)
sys.path.append(dir_parent_3)


from devkit.configuration.sim_engine_conf import SimConfig as sim_config
from devkit.database.common.sequential_load_filter import SequentialLoadFilter
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.sim_engine.planning.planner.simple_planner import SimplePlanner
from devkit.sim_engine.callback.abstract_callback import AbstractCallback
from devkit.sim_engine.callback.multi_callback import MultiCallback
from devkit.sim_engine.runner.simulations_runner import SimulationRunner
from devkit.sim_engine.environment_manager.scenario_organizer import ScenarioOrganizer
from devkit.sim_engine.environment_manager.simulation_setup import SimulationSetup
from devkit.sim_engine.environment_manager.environment_simulation import EnvironmentSimulation
from devkit.utils.multithreading.worker_pool import WorkerPool
from devkit.script.builders.planner_builder import build_planners

logger = logging.getLogger(__name__)


# def build_simulations(pre_built_planners: Optional[List[AbstractPlanner]] = None,):
def build_simulations(
    cfg: DictConfig,
    callbacks: List[AbstractCallback],
    worker: WorkerPool,
    callbacks_worker: Optional[WorkerPool] = None,
    pre_built_planners: Optional[List[AbstractPlanner]] = None,
):
    """
    Build simulations.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param callbacks: Callbacks for simulation.
    :param worker: Worker for job execution.
    :param callbacks_worker: worker pool to use for callbacks from sim
    :param pre_built_planners: List of pre-built planners to run in simulation.
    :return A dict of simulation engines with challenge names.
    """
    logger.info("#log# Building simulations...")

    # 存储生成的所有仿真对象 # Create Simulation object container
    simulations = list()

    # 读取并构建场景过滤器，用于在构建器中筛选需要的场景  # Retrieve scenarios
    logger.info("#log# Extracting scenarios...")
    # 使用 原先的 OnSite 1.0 方式 读取场景库中的场景；
    logger.info("#log# Load and parse scenarios using [OnSite 1.0 script]...")
    so = ScenarioOrganizer(sim_config=sim_config)
    so.load()
    # print(f"###log### <测试参数>\n{so.sim_config}\n")
    scenario_filter = SequentialLoadFilter(cfg=cfg)
    scenarios = scenario_filter.get_scenarios(scenario_file_list=so.scenario_list)
    logger.info("#log# Load and parse scenarios using [OnSite 1.0 script], END.")

    # todo : 转移到最后加载
    # 初始化 指标度量引擎 映射，用于对不同场景类型进行评估
    # metric_engines_map = {}
    # if cfg.run_metric:
    #     logger.info("Building metric engines...")
    #     # 根据配置和已获取的场景来构建度量引擎
    #     metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
    #     logger.info("Building metric engines...DONE")
    # else:
    #     logger.info("Metric engine is disable")

    logger.info("#log# Building simulations from %d scenarios...", len(scenarios))

    # 建立用于缓存规划器的字典，避免重复构建非线程安全的规划器实例 # Cache used to keep a single instance of non-thread-safe planners
    planners_cache: Dict[str, AbstractPlanner] = dict()

    while True:  # 迭代所有场景，逐一构建仿真
        scenario_to_test = so.next_scenario()  # 该场景迭代方式使用 [OnSite 1.0 script]
        if scenario_to_test is None:
            break

        scenario = scenario_filter.get_test_scenario(scenario_file_to_test=scenario_to_test)
        #! refence devkit-devkit-comment/devkit/planning/script/builders/simulation_builder.py --------------------------------
        # Build planners
        if pre_built_planners is None:
            if "planner" not in cfg.keys():
                raise KeyError('#log# Planner not specified in config. Please specify a planner using "planner" field.')
            planners = build_planners(planners_cfg=cfg.planner, scenario=scenario, cache=planners_cache)
        else:
            planners = pre_built_planners

        # NOTE : 目前测试了只创建一个planner的代码。
        for planner in planners:
            # ! 转移至SimulationSetup
            # ego_controller: AbstractEgoController = instantiate(cfg.ego_controller, scenario=scenario)
            # simulation_time_controller: AbstractSimulationTimeController = instantiate(cfg.simulation_time_controller, scenario=scenario)
            # observations: AbstractObservation = build_observations(cfg.observation, scenario=scenario)

            # TODO 转移到最后加载
            # Metric Engine
            # metric_engine = metric_engines_map.get(scenario.scenario_type, None)
            # if metric_engine is not None:
            #     stateful_callbacks = [MetricCallback(metric_engine=metric_engine, worker_pool=callbacks_worker)]
            # else:
            #     stateful_callbacks = []
            stateful_callbacks = []

            # 如果配置文件中存在 simulation_log_callback，则同样实例化并加入 stateful_callbacks
            if "simulation_log_callback" in cfg.callback:
                stateful_callbacks.append(instantiate(cfg.callback["simulation_log_callback"], worker_pool=callbacks_worker))

            # 组装仿真时所需的核心组件(时间控制、观察器、自车控制器、场景等)
            simulation_setup = SimulationSetup(sim_config=sim_config, scenario=scenario, cfg=cfg)

            # 创建 Simulation 对象，将通用回调 callbacks 与针对该场景的 stateful_callbacks 合并
            envi_simulation = EnvironmentSimulation(
                simulation_setup=simulation_setup,
                callback=MultiCallback(callbacks + stateful_callbacks),
                simulation_history_buffer_duration=cfg.simulation_history_buffer_duration,
            )

            # 使用 SimulationRunner 将仿真(Scenario)与 Planner 组合在一起
            simulations.append(SimulationRunner(simulation=envi_simulation, planner=planner))

        pass
    logger.info("#log# Building simulations...DONE!")
    return simulations


if __name__ == "__main__":
    build_simulations(pre_built_planners="todo")
    # todo 多 worker
