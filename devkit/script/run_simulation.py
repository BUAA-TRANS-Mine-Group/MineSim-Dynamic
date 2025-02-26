import sys
import logging
import os
import tempfile
import uuid
from typing import List, Optional, Union
from datetime import datetime


import hydra  # 导入 Hydra，用于基于配置文件灵活地管理实验/脚本配置
from omegaconf import DictConfig, OmegaConf

# NOTE 这种方法确保无论当前工作目录如何变化，路径都会被解析成绝对路径，避免路径解析差异。
dir_current_file = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_current_file)
dir_parent_1 = os.path.dirname(dir_current_file)  # devkit
sys.path.append(dir_parent_1)
dir_parent_2 = os.path.dirname(dir_parent_1)
sys.path.append(dir_parent_2)
dir_parent_3 = os.path.dirname(dir_parent_2)
sys.path.append(dir_parent_3)

from devkit.configuration.sim_engine_conf import SimConfig as sim_config

from devkit.script.builders.simulation_builder import build_simulations
from devkit.script.builders.simulation_callback_builder import build_callbacks_worker
from devkit.script.builders.simulation_callback_builder import build_simulation_callbacks
from devkit.script.utils import run_runners
from devkit.script.utils import set_up_common_builder
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.script.utils import set_default_path_by_environment_variable


logging.basicConfig(level=logging.INFO)  # 配置日志系统，使其默认以 INFO 级别输出
logger = logging.getLogger(__name__)  # 获取一个与当前模块同名的日志记录器


def get_cfg(BASE_DEVKIT_PATH, config_name="simulation_mode_1_default_replay_test_mode"):
    """read all config by YAML.

    - 配置文件主要来自以下位置:
        - devkit/script/config/*.YMAL [配置不同组件的具体细节参数:例如车辆lag time constant, ego_motion_controller 参数等]

    Args:
        BASE_DEVKIT_PATH (str): 绝对路径,e.g. '/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic/MineSim-Dynamic-Dev/devkit'
        config_name (str, optional): 主配置文件的名字（不需要文件扩展名）. Defaults to "default_simulation".
    """
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    BASE_SCRIPT_PATH_ABS = os.path.join(BASE_DEVKIT_PATH, "script")
    relative_script_path = os.path.relpath(BASE_SCRIPT_PATH_ABS, current_file_path)
    relative_sim_engine_path = os.path.join(relative_script_path, "config", "sim_engine")

    searchpath = [
        f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config')}",
        f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config', 'common')}",
        f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config', 'sim_engine')}",
    ]

    # !创建一个临时目录来存储仿真过程: log ,...
    # 生成随机的不重复目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 格式：2025-02-25_10-41-08
    unique_dir_name = f"log_{timestamp}_{uuid.uuid4().hex[:8]}"  # 结构：log_时间戳_UUID前8位
    # 生成完整路径
    outputs_log_dir = os.path.join(dir_parent_2, "outputs", "outputs_log")
    SAVE_DIR = os.path.join(outputs_log_dir, unique_dir_name)
    # 创建目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Simulation logs will be saved in: {SAVE_DIR}")

    DATASET_PARAMS = [
        "scenario_builder=minesim_dynamic_scenario",  # use minesim scenario library
        # "scenario_filter=one_continuous_log",  # simulate only one log
        # "scenario_filter.log_names=['Scenario-dapai_intersection_1_3_19']",
        # "scenario_filter.limit_total_scenarios=2",  # use 2 total scenarios
    ]

    # 不使用 sim_engine_conf.py 中的配置覆盖 simulation engine core components
    overrides = [
        f"hydra.searchpath={searchpath}",
        f"group={SAVE_DIR}",
        "worker=sequential",
        "experiment=${experiment_name}/${job_name}",  # experiment_name 来自 YAML 文件读取
        "output_dir=${group}/${experiment}",
        *DATASET_PARAMS,
    ]

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=relative_sim_engine_path, version_base="1.1")
    cfg = hydra.compose(
        config_name=config_name,  # 主配置文件的名字（不需要文件扩展名YAML）
        overrides=overrides,  # 覆盖原始配置中的某些参数
    )
    return cfg


def run_simulation(cfg: DictConfig, planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None) -> None:
    """规划仿真，入口

    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    # 构建场景构建器，用于生成需要仿真的场景
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=None)

    # 构建仿真回调的 worker 池(若需要并行执行回调)，以及一系列回调函数  Build simulation callbacks
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool)

    # 若外部传入了一个或多个 planner，且配置文件中也有 planner，则优先使用外部传入的 planner
    # Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
    if planners and "planner" in cfg.keys():
        logger.info("Using pre-instantiated planner. Ignoring planner in config")
        OmegaConf.set_struct(cfg, False)  # 临时关闭 strict 模式，以便修改 cfg
        cfg.pop("planner")  # 移除 config 中的 planner
        OmegaConf.set_struct(cfg, True)  # 重新开启 strict 模式

    # 若只传入了一个 planner，则将其放入列表，统一处理 # Construct simulations
    if isinstance(planners, AbstractPlanner):
        planners = [planners]

    # 根据配置文件和预设的回调、场景构建器等信息，构建出若干 simulations(仿真运行器)
    runners = build_simulations(
        cfg=cfg,
        callbacks=callbacks,
        worker=common_builder.worker,
        callbacks_worker=callbacks_worker_pool,
        pre_built_planners=planners,
    )

    logger.info("#log# Running simulation...")
    # 使用 run_runners 函数来执行构建好的所有仿真运行器;通常每个 scenario 会有一个 SimulationRunner
    run_runners(runners=runners, common_builder=common_builder, cfg=cfg, profiler_name="running_simulation")
    logger.info("#log# Finished running simulation!")


def main(config_name: str) -> None:
    set_default_path_by_environment_variable()
    cfg = get_cfg(BASE_DEVKIT_PATH=sim_config["BASE_DEVKIT_PATH"], config_name=config_name)

    # 在执行仿真时，simulation_log_main_path 必须未被设置，否则抛出异常
    assert cfg.simulation_log_main_path is None, "Simulation_log_main_path must not be set when running simulation."

    # Execute simulation with preconfigured planner(s).
    run_simulation(cfg=cfg)


if __name__ == "__main__":
    # ! 仿真入口
    # SIMULATION MODE Choice
    # simulation_mode_0
    # simulation_mode_1_default_replay_test_mode
    # simulation_mode_2_interactive_test_mode
    # simulation_mode_3
    # simulation_mode_4
    # simulation_mode_5
    # simulation_mode_6
    main(config_name="simulation_mode_1_default_replay_test_mode")
