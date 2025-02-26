# Python library
import gc
import sys
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Third-party library
import hydra

# Local library, `minesim.devkit`, ` minesim.devkit` dir is necessary
dir_current_file = os.path.dirname(__file__)
sys.path.append(dir_current_file)
dir_parent_1 = os.path.dirname(dir_current_file)
sys.path.append(dir_parent_1)
dir_parent_2 = os.path.dirname(dir_parent_1)
sys.path.append(dir_parent_2)
dir_parent_3 = os.path.dirname(dir_parent_2)
sys.path.append(dir_parent_3)
dir_parent_4 = os.path.dirname(dir_parent_3)
sys.path.append(dir_parent_4)

from devkit.sim_engine.environment_manager.scenario_organizer import ScenarioOrganizer
from devkit.configuration.sim_engine_conf import SimConfig as sim_config
from devkit.scenario_builder.minesim_scenario_json.minesim_dynamic_scenario_builder import MineSimDynamicScenarioBuilder


def set_environment_variable():
    # 设置 MINESIM_DATA_ROOT 和 MINESIM_MAPS_ROOT 环境变量
    os.environ["MINESIM_DATA_ROOT"] = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets"
    os.environ["MINESIM_DYNAMIC_SCENARIO_ROOT"] = (
        "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/scenarios/dynamic_obstacle_scenarios"
    )
    os.environ["MINESIM_MAPS_ROOT"] = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/maps"
    os.environ["MINESIM_INPUTS_SCENARIO_ROOT"] = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/inputs"


def get_cfg(BASE_DEVKIT_PATH, config_name="default_simulation"):
    """read all config by YAML.

    Args:
        BASE_DEVKIT_PATH (_type_): 绝对路径
        config_name (str, optional): _description_. Defaults to "default_simulation".
    """
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    BASE_SCRIPT_PATH_ABS = os.path.join(BASE_DEVKIT_PATH, "script")
    relative_script_path = os.path.relpath(BASE_SCRIPT_PATH_ABS, current_file_path)
    relative_sim_engine_path = os.path.join(relative_script_path, "config", "sim_engine")

    # ste search path
    searchpath = [
        f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config')}",
        f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config', 'common')}",
        f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config', 'sim_engine')}",
        # f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config', 'sim_engine', 'ego_motion_controller')}",
        # f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config', 'sim_engine', 'ego_motion_controller', 'tracker')}",
        # f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config', 'sim_engine', 'ego_update_model')}",
        # f"file://{os.path.join(BASE_DEVKIT_PATH, 'script', 'config', 'sim_engine', 'simulation_time_controller')}",
    ]
    EGO_CONTROLLER = "two_stage_controller"  # [log_play_back_controller, perfect_tracking_controller]
    OBSERVATION_AGENT_UPDATE = "box_observation"  # [box_observation, idm_agents_observation, lidar_pc_observation]

    # Create a temporary directory to store the simulation artifacts
    SAVE_DIR = tempfile.mkdtemp()

    DATASET_PARAMS = [
        "scenario_builder=minesim_dynamic_scenario",  # use minesim mini database
        # "scenario_filter=one_continuous_log",  # simulate only one log
        # "scenario_filter.log_names=['Scenario-dapai_intersection_1_3_19']",
        # "scenario_filter.limit_total_scenarios=2",  # use 2 total scenarios
    ]

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=relative_sim_engine_path)

    cfg = hydra.compose(
        config_name=config_name,  # 主配置文件的名字（不需要文件扩展名）
        overrides=[  # 覆盖原始配置中的某些参数
            "experiment_name=planner_demo",
            f"ego_motion_controller={EGO_CONTROLLER}",
            f"observation_agent_update={OBSERVATION_AGENT_UPDATE}",
            f"hydra.searchpath={searchpath}",
            f"+group={SAVE_DIR}",
            "+job_name=planner_demo_2",
            "+experiment=${experiment_name}/${job_name}",
            "+output_dir=${group}/${experiment}",
            *DATASET_PARAMS,
        ],
    )
    return cfg


def test():
    tic = time.time()
    set_environment_variable()

    so = ScenarioOrganizer(sim_config=sim_config)
    so.load()
    print(f"###log### <测试参数>\n{so.sim_config}\n")

    scenario_builder = MineSimDynamicScenarioBuilder(
        data_root=os.environ["MINESIM_DATA_ROOT"],
        map_root=os.environ["MINESIM_MAPS_ROOT"],
        map_version="v_1_5",
    )
    scenarios = scenario_builder.get_scenarios(so.scenario_list)

    scenario_to_test = so.next_scenario()
    scenario = scenario_builder.get_test_scenario(scenario_file_to_test=scenario_to_test)

    pass


if __name__ == "__main__":
    test()
    pass
