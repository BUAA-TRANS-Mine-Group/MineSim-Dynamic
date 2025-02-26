import logging
import os
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union, cast

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from pandas.errors import EmptyDataError
from hydra.utils import instantiate

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.environment_manager.scenario_organizer import ScenarioFileBaseInfo


logger = logging.getLogger(__name__)


class SequentialLoadFilter:
    """
    inputs文件夹全部加载.
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self.scenario_builder = None

    def get_scenarios(self, scenario_file_list: List[ScenarioFileBaseInfo]) -> List[AbstractScenario]:
        # scenario_builder = MineSimDynamicScenarioBuilder(
        #     data_root=os.environ["MINESIM_DATA_ROOT"],
        #     map_root=os.environ["MINESIM_MAPS_ROOT"],
        #     map_version="v_1_5",
        # )
        # scenarios = scenario_builder.get_scenarios(so.scenario_list)

        logger.info("#log# Building Scenarios in inputs.")

        self.scenario_builder = instantiate(self._cfg.scenario_builder)
        scenarios = self.scenario_builder.get_scenarios(scenario_file_list=scenario_file_list)
        return scenarios

    def get_test_scenario(self, scenario_file_to_test: ScenarioFileBaseInfo) -> AbstractScenario:
        """获取下一个即将测试的场景。"""
        return self.scenario_builder.get_test_scenario(scenario_file_to_test=scenario_file_to_test)
