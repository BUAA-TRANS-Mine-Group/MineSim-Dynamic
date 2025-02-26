from __future__ import annotations

import logging
from functools import partial
import random
from typing import Any, List, Optional, Tuple, Type, Union, cast, Dict

from devkit.common.actor_state.vehicle_parameters import VehicleParameters

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from devkit.scenario_builder.minesim_scenario_json.minesim_dynamic_scenario import MineSimDynamicScenario
from devkit.sim_engine.map_manager.abstract_map import AbstractMap
from devkit.scenario_builder.minesim_scenario_json.minesim_scenario_utils import absolute_path_to_log_name
from devkit.sim_engine.environment_manager.scenario_organizer import ScenarioFileBaseInfo

ScenarioDict = Dict[str, List[MineSimDynamicScenario]]

logger = logging.getLogger(__name__)


class MineSimDynamicScenarioBuilder(AbstractScenarioBuilder):
    """Builder class for constructing MIneSimn scenarios for training and simulation."""

    def __init__(
        self,
        data_root: str,
        map_root: str,
        map_version: str,
    ):
        """
        Initialize scenario builder that filters and retrieves scenarios from the MineSim dataset.
        :param data_root: Local data root for loading (or storing downloaded) the log databases.
                          If `json_files` is not None, all downloaded databases will be stored to this data root.
                          E.g.: /data/sets/devkit
        :param map_root: Local map root for loading (or storing downloaded) the map database.
        :param json_files: Path to load the log database(s) from.
                         It can be a local/remote path to a single database, list of databases or dir of databases.
                         If None, all database filenames found under `data_root` will be used.
                         E.g.: /data/sets/devkit/devkit-v1.1/splits/mini/2021.10.11.08.31.07_veh-50_01750_01948.db
        :param map_version: Version of map database to load. The map database is passed to each loaded log database.
        :param max_workers: Maximum number of workers to use when loading the databases concurrently.
                            Only used when the number of databases to load is larger than this parameter.
        :param verbose: Whether to print progress and details during the database loading and scenario building.
        :param scenario_mapping: Mapping of scenario types to extraction information.
        :param vehicle_parameters: Vehicle parameters for this db.

        初始化场景构建器，该构建器从 MineSim 数据集中筛选并检索场景。
        :param data_root: 用于加载（或存储下载的）日志数据库的本地数据根目录。
                            如果`json_files`不为None，则所有下载的数据库都将存储到此数据根目录下。例如：/data/sets/devkit
        :param map_root: 用于加载（或存储下载的）地图数据库的本地地图根目录。
        :param json_files: 用于加载日志数据库的路径。
                            它可以是单个数据库的本地/远程路径、数据库列表或数据库目录。
                            如果为 None，则将使用在`data_root`下找到的所有数据库文件名。
                            例如：/data/sets/devkit/devkit-v1.1/splits/mini/2021.10.11.08.31.07_veh-50_01750_01948.db
        :param map_version: 要加载的地图数据库的版本。该地图数据库会被传递给每个已加载的日志数据库。
        # :param max_workers: 并发加载数据库时使用的最大工作线程数。仅当要加载的数据库数量超过此参数时使用。
        # :param verbose: 是否在数据库加载和场景构建过程中打印进度和详细信息。
        # :param scenario_mapping: 场景类型与提取信息的映射。
        # :param vehicle_parameters: 该数据库的车辆参数。
        """
        self._data_root = data_root
        self._map_root = map_root
        self._map_version = map_version

        self.scenarios = None

    def __reduce__(self) -> Tuple[Type[MineSimDynamicScenarioBuilder], Tuple[Any, ...]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class
        """
        return self.__class__, (
            self._data_root,
            self._map_root,
            self._map_version,
        )

    @classmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Inherited. See superclass.

        return cast(Type[AbstractScenario], MineSimDynamicScenario)：
        - cast 是一种类型提示工具（来自 typing 模块），用于帮助类型检查器理解类型转换。它不会对运行时的类型进行实际更改。
        - cast(Type[AbstractScenario], MineSimDynamicScenario) 将 MineSimDynamicScenario 视作 Type[AbstractScenario] 类型。
        - MineSimDynamicScenario 应当是一个符合 AbstractScenario 接口或基类要求的类。此处 cast 提供了类型信息，避免静态类型检查器（如 MyPy）产生警告或错误。
        """
        return cast(Type[AbstractScenario], MineSimDynamicScenario)

    def _create_scenario(self, scenario_file: ScenarioFileBaseInfo) -> MineSimDynamicScenario:
        scenario = MineSimDynamicScenario(scenario_file=scenario_file)
        scenario.extract_tracked_objects_within_entrintime_window()

        return scenario

    def get_scenarios(self, scenario_file_list: List[ScenarioFileBaseInfo]) -> List[AbstractScenario]:
        """Implemented. See interface."""
        scenarios = list()
        for scenario_file in scenario_file_list:

            scenarios.append(self._create_scenario(scenario_file))

        self.scenarios = scenarios
        return self.scenarios

    def get_test_scenario(self, scenario_file_to_test: ScenarioFileBaseInfo) -> AbstractScenario:
        for scenario in self.scenarios:
            if scenario.scenario_name == scenario_file_to_test.scenario_name:
                return scenario
        return ValueError("scenarios load error.")

    # @property
    # def repartition_strategy(self) -> RepartitionStrategy:
    #     """Implemented. See interface."""
    #     return RepartitionStrategy.REPARTITION_FILE_DISK
