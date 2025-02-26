import os
import sys
from copy import deepcopy


dir_current_file = os.path.dirname(__file__)
sys.path.append(dir_current_file)
dir_parent_1 = os.path.dirname(dir_current_file)
sys.path.append(dir_parent_1)
dir_parent_2 = os.path.dirname(dir_parent_1)
sys.path.append(dir_parent_2)
# from   devkit.sim_engine.observation_manager.observation_mine _mine  import Observation
# from sim_engine.environment_manager.scenario_controller import ScenarioController
from devkit.configuration.sim_engine_conf import SimConfig as sim_config
from devkit.scenario_builder.minesim_scenario_json.minesim_convert_data_type import ScenarioFileBaseInfo


class ScenarioOrganizer:
    """场景管理器，读取本地多个场景，按照场景顺序逐个仿真."""

    def __init__(self, sim_config):
        self.scenario_list = []
        self.scenario_list_tested = []
        self.scenario_num = 0
        self._scenario_info = dict
        self._sim_config = sim_config

    def load(self) -> None:
        """读取配置文件,按照配置文件(py格式)准备场景内容.加载多个场景."""
        self._check_dir()

        # self.config["file_info"] = {
        #     "dir_inputs": dir_inputs,
        #     "dir_outputs": dir_outputs,
        #     # 'dir_ScenariosResultes':os.path.join(dir_inputs,'ScenariosResultes'),
        #     "dir_ScenariosResultes": dir_inputs,
        #     "dir_TrajDataResultes": os.path.join(dir_inputs, "TrajDataResultes"),
        #     "dir_datasets": dir_datasets,
        #     "location": "jiangxi_jiangtong",  # dfdaute
        # }
        # 取文件夹下所有待测场景,存在列表中
        # self._sim_config['directory_conf']['dir_inputs']
        # self.config["test_settings"].setdefault("skip_exist_scene", False)
        dir_scenarios = self._sim_config["directory_conf"]["dir_inputs"]
        for item in os.listdir(dir_scenarios):
            if item != "__pycache__" and item[0] != ".":
                dir_scene_file = os.path.join(dir_scenarios, item)
                scenario_name = item[9:-5]
                location_temp = scenario_name.split("_")[0]
                if location_temp == "jiangtong":
                    location = "jiangxi_jiangtong"
                elif location_temp == "dapai":
                    location = "guangdong_dapai"
                else:
                    raise Exception("###Exception### 地图location 错误!")
                type_str = scenario_name.split("_")[1]
                if type_str == "intersection":
                    scenario_type = "intersection_mixd"
                scenario_type = "intersection_mixd"
                scenario_file_info = ScenarioFileBaseInfo(
                    log_file_load_path=dir_scene_file,
                    log_file_name=item,
                    scenario_name=scenario_name,
                    location=location,
                    scenario_type=scenario_type,
                    data_root=os.environ["MINESIM_DATA_ROOT"],
                    map_root=os.environ["MINESIM_MAPS_ROOT"],
                )

                # 将场景加入列表中
                self.scenario_list.append(deepcopy(scenario_file_info))

        self.scenario_num = len(self.scenario_list)
        pass

    def next_scenario(self):
        """
        给出下一个待测场景与测试模式,如果没有场景了,则待测场景名称为None
        """
        if self.scenario_list:  # 首先判断列表是否为空,如果列表不为空,则取场景;否则,输出None
            # 列表不为空,输出0号场景,且将其从列表中删除(通过pop函数实现)
            scenario_to_test = self.scenario_list.pop(0)
        else:
            # 列表为空,输出None
            scenario_to_test = None

        return scenario_to_test

    def add_result(self, concrete_scenario: dict, res: float) -> None:
        # 判断测试模式,如果是replay,则忽略测试结果
        if self.test_mode == "replay":
            return

    def _check_dir(self) -> None:
        dir_maps = self._sim_config["directory_conf"]["dir_maps"]
        if not os.path.exists(dir_maps):
            raise ValueError(f"#log# map dir error:{dir_maps}")
        dir_inputs = self._sim_config["directory_conf"]["dir_inputs"]
        if not os.path.exists(dir_inputs):
            raise ValueError(f"#log# inputs dir error:{dir_inputs}")
        dir_outputs_log = self._sim_config["directory_conf"]["dir_outputs_log"]
        if not os.path.exists(dir_outputs_log):
            os.makedirs(dir_outputs_log)
            print(f"#log# dir_outputs_log:{dir_outputs_log} \n")
        dir_outputs_figure = self._sim_config["directory_conf"]["dir_outputs_figure"]
        if not os.path.exists(dir_outputs_figure):
            os.makedirs(dir_outputs_figure)
            print(f"#log# dir_outputs_figure:{dir_outputs_figure} \n")

    @property
    def sim_config(self) -> dict:
        return self._sim_config

    @property
    def scenarior_info(self) -> dict:
        return self._scenarior_info


if __name__ == "__main__":
    pass
