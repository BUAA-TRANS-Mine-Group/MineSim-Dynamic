# Python library
import os
import sys

# Third-party library
import json


# Local library
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

from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMap
from devkit.sim_engine.map_manager.map_expansion.bit_map import BitMap
from devkit.sim_engine.scenario_manager.scenario_info import ScenarioInfo


class ScenarioParser:
    """Parser single scenario file and map file."""

    def __init__(self, sim_config):
        self.scenario_info = ScenarioInfo()
        self._sim_config = sim_config

    def parse(self, scenario: dict) -> ScenarioInfo:
        """解析动态场景.动态场景包括:
        1) 自车信息,self.scenario_info.ego_info;
        2) 高精度地图信息,self.scenario_info.hdmaps;
        3) 测试场景信息,self.scenario_info.test_setting;
        4) 其它背景车辆全部轨迹信息,self.scenario_info.vehicle_traj;
        """
        self.scenario_info.add_settings(scenario_name=scenario["scenario_name"], scenario_type=scenario["scene_type"])
        self._parse_scenario(scenario)
        self._parse_hdmaps(scenario)

        return self.scenario_info

    def _parse_scenario(self, scenario: dict):
        """解析多车场景.
        """
        with open(scenario["file_info"], "r") as f:
            one_scenario = json.load(f)

        # 1) 获取ego车辆的目标区域,goal box
        self.scenario_info.test_setting["goal"] = one_scenario["goal"]

        # 2) 步长,最大时间,最大最小xy范围
        self.scenario_info._get_dt_maxt(one_scenario)

        # 3) 读取ego车初始信息
        self.scenario_info._init_vehicle_ego_info(one_scenario)

        for idex, value_traj_segment in enumerate(one_scenario["TrajSegmentInfo"]):
            # if value_traj_segment['TrajSetToken'] != "ego":
            num_vehicle = idex + 1
            # 4) 读取车辆长度与宽度等形状信息,录入scenario_info.背景车id从1号开始,ego车为0
            self.scenario_info.add_vehicle_shape(id=num_vehicle, t=-1, traj_info=value_traj_segment)
            # 5) 以下读取背景车相关信息,车辆编号从1号开始,轨迹信息记录在vehicle_traj中
            self.scenario_info.add_vehicle_traj(id=num_vehicle, t=-1, traj_info=value_traj_segment)

        return self.scenario_info

    def _parse_hdmaps(self, scenario: str) -> None:
        """解析高清地图文件并更新到scenario_info.
        功能:
        1)获取可行驶区域的mask图信息.
        2)获取rgb图的目录(用于可视化) .
        3)加载路网结构.
        """
        dataroot = self._sim_config["directory_conf"]["dir_datasets"]
        location = scenario["location"]

        # 1) 获取mask图信息并确定最大方形区域
        self._load_mask_and_calculate_square_region(dataroot, location)

        # 2) 获取rgb图信息 (  如果需要 )
        # self._load_rgb_image(dataroot,location)

        # 3) 加载路网信息
        self._load_road_network(dataroot, location)

    def _load_mask_and_calculate_square_region(self, dataroot: str, location: str) -> None:
        """加载mask并确定最大的方形区域."""
        self.scenario_info.hdmaps["image_mask"] = BitMap(dataroot, location, "bitmap_mask")

        # 整合所有的坐标点
        x_coords = (
            [self.scenario_info.ego_info["x"]]
            + self.scenario_info.test_setting["goal"]["x"]
            + [self.scenario_info.test_setting["x_min"], self.scenario_info.test_setting["x_max"]]
        )
        y_coords = (
            [self.scenario_info.ego_info["y"]]
            + self.scenario_info.test_setting["goal"]["y"]
            + [self.scenario_info.test_setting["y_min"], self.scenario_info.test_setting["y_max"]]
        )

        # 根据坐标确定最大的方形框
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        max_range = max(x_max - x_min, y_max - y_min)
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        utm_local_range = (x_center - max_range / 2, y_center - max_range / 2, x_center + max_range / 2, y_center + max_range / 2)
        self.scenario_info.hdmaps["image_mask"].load_bitmap_using_utm_local_range(utm_local_range, 20, 20)

    def _load_rgb_image(self, dataroot: str, location: str) -> None:
        """加载地图的RGB图像."""
        self.scenario_info.hdmaps["image_rgb"] = BitMap(dataroot, location, "bitmap_rgb", is_transform_gray=True)

    def _load_road_network(self, dataroot: str, location: str) -> None:
        """加载地图的路网信息."""
        self.scenario_info.hdmaps["tgsc_map"] = TgScenesMap(dataroot, location)


if __name__ == "__main__":
    from devkit.sim_engine.environment_manager.scenario_organizer import ScenarioOrganizer
    from configuration.sim_engine_conf import SimConfig as sim_config

    so = ScenarioOrganizer(sim_config=sim_config)
    so.load()
    # envi_sim = EnvironmentSimulation()
    print(f"###log### <测试参数>\n{so.sim_config}\n")

    scenario_to_test = so.next_scenario()
    pass

    scenario_parser = ScenarioParser(sim_config=sim_config)
    scenario_info = scenario_parser.parse(scenario_to_test)
    pass
