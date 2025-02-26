from typing import Dict
import sys

sys.path.append("/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/MineSim-Dynamic-Dev")
from devkit.sim_engine.map_manager.abstract_map import AbstractMap
from devkit.sim_engine.map_manager.abstract_map_factory import AbstractMapFactory


class MapManager:
    """Class to store created maps using a map factory."""

    def __init__(self, map_factory: AbstractMapFactory):
        """
        Constructor of MapManager.
        :param map_factory: map factory.
        """
        self.map_factory = map_factory
        self.maps: Dict[str, AbstractMap] = {}

    def get_map(self, map_name: str) -> AbstractMap:
        """
        Returns the queried map from the map factory, creating it if it's missing.
        :param map_name: Name of the map.
        :return: The queried map.
        """
        if map_name not in self.maps:
            self.maps[map_name] = self.map_factory.build_map_from_name(map_name)

        return self.maps[map_name]


if __name__ == "__main__":
    # 测试代码

    dataroot = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets"
    location = "guangdong_dapai"

    from devkit.sim_engine.map_manager.minesim_map.minesim_map_factory import MineSimMapFactory, get_mine_maps_mask_png, get_mine_maps_semantic_json
    from devkit.sim_engine.map_manager.minesim_map_data.minesim_bitmap_png_loader import MineSimBitMapPngLoader
    from devkit.sim_engine.map_manager.minesim_map_data.minesim_semanticmap_json_loader import MineSimSemanticMapJsonLoader

    # bitmap_png_loader = MineSimBitMapPngLoader(
    #     map_root="/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/maps",
    #     location="jiangxi_jiangtong",
    #     bitmap_type="bitmap_mask",
    #     is_transform_gray=False,
    # )

    bitmap_png_loader = get_mine_maps_mask_png(
        map_root="/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/maps",
        location="jiangxi_jiangtong",
    )
    # semanticmap_json_loader = MineSimSemanticMapJsonLoader(
    #     map_root="/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/maps",
    #     location="jiangxi_jiangtong",
    # )
    semanticmap_json_loader = get_mine_maps_semantic_json(
        map_root="/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/maps",
        location="jiangxi_jiangtong",
    )

    mine_map_factory = MineSimMapFactory(bitmap_png_loader=bitmap_png_loader, semanticmap_json_loader=semanticmap_json_loader)

    AbstractMapFactory
    mine_map_manager = MapManager(map_factory=mine_map_factory)
    mine_map_api = mine_map_manager.get_map(map_name="jiangxi_jiangtong")
    a = 1
