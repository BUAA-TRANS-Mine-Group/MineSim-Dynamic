from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Type
import sys

sys.path.append("/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/MineSim-Dynamic-Dev")

from devkit.sim_engine.map_manager.minesim_map.minesim_map_factory import get_maps_api


# 测试代码
map_root = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/maps"
map_name = "guangdong_dapai"

mine_map_api = get_maps_api(map_root=map_root, map_name=map_name)
a = 1
