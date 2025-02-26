import sys
sys.path.append("/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/MineSim-Dynamic-Dev")

from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMap


dataroot = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets"
location = "guangdong_dapai"
mine_map_api = TgScenesMap(dataroot=dataroot, location=location)
a = 1
