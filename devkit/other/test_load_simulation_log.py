# Python library
import gc
import sys
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging

# Third-party library
import hydra
import matplotlib
import matplotlib.pyplot as plt

import pathlib
import pickle  # Python 内置序列化工具
from concurrent.futures import ThreadPoolExecutor  # 用于在多个线程中并发执行任务
from functools import partial  # 用于固定部分函数参数的便捷工具
from pathlib import Path  # pathlib.Path 用于文件路径管理

# Local library, `minesim.devkit`, ` minesim.devkit` dir is necessary
dir_current_file = os.path.dirname(__file__)
sys.path.append(dir_current_file)
dir_parent_1 = os.path.dirname(dir_current_file)
sys.path.append(dir_parent_1)
dir_parent_2 = os.path.dirname(dir_parent_1)
sys.path.append(dir_parent_2)
dir_parent_3 = os.path.dirname(dir_parent_2)
sys.path.append(dir_parent_3)


from devkit.sim_engine.log_api.simulation_log import SimulationLog

# /tmp/tmpax80lbia/planner_demo/planner_demo_2/simulation_log/SimplePlanner/intersection_mixd/Scenario-jiangtong_intersection_9_3_2.json/jiangtong_intersection_9_3_2/jiangtong_intersection_9_3_2.pkl.xz
# /tmp/tmp87gba4nm/planner_demo/planner_demo_2/simulation_log/SimplePlanner/intersection_mixd/Scenario-dapai_intersection_1_3_4.json/dapai_intersection_1_3_4/dapai_intersection_1_3_4.pkl.xz
# /tmp/tmpbhxyj1b9/planner_demo/planner_demo_2/simulation_log/SimplePlanner/intersection_mixd/Scenario-jiangtong_intersection_9_3_2.json/jiangtong_intersection_9_3_2/jiangtong_intersection_9_3_2.pkl.xz
# /tmp/tmpwqaimhr_/planner_demo/planner_demo_2/simulation_log/SimplePlanner/intersection_mixd/Scenario-jiangtong_intersection_9_3_2.json/jiangtong_intersection_9_3_2/jiangtong_intersection_9_3_2.pkl.xz
selected_scenario_log = "/tmp/tmpwqaimhr_/planner_demo/planner_demo_2/simulation_log/SimplePlanner/intersection_mixd/Scenario-jiangtong_intersection_9_3_2.json/jiangtong_intersection_9_3_2/jiangtong_intersection_9_3_2.pkl.xz"
selected_scenario_log_p = pathlib.Path(selected_scenario_log)

simulation_log = SimulationLog.load_data(file_path=selected_scenario_log_p)
a = 2
