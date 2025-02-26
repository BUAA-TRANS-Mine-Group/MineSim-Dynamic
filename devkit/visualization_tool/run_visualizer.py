# Python library
import gc
import sys
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging
import csv

# Third-party library
import hydra
import matplotlib
import matplotlib.pyplot as plt

import pathlib
import pickle  # Python 内置序列化工具
from concurrent.futures import ThreadPoolExecutor  # 用于在多个线程中并发执行任务
from functools import partial  # 用于固定部分函数参数的便捷工具
from pathlib import Path  # pathlib.Path 用于文件路径管理

# Local library, `minesim.devkit` dir is necessary
dir_current_file = os.path.dirname(__file__)
sys.path.append(dir_current_file)
dir_parent_1 = os.path.dirname(dir_current_file)
sys.path.append(dir_parent_1)
dir_parent_2 = os.path.dirname(dir_parent_1)
sys.path.append(dir_parent_2)
dir_parent_3 = os.path.dirname(dir_parent_2)
sys.path.append(dir_parent_3)


from devkit.sim_engine.log_api.simulation_log import SimulationLog
from devkit.visualization_tool.visualizer_2D import PlanVisualizer2D

logger = logging.getLogger(__name__)


def get_simulation_log(log_file_list: str = None, log_number: int = None):
    """
    读取 log_file_list.csv 中 `number` 列 = log_number 的那一行,
    提取其 log_file_path 字段, 并执行后续处理。
    """
    if not log_file_list or not os.path.isfile(log_file_list):
        logger.error(f"#log# log_file_list.csv 文件不存在: {log_file_list}")
        return

    if log_number is None:
        logger.error("#log# log_number 为空，请指定要读取的日志编号。")
        return

    selected_scenario_log = None

    # 1. 打开 CSV 文件，读取其中的行，查找第 log_number 条记录
    with open(log_file_list, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # 跳过表头
        header = next(reader, None)
        for row in reader:
            # 确保当前行有内容，且第一列为 number
            if row and len(row) >= 3:
                try:
                    current_num = int(row[0])  # number 列
                except ValueError:
                    continue  # 如果第一列无法转换为整型，跳过

                if current_num == log_number:
                    # 找到了与 log_number 匹配的行
                    selected_scenario_log = row[2]  # 第三列为 log_file_path
                    break
    if not selected_scenario_log:
        logger.error(f"#log# 在 {log_file_list} 中未找到 number={log_number} 的日志记录。")
        return
    
  
    # 2. 将日志地址转换为 pathlib.Path
    selected_scenario_log_p = pathlib.Path(selected_scenario_log)
    if not selected_scenario_log_p.is_file():
        logger.error(f"#log# 指定的日志文件不存在: {selected_scenario_log_p}")
        return

    # 分割路径层级列表
    parts = selected_scenario_log_p.parts
    try:
        # /outputs/outputs_log/log_2025-02-25_11-53-12_945fa6d3/simulation_mode_4/
        # 找到 "outputs_log" 在路径中的索引
        log_index = parts.index("outputs_log")
        # 动态日志目录名是下一级目录（index+1）
        dynamic_part_log_name = parts[log_index + 1]
    except ValueError:
        # 如果未找到 "outputs_log"（理论上不应触发，因已验证路径有效性）
        dynamic_part_log_name = None
        raise RuntimeError("Path does not contain 'outputs_log'")
        
    # 3. 调用 SimulationLog.load_data 加载日志
    logger.info(f"#log# 开始加载仿真日志文件: {selected_scenario_log_p}")
    simulation_log = SimulationLog.load_data(file_path=selected_scenario_log_p)
    logger.info(f"#log# 已成功加载仿真日志文件: {selected_scenario_log_p}")

    return simulation_log,dynamic_part_log_name


def run_visualizer(simulation_log: SimulationLog,dynamic_part_log_name:str):
    """
    可视化、分析等
    """
    visualizer_plan = PlanVisualizer2D(bitmap_type="bitmap_mask")
    # visualizer_plan = PlanVisualizer(bitmap_type="bitmap_rgb")
    visualizer_plan.init(scenario=simulation_log.scenario, planner=simulation_log.planner,dynamic_part_log_name=dynamic_part_log_name)
    visualizer_plan.plot_scenario(
        scenario=simulation_log.scenario,
        planner=simulation_log.planner,
        flag_plot_route_refpath=False,
    )
    visualizer_plan.get_simulatiion_all_frames(
        scenario=simulation_log.scenario, planner=simulation_log.planner, simulation_history=simulation_log.simulation_history
    )

    # 绘制 simulation 过程可视化场景信息
    for index, ego_state in enumerate(simulation_log.simulation_history.extract_ego_state):
        # --------- 场景基本要素更新绘制 ---------
        visualizer_plan.update_base_info(scenario=simulation_log.scenario, simulation_history=simulation_log.simulation_history, index=index)
        visualizer_plan.update_all_local_planning_info(
            scenario=simulation_log.scenario, planner=simulation_log.planner, simulation_history=simulation_log.simulation_history, index=index
        )

        # visualizer_plan.save_figure_as_png()
        visualizer_plan.save_figure_as_svg()  # !需要paper绘制矢量图再开启，耗时
        a = 1


if __name__ == "__main__":
    # log_file_list = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/MineSim-Dynamic-Dev/outputs/log_file_list.csv"
    # log_number = 2
    from devkit.visualization_tool.configuration.visualizer_conf import log_file_list
    from devkit.visualization_tool.configuration.visualizer_conf import log_number

    simulation_log,dynamic_part_log_name = get_simulation_log(log_file_list=log_file_list, log_number=log_number)
    run_visualizer(simulation_log=simulation_log,dynamic_part_log_name=dynamic_part_log_name)
