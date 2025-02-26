import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import DictConfig

from devkit.common.utils.file_backed_barrier import distributed_sync
from devkit.script.builders.logging_builder import build_logger
from devkit.script.builders.main_callback_builder import build_main_multi_callback
from devkit.script.builders.utils.utils_config import update_config_for_simulation
from devkit.script.builders.worker_pool_builder import build_worker
from devkit.sim_engine.main_callback.multi_main_callback import MultiMainCallback
from devkit.sim_engine.runner.abstract_runner import AbstractRunner
from devkit.sim_engine.runner.executor import execute_runners
from devkit.sim_engine.runner.runner_report import RunnerReport
from devkit.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)


@dataclass
class CommonBuilder:
    """Common builder data."""

    worker: WorkerPool
    multi_main_callback: MultiMainCallback
    output_dir: Path
    # profiler: ProfileCallback


def set_default_path_by_environment_variable() -> None:
    """设置 MINESIM_DATA_ROOT 和 MINESIM_MAPS_ROOT 环境变量到os中"""
    if "MINESIM_DATA_ROOT" not in os.environ:
        os.environ["MINESIM_DATA_ROOT"] = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets"
    os.environ["MINESIM_DYNAMIC_SCENARIO_ROOT"] = (
        "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/scenarios/dynamic_obstacle_scenarios"
    )
    os.environ["MINESIM_MAPS_ROOT"] = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/datasets/maps"
    os.environ["MINESIM_INPUTS_SCENARIO_ROOT"] = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/inputs"
    os.environ["MINESIM_EXP_ROOT"] = "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/outputs/dynamic_scenario"


def save_runner_reports(reports: List[RunnerReport], output_dir: Path, report_name: str) -> None:
    """
    Save runner reports to a parquet file in the output directory.
    Output directory can be local or s3.
    :param reports: Runner reports returned from each simulation.
    :param output_dir: Output directory to save the report.
    :param report_name: Report name.
    """
    save_path = output_dir / report_name
    # df.to_parquet(safe_path_to_string(save_path))
    logger.info(f"#log# Saved runner reports to {save_path}")


def set_up_common_builder(cfg: DictConfig, profiler_name: str = None) -> CommonBuilder:
    """

    Set up a common builder when running simulations.
    :param cfg: Hydra configuration.
    :param profiler_name: Profiler name.
    :return A data classes with common builders.

    profiler_name: 省略.引入 PyTorch Lightning 库，用于高级训练流程管理. ProfileCallback 用于输出性能分析的 HTML 报告。
    """
    # Build multi main callback
    multi_main_callback = build_main_multi_callback(cfg)

    # After run_simulation start
    multi_main_callback.on_run_simulation_start()

    # Update and override configs for simulation
    update_config_for_simulation(cfg=cfg)

    # Configure logger
    build_logger(cfg)

    # Construct builder
    worker = build_worker(cfg)

    # Simulation Callbacks
    output_dir = Path(cfg.output_dir)

    return CommonBuilder(
        worker=worker,
        multi_main_callback=multi_main_callback,
        output_dir=output_dir,
    )


def run_runners(runners: List[AbstractRunner], common_builder: CommonBuilder, profiler_name: str, cfg: DictConfig) -> None:
    """
    Run a list of runners.
    :param runners: A list of runners.
    :param common_builder: Common builder.
    :param profiler_name: Profiler name.
    :param cfg: Hydra config.
    """
    assert len(runners) > 0, "No scenarios found to simulate!"

    logger.info("#log# Executing runners...\n")
    #! 核心
    reports = execute_runners(
        runners=runners,
        worker=common_builder.worker,
        num_gpus=cfg.number_of_gpus_allocated_per_simulation,
        num_cpus=cfg.number_of_cpus_allocated_per_simulation,
        exit_on_failure=cfg.exit_on_failure,
        verbose=cfg.verbose,
    )
    logger.info("#log# Finished executing runners!\n")

    # Save RunnerReports as parquet file
    save_runner_reports(reports, common_builder.output_dir, cfg.runner_report_file)

    # Sync up nodes when running distributed simulation
    distributed_sync(Path(cfg.output_dir / Path("barrier")), cfg.distributed_timeout_seconds)

    # # Only run on_run_simulation_end callbacks on master node
    if int(os.environ.get("NODE_RANK", 0)) == 0:
        common_builder.multi_main_callback.on_run_simulation_end()
