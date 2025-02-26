import logging
import os
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union, cast

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from pandas.errors import EmptyDataError


from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.script.builders.scenario_building_builder import build_scenario_builder
from devkit.script.builders.scenario_filter_builder import build_scenario_filter

# from devkit.utils.multithreading.worker_pool import WorkerPool
from devkit.utils.multithreading.worker_utils import chunk_list

logger = logging.getLogger(__name__)


class DistributedMode(Enum):
    """
    Constants to use to look up types of distributed processing for DistributedScenarioFilter
    They are:
    :param SCENARIO_BASED: Works in two stages, first getting a list of all, scenarios to process,
                           then breaking up that list and distributing across the workers
    :param LOG_FILE_BASED: Works in a single stage, breaking up the scenarios based on what log file they are in and
                           distributing the number of log files evenly across all workers
    :param SINGLE_NODE: Does no distribution, processes all scenarios in config

    用于查找分布式处理类型的常量：
    用于 DistributedScenarioFilter 的分布式处理类型：
    ：param SCENARIO_BASED： 分两步执行，首先获取要处理的所有场景的列表，然后将该列表拆分并分发到各个 Worker 中。
    ：param LOG_FILE_BASED： 分单个阶段执行，根据场景所在的日志文件将场景拆分，并将日志文件的数量平均分发到所有 Worker 中。
    ：param SINGLE_NODE： 不进行分发，在配置中处理所有场景。
    """

    SCENARIO_BASED = "scenario_based"
    LOG_FILE_BASED = "log_file_based"
    SINGLE_NODE = "single_node"


class DistributedScenarioFilter:
    """
    Class to distribute the work to build / filter scenarios across workers, and to break up those scenarios in chunks to be
    handled on individual machines

    用于将构建/过滤场景的worker分配给多个workers，并将这些场景分成小块，以便在单个机器上处理。
    """

    def __init__(
        self,
        cfg: DictConfig,
        # worker: WorkerPool,
        node_rank: int,
        num_nodes: int,
        # synchronization_path: str,
        timeout_seconds: int = 7200,
        distributed_mode: DistributedMode = DistributedMode.SCENARIO_BASED,
    ):
        """
        :param cfg: top level config for the job (used to build scenario builder / scenario_filter)
        :param worker: worker to use in each node to parallelize the work
        :param node_rank: number from (0, num_nodes -1) denoting "which" node we are on
        :param num_nodes: total number of nodes the job is running on
        :param synchronization_path: path that can be in s3 or on a shared file system that will be used to synchronize
                                     across workers
        :param timeout_seconds: how long to wait during sync operations
        :param distributed_mode: what distributed mode to use to distribute computation

        ：param cfg：作业的顶级配置（用于构建scenario builder / scenario filter）
        ：param worker：每个节点中用于并行化工作的worker
        ：param node_rank：从（0,num_nodes -1）开始的数字，表示我们在“哪个”节点上
        ：param num_nodes：作业正在运行的节点总数
        ：param synchronization_path： s3或共享文件系统中用于跨worker同步的路径
        ：param timeout_seconds：同步操作需要等待的时间
        ：param distributed_mode：使用哪种分布式模式来分配计算

        """
        self._cfg = cfg
        # self._worker = worker
        self._node_rank = node_rank
        self._num_nodes = num_nodes
        # self.synchronization_path = synchronization_path
        self._timeout_seconds = timeout_seconds
        self._distributed_mode = distributed_mode

    def get_scenarios(self) -> List[AbstractScenario]:
        """
        Get all the scenarios that the current node should process
        :returns: list of scenarios for the current node
        """
        if self._num_nodes == 1 or self._distributed_mode == DistributedMode.SINGLE_NODE:
            logger.info("#log# Building Scenarios in mode %s", DistributedMode.SINGLE_NODE)
            scenario_builder = build_scenario_builder(cfg=self._cfg)
            scenario_filter = build_scenario_filter(cfg=self._cfg.scenario_filter)
        elif False:
            # elif self._distributed_mode in (DistributedMode.LOG_FILE_BASED, DistributedMode.SCENARIO_BASED):
            logger.info("#log# Getting Log Chunks")
            current_chunk = self._get_log_db_files_for_single_node()

            logger.info("#log# Getting Scenarios From Log Chunk of size %d", len(current_chunk))
            scenarios = self._get_scenarios_from_list_of_log_files(current_chunk)

            if self._distributed_mode == DistributedMode.LOG_FILE_BASED:
                logger.info(
                    "#log# Distributed mode is %s, so we are just returning the scenarios"
                    "found from log files on the current worker.  There are %d scenarios to process"
                    "on node %d/%d",
                    DistributedMode.LOG_FILE_BASED,
                    len(scenarios),
                    self._node_rank,
                    self._num_nodes,
                )
                return scenarios

            logger.info(
                "#log# Distributed mode is %s, so we are going to repartition the "
                "scenarios we got from the log files to better distribute the work",
                DistributedMode.SCENARIO_BASED,
            )
            logger.info("#log# Getting repartitioned scenario tokens")
            tokens, log_db_files = self._get_repartition_tokens(scenarios)

            OmegaConf.set_struct(self._cfg, False)
            self._cfg.scenario_filter.scenario_tokens = tokens
            self._cfg.scenario_builder.db_files = log_db_files
            OmegaConf.set_struct(self._cfg, True)

            logger.info("#log# Building repartitioned scenarios")
            scenario_builder = build_scenario_builder(cfg=self._cfg)
            scenario_filter = build_scenario_filter(cfg=self._cfg.scenario_filter)
        else:
            raise ValueError(
                f"#log# Distributed mode must be one of " f"{[x.name for x in fields(DistributedMode)]}, " f"got {self._distributed_mode} instead!"
            )
        scenarios = scenario_builder.get_scenarios(scenario_filter, self._worker)
        return scenarios
