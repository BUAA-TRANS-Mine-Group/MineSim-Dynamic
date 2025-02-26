import logging
import pathlib
from typing import List, Optional

# import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from devkit.script.builders.utils.utils_type import is_target_type, validate_type
from devkit.sim_engine.callback.abstract_callback import AbstractCallback
from devkit.sim_engine.callback.serialization_callback import SerializationCallback
from devkit.sim_engine.callback.simulation_log_callback import SimulationLogCallback

# 导入单机并行执行器 SingleMachineParallelExecutor，用于多进程或多线程处理任务
from devkit.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

# 导入 WorkerPool(工作池) 和 WorkerResources(工作资源信息)，用于管理多线程/多进程资源
from devkit.utils.multithreading.worker_pool import WorkerPool, WorkerResources

# 导入顺序执行器 Sequential，表示在单线程里顺序执行任务
from devkit.utils.multithreading.worker_sequential import Sequential

logger = logging.getLogger(__name__)
# 创建一个与当前模块同名的 logger，用于记录日志


def build_callbacks_worker(cfg: DictConfig) -> Optional[WorkerPool]:
    """
    Builds workerpool for callbacks.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Workerpool, or None if we'll run without one.
    """
    # 如果配置指定了不使用并行回调或者当前 worker 配置不符合 Sequential 类型，就返回 None
    if not is_target_type(cfg.worker, Sequential) or cfg.disable_callback_parallelization:
        return None

    # 如果分配给单次仿真的 CPU 数量不是 None 或 1，则抛出异常 # 因为 Sequential worker 下只应该分配 1 个 CPU
    if cfg.number_of_cpus_allocated_per_simulation not in [None, 1]:
        raise ValueError("Expected `number_of_cpus_allocated_per_simulation` to be set to 1 with Sequential worker.")

    # 计算最大可用的并行 worker 数量：当前节点的可用 CPU 数量 减去 分配的 CPU，然后与配置的最大回调进程数取最小值
    max_workers = min(
        WorkerResources.current_node_cpu_count() - (cfg.number_of_cpus_allocated_per_simulation or 1),
        cfg.max_callback_workers,
    )

    # 创建一个单机并行执行器，启用进程池，并设置最大 worker 数
    callbacks_worker_pool = SingleMachineParallelExecutor(use_process_pool=True, max_workers=max_workers)
    return callbacks_worker_pool  # 返回构建好的回调 worker 池


def build_simulation_callbacks(cfg: DictConfig, output_dir: pathlib.Path, worker: Optional[WorkerPool] = None) -> List[AbstractCallback]:
    """
    Builds callback.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param output_dir: directory for all experiment results.
    :param worker: to run certain callbacks in the background (everything runs in main process if None).
    :return: List of callbacks.
    """
    logger.info("#log# Building AbstractCallback...")
    callbacks = []  # 返回构建好的回调 worker 池

    # 遍历配置文件中所有 callback 的配置
    for config in cfg.callback.values():
        # 如果配置的类型是 SerializationCallback，则实例化并传入 output_directory 参数
        if is_target_type(config, SerializationCallback):
            callback: SerializationCallback = instantiate(config, output_directory=output_dir)

        # 如果配置的类型是 TimingCallback，则创建一个 TensorBoard writer 并将其注入实例化
        # elif is_target_type(config, TimingCallback):
        elif False:
            tensorboard = torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)
            callback = instantiate(config, writer=tensorboard)

        # 如果配置是 SimulationLogCallback 或 MetricCallback，这些回调会在 simulation builder 中初始化，因此这里跳过
        # elif is_target_type(config, SimulationLogCallback) or is_target_type(config, MetricCallback):
        elif is_target_type(config, SimulationLogCallback):
            # SimulationLogCallback and MetricCallback store state (futures) from each runner, so they are initialized in the simulation builder
            continue

        else:  # 其它情况直接根据配置实例化回调
            callback = instantiate(config)

        # 验证实例化的对象是否是 AbstractCallback 类型，不是则抛出错误
        validate_type(callback, AbstractCallback)
        # 将回调加入列表
        callbacks.append(callback)

    logger.info(f"#log# Building AbstractCallback: {len(callbacks)}...DONE!")
    return callbacks  # 返回创建好的回调列表
