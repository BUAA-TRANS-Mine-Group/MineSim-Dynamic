import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from devkit.script.builders.utils.utils_type import is_target_type, validate_type
from devkit.utils.multithreading.worker_pool import WorkerPool
from devkit.utils.multithreading.worker_ray import RayDistributed

logger = logging.getLogger(__name__)


def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """
    logger.info("#log# Building WorkerPool...")
    worker: WorkerPool = instantiate(cfg.worker, output_dir=cfg.output_dir) if is_target_type(cfg.worker, RayDistributed) else instantiate(cfg.worker)
    validate_type(worker, WorkerPool)

    logger.info("#log# Building WorkerPool...DONE!")
    return worker
