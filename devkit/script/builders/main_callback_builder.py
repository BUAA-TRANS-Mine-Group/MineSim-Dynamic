import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from devkit.script.builders.utils.utils_type import is_target_type, validate_type
from devkit.sim_engine.main_callback.abstract_main_callback import AbstractMainCallback
from devkit.sim_engine.main_callback.multi_main_callback import MultiMainCallback

logger = logging.getLogger(__name__)


def build_main_multi_callback(cfg: DictConfig) -> MultiMainCallback:
    """
    Build a multi main callback.
    :param cfg: Configuration that is used to run the experiment.
    """
    logger.info("#log# Building MultiMainCallback...")

    main_callbacks = []

    for callback_name, config in cfg.main_callback.items():
        # if is_target_type(config, MetricAggregatorCallback):
        if False:
            # Build metric aggregators and callbacks
            metric_aggregators = build_metrics_aggregators(cfg)
            callback: MetricAggregatorCallback = instantiate(config, metric_aggregators=metric_aggregators)
        else:
            callback = instantiate(config)
        validate_type(callback, AbstractMainCallback)
        main_callbacks.append(callback)

    multi_main_callback = MultiMainCallback(main_callbacks)
    logger.info(f"#log# Building MultiMainCallback: {len(multi_main_callback)}...DONE!")

    return multi_main_callback
