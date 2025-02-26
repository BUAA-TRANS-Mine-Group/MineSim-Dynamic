import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from devkit.script.builders.utils.utils_type import validate_type
from devkit.sim_engine.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

logger = logging.getLogger(__name__)


def build_torch_module_wrapper(cfg: DictConfig) -> TorchModuleWrapper:
    """
    Builds the NN module.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of TorchModuleWrapper.
    """
    logger.info("#log# Building TorchModuleWrapper...")
    model = instantiate(cfg)
    validate_type(model, TorchModuleWrapper)
    logger.info("#log# Building TorchModuleWrapper...DONE!")

    return model
