import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from devkit.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from devkit.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_scenario_builder(cfg: DictConfig) -> AbstractScenarioBuilder:
    """
    Builds scenario builder.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of scenario builder.
    """
    logger.info("#log# Building AbstractScenarioBuilder...")
    scenario_builder = instantiate(cfg.scenario_builder)
    validate_type(scenario_builder, AbstractScenarioBuilder)
    logger.info("#log# Building AbstractScenarioBuilder...DONE!")
    return scenario_builder
