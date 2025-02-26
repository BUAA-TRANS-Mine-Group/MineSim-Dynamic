from typing import cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.observation_manager.abstract_observation import AbstractObservation


def build_observations(observation_cfg: DictConfig, scenario: AbstractScenario) -> AbstractObservation:
    """
    Instantiate observations
    :param observation_cfg: config of a planner
    :param scenario_info: scenario_info
    :return AbstractObservation
    """
    if False:
        # if is_TorchModuleWrapper_config(observation_cfg):
        # Build model and feature builders needed to run an ML model in simulation
        # torch_module_wrapper = build_torch_module_wrapper(observation_cfg.model_config)
        # model = LightningModuleWrapper.load_from_checkpoint(
        #     observation_cfg.checkpoint_path, model=torch_module_wrapper
        # ).model
        model = None

        # Remove config elements that are redundant to MLPlanner
        config = observation_cfg.copy()
        OmegaConf.set_struct(config, False)
        config.pop("model_config")
        config.pop("checkpoint_path")
        OmegaConf.set_struct(config, True)

        observation: AbstractObservation = instantiate(config, model=model, scenario_info=scenario_info)
    else:
        observation = cast(AbstractScenario, instantiate(observation_cfg, scenario=scenario))

    return observation
