from typing import Dict, List, Optional, Type, cast

from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.script.builders.utils.utils_type import is_target_type
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.common.actor_state.vehicle_parameters import get_mine_truck_parameters

# TODO 先不移植 MLPlanner
# from devkit.sim_engine.planning.planner.ml_planner.ml_planner import MLPlanner
# from devkit.sim_engine.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper


def _build_planner(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> AbstractPlanner:
    """
    Instantiate planner
    :param planner_cfg: config of a planner
    :param scenario: scenario
    :return AbstractPlanner
    """
    config = planner_cfg.copy()
    if False:
        pass  # TODO 先不移植 MLPlanner
    else:
        # 定位配置的 planner 是可以正确初始化
        planner_cls: Type[AbstractPlanner] = _locate(config._target_)

        if planner_cls.requires_scenario:
            assert scenario is not None, "Scenario was not provided to build the planner. " f"Planner {config} can not be build!"
            planner = cast(typ=AbstractPlanner, val=instantiate(config=config, scenario=scenario))
        else:
            planner = cast(AbstractPlanner, instantiate(config))

    return planner


def build_planners(planners_cfg: DictConfig, scenario: AbstractScenario, cache: Dict[str, AbstractPlanner] = dict()) -> List[AbstractPlanner]:
    """
    Instantiate multiple planners by calling build_planner
    :param planners_cfg: planners config
    :param scenario: scenario
    :return planners: List of AbstractPlanners
    """
    planners = []
    for name in planners_cfg:
        config = planners_cfg[name].copy()
        thread_safe = config.thread_safe
        # Remove the thread_safe element given that it's used here
        OmegaConf.set_struct(config, False)
        config.pop("thread_safe")
        OmegaConf.set_struct(config, True)
        # Build the planner making sure to keep only 1 instance if it's non-thread-safe
        planner = cache.get(name, _build_planner(config, scenario))
        planners.append(planner)
        if not thread_safe and name not in cache:
            cache[name] = planner
    return planners
