import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, cast

from omegaconf import DictConfig

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.script.builders.scenario_building_builder import build_scenario_builder
from devkit.script.builders.scenario_filter_builder import build_scenario_filter
from devkit.utils.multithreading.worker_utils import WorkerPool

logger = logging.getLogger(__name__)


def get_local_scenario_cache(cache_path: str, feature_names: Set[str]) -> List[Path]:
    """
    Get a list of cached scenario paths from a local cache.
    :param cache_path: Root path of the local cache dir.
    :param feature_names: Set of required feature names to check when loading scenario paths from the cache.
    :return: List of discovered cached scenario paths.
    """
    cache_dir = Path(cache_path)
    assert cache_dir.exists(), f"#log# Local cache {cache_dir} does not exist!"
    assert any(cache_dir.iterdir()), f"No files found in the local cache {cache_dir}!"

    candidate_scenario_dirs = {x.parent for x in cache_dir.rglob("*.gz")}

    # Keep only dir paths that contains all required feature names
    scenario_cache_paths = [path for path in candidate_scenario_dirs if not (feature_names - {feature_name.stem for feature_name in path.iterdir()})]

    return scenario_cache_paths


def extract_scenarios_from_dataset(cfg: DictConfig, worker: WorkerPool) -> List[AbstractScenario]:
    """
    Extract and filter scenarios by loading a dataset using the scenario builder.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: List of extracted scenarios.
    """
    scenario_builder = build_scenario_builder(cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, worker)

    return scenarios


def build_scenarios(cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper) -> List[AbstractScenario]:
    """
    Build the scenario objects that comprise the training dataset.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :param model: NN model used for training.
    :return: List of extracted scenarios.
    """
    scenarios = (
        extract_scenarios_from_cache(cfg, worker, model) if cfg.cache.use_cache_without_dataset else extract_scenarios_from_dataset(cfg, worker)
    )

    logger.info(f"Extracted {len(scenarios)} scenarios for training")
    assert len(scenarios) > 0, "No scenarios were retrieved for training, check the scenario_filter parameters!"

    return scenarios


def validate_scenario_type_in_cache_path(paths: List[Path]) -> None:
    """
    Checks if scenario_type is in cache path.
    :param path: Scenario cache path
    :return: Whether scenario type is in cache path
    """
    sample_cache_path = paths[0]
    assert all(
        not char.isdigit() for char in sample_cache_path.parent.name
    ), "Unable to filter cache by scenario types as it was generated without scenario type information. Please regenerate a new cache if scenario type filtering is required."


# def create_scenario_from_paths(paths: List[Path]) -> List[AbstractScenario]:
#     """
#     Create scenario objects from a list of cache paths in the format of ".../log_name/scenario_token".
#     :param paths: List of paths to load scenarios from.
#     :return: List of created scenarios.
#     """
#     scenarios = [
#         CachedScenario(
#             log_name=path.parent.parent.name,
#             token=path.name,
#             scenario_type=path.parent.name,
#         )
#         for path in paths
#     ]

#     return scenarios
