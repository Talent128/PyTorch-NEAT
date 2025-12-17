#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""
Hydra配置加载模块
"""
import importlib
from dataclasses import is_dataclass
from pathlib import Path

from experiment.experiment import Experiment, ExperimentConfig

_has_hydra = importlib.util.find_spec("hydra") is not None

if _has_hydra:
    from hydra import compose, initialize, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf


def load_experiment_from_hydra(
    cfg: DictConfig, task_name: str, algorithm_name: str
) -> Experiment:
    """从Hydra配置创建实验对象

    Args:
        cfg (DictConfig): Hydra配置字典
        task_name (str): 任务名称，格式为 "vmas/transport"
        algorithm_name (str): 算法名称，如 "recurrent", "adaptive_linear", "adaptive"

    Returns:
        Experiment: 实验对象
    """
    # 加载实验配置
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    
    # 加载算法配置
    algorithm_config = OmegaConf.to_object(cfg.algorithm)
    
    # 加载任务配置
    task_config = OmegaConf.to_object(cfg.task)
    
    return Experiment(
        task_name=task_name,
        algorithm_name=algorithm_name,
        algorithm_config=algorithm_config,
        task_config=task_config,
        experiment_config=experiment_config,
        seed=cfg.seed,
    )


def load_experiment_config_from_hydra(cfg: DictConfig) -> ExperimentConfig:
    """从Hydra配置加载实验配置

    Args:
        cfg (DictConfig): 实验配置字典

    Returns:
        ExperimentConfig: 实验配置对象
    """
    return OmegaConf.to_object(cfg)
