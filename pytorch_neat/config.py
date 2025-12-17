from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AdaptiveLinearNetConfig:
    """Hydra schema for AdaptiveLinearNet."""

    neat_config_path: str = "conf/algorithm/neat_config/adaptive_linear.cfg"
    weight_threshold: float = 0.2
    weight_max: float = 3.0
    activation: str = "tanh"
    cppn_activation: str = "identity"
    #batch_size: Optional[int] = None  # None -> fallback to experiment.trials
    #device: str = "cpu"
    input_coords: Optional[List[List[float]]] = None
    output_coords: Optional[List[List[float]]] = None


@dataclass
class AdaptiveNetConfig:
    """Hydra schema for AdaptiveNet."""

    neat_config_path: str = "conf/algorithm/neat_config/adaptive.cfg"
    weight_threshold: float = 0.2
    activation: str = "tanh"
    #batch_size: Optional[int] = None  # None -> fallback to experiment.trials
    #device: str = "cpu"
    input_coords: Optional[List[List[float]]] = None
    hidden_coords: Optional[List[List[float]]] = None
    output_coords: Optional[List[List[float]]] = None


@dataclass
class RecurrentNetConfig:
    """Hydra schema for RecurrentNet."""

    neat_config_path: str = "conf/algorithm/neat_config/recurrent.cfg"
    #batch_size: Optional[int] = None  # None -> fallback to experiment.trials
    activation: str = "sigmoid"
    prune_empty: bool = False
    use_current_activs: bool = False
    n_internal_steps: int = 1
    dtype: str = "float64"
    #device: str = "cpu"
