"""
Config loader and model registry for EELS training.

- Load full training + model config from YAML (e.g. configs/paper_unet.yaml).
- Model registry: get_model(name, **kwargs) so you can switch models via config
  while keeping the same training protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Default configs directory next to this file
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


@dataclass
class TrainConfig:
    """Full training + data + model config (from YAML or defaults)."""

    # Data
    root: str = "EELS"
    batch_size: int = 32
    num_workers: int = 4

    # Training protocol (important for reproducibility)
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps_per_epoch: int = 1000

    # Loss
    loss_mode: str = "bce_softf1"
    lambda_soft_f1: float = 1.0

    # Logging / selection
    threshold: float = 0.8
    save_dir: str = "checkpoints"

    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu: bool = False

    # Smoke run
    smoke: bool = False
    smoke_max_train_batches: int = 2
    smoke_max_val_batches: int = 1

    # Model: registry name + kwargs passed to the model constructor
    model_name: str = "paper_unet"
    model_kwargs: dict[str, Any] = field(default_factory=lambda: {"num_classes": 80, "activation": "relu", "dropout_p": 0.3})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        """Build TrainConfig from a flat or nested dict (e.g. from YAML)."""
        # Flatten nested "model" key into model_name + model_kwargs
        model_cfg = d.pop("model", None)
        if isinstance(model_cfg, dict):
            model_name = model_cfg.pop("name", "paper_unet")
            model_kwargs = model_cfg  # rest is kwargs
        else:
            model_name = d.pop("model_name", "paper_unet")
            model_kwargs = d.pop("model_kwargs", {})

        # Only pass keys that exist on TrainConfig
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        valid.discard("model_name")
        valid.discard("model_kwargs")
        kwargs = {k: v for k, v in d.items() if k in valid}
        kwargs["model_name"] = model_name
        kwargs["model_kwargs"] = model_kwargs
        return cls(**kwargs)


def load_config(path: str | Path) -> TrainConfig:
    """
    Load TrainConfig from a YAML file.
    path: path to .yaml file, or a short name like 'paper_unet' (then configs/paper_unet.yaml is used).
    """
    path = Path(path)
    if not path.suffix:
        path = CONFIGS_DIR / f"{path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML configs. Install: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a dict; got {type(data)}")
    return TrainConfig.from_dict(data)


# ----- Model registry -----
# Maps model_name -> (class, default_kwargs). Add new models here to experiment.
_MODEL_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {}


def register_model(name: str, default_kwargs: dict[str, Any] | None = None):
    """Decorator to register a model class under a given name."""

    def decorator(clz: type):
        _MODEL_REGISTRY[name] = (clz, default_kwargs or {})
        return clz

    return decorator


def get_model(name: str, **kwargs) -> torch.nn.Module:
    """
    Build model by registry name. kwargs are passed to the model constructor;
    they are merged with the registered default_kwargs (kwargs override).
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model {name!r}. Available: {available}")
    clz, default_kwargs = _MODEL_REGISTRY[name]
    merged = {**default_kwargs, **kwargs}
    return clz(**merged)


def list_models() -> list[str]:
    """Return list of registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


# Register the current successful paper UNet (from model.py)
from model import PaperUNet1D

register_model("paper_unet", {"num_classes": 80, "activation": "relu", "dropout_p": 0.3})(PaperUNet1D)

# To add another model for experiments, register it and add a config YAML, e.g.:
#   from model_old import EELSModel as OldEELSModel
#   register_model("old_prototype", {})(OldEELSModel)
# Then create configs/old_prototype.yaml with model.name: old_prototype and same training protocol.
