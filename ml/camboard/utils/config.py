"""Configuration loading and validation utilities using OmegaConf."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load a YAML config file and optionally merge CLI dot-separated overrides.

    Args:
        path: Path to YAML configuration file.
        overrides: List of key=value override strings (e.g. ['training.lr=1e-4']).

    Returns:
        Resolved OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(str(path))
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def save_config(cfg: DictConfig | Dict[str, Any], path: str | Path) -> None:
    """Save resolved config to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    with open(p, "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f)


def config_to_dict(cfg: DictConfig | Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OmegaConf DictConfig or dict to a standard Python dictionary."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    return dict(cfg)

