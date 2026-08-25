"""
Configuration loader for smart-adaptive-headlight.

All YAML configs are loaded via `get_config(name)` and cached in memory.
Config files live in the same directory as this module.
"""

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_cache: dict[str, dict[str, Any]] = {}


def get_config(name: str) -> dict[str, Any]:
    """Load and cache a YAML config by base name (without extension).

    Args:
        name: Config file base name, e.g. ``"zones"`` loads ``zones.yaml``.

    Returns:
        Parsed YAML as a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    if name in _cache:
        return _cache[name]

    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    _cache[name] = data
    return data


def reload_config(name: str) -> dict[str, Any]:
    """Force-reload a config (bypasses cache)."""
    _cache.pop(name, None)
    return get_config(name)


def get_all_configs() -> dict[str, dict[str, Any]]:
    """Load and return all standard configs."""
    names = ["zones", "thresholds", "camera_calib"]
    return {n: get_config(n) for n in names}


if __name__ == "__main__":
    # Demo: print all configs
    for cfg_name in ("zones", "thresholds", "camera_calib"):
        try:
            cfg = get_config(cfg_name)
            print(f"\n=== {cfg_name} ===")
            print(yaml.dump(cfg, default_flow_style=False))
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
