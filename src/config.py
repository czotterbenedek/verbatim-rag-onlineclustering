from pathlib import Path

import yaml


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_dir"] = str(config_path.parent)
    return config


def config_path(config, value):
    path = Path(value)
    return path if path.is_absolute() else Path(config["_config_dir"]) / path


def write_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {key: value for key, value in config.items() if key != "_config_dir"}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(stored, handle, sort_keys=False)