"""Configuration loading and canonical experiment fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when a configuration or override is invalid."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError("configuration contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ConfigError(f"unsupported configuration value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError(f"override must be KEY=VALUE: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        value = cursor.get(part)
        if not isinstance(value, dict):
            raise ConfigError(f"unknown override: {dotted_key}")
        cursor = value
    leaf = parts[-1]
    if leaf not in cursor:
        raise ConfigError(f"unknown override: {dotted_key}")
    try:
        cursor[leaf] = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid override value for {dotted_key}: {raw_value}") from error


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot load configuration {config_path}: {error}") from error
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    config = dict(loaded)
    for expression in overrides:
        _apply_override(config, expression)
    canonical_json(config)
    return config

