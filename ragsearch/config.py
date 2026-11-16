"""Configuration loading: config.yaml, optional config.local.yaml, CLI overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """A dict with attribute access (cfg.bm25.k1) that still serialises as a dict."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides (values parsed as YAML scalars)."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("override must look like section.key=value, got %r" % item)
        dotted, raw = item.split("=", 1)
        node = out
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Iterable[str]] = None) -> Config:
    """Load config.yaml, merge config.local.yaml (machine-specific), then overrides."""
    cfg_path = Path(path) if path else Path(os.environ.get("RAGSEARCH_CONFIG", ROOT / "config.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    local = cfg_path.parent / "config.local.yaml"
    if local.exists():
        with open(local, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    cfg = apply_overrides(cfg, overrides)
    return Config(cfg)


def resolve(cfg: Config, key: str) -> Path:
    """Resolve a path from cfg.paths relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p
