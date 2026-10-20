"""YAML configuration with attribute access and recursive defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Cfg(dict):
    """A dict whose keys are also attributes (nested dicts are wrapped lazily)."""

    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
        except KeyError as exc:  # pragma: no cover - mirrors attribute protocol
            raise AttributeError(key) from exc
        if isinstance(v, dict) and not isinstance(v, Cfg):
            v = Cfg(v)
            self[key] = v
        return v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> Cfg:
    """Load ``path``; if it has a ``base:`` key, merge it on top of that file first."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base_path = None
    if "base" in raw:
        base_path = (path.parent / raw.pop("base")).resolve()
        base = dict(load_config(base_path))
        raw = _merge(base, raw)
    if overrides:
        raw = _merge(raw, overrides)
    cfg = Cfg(raw)
    cfg["_config_path"] = str(path)
    cfg["_base_config_path"] = str(base_path) if base_path else None    # the profile this file modifies, if any
    return cfg


def apply_dotted_overrides(cfg: Cfg, items: list[str]) -> Cfg:
    """Apply ``key.sub=value`` command-line overrides (values parsed as YAML)."""
    out = dict(cfg)
    for item in items or []:
        key, _, val = item.partition("=")
        value = yaml.safe_load(val)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return Cfg(out)
