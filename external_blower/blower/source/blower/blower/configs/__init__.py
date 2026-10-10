"""Blower configuration exports."""

from __future__ import annotations

from importlib import import_module

from .airflow_cfg import DragCfg, JetCfg, SamplingCfg, VizCfg

_LAZY_EXPORTS = {
    'SIMPLE_BLOWER_CFG': ('blower.configs.blower_cfg', 'SIMPLE_BLOWER_CFG'),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'blower.configs' has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    return getattr(import_module(module_name), attr_name)


__all__ = [
    'JetCfg',
    'DragCfg',
    'SamplingCfg',
    'VizCfg',
    'SIMPLE_BLOWER_CFG',
]
