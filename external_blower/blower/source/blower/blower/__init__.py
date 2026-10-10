"""Blower extension package for Isaac Lab.

The package root intentionally uses lazy exports. This keeps pure numerical
imports such as ``blower.models`` usable in a normal Python interpreter, while
Isaac Lab/Omniverse-heavy modules are imported only when explicitly requested.
"""

from __future__ import annotations

from importlib import import_module

_PUBLIC_EXPORTS = {
    'AirflowActionTerm': ('blower.mdp', 'AirflowActionTerm'),
    'AirflowActionTermCfg': ('blower.mdp', 'AirflowActionTermCfg'),
    'AirflowTargetCfg': ('blower.mdp', 'AirflowTargetCfg'),
    'AirflowModel': ('blower.models', 'AirflowModel'),
    'AirflowModelCfg': ('blower.models', 'AirflowModelCfg'),
    'BallCfg': ('blower.objects', 'BallCfg'),
    'BoxCfg': ('blower.objects', 'BoxCfg'),
    'CuboidCfg': ('blower.objects', 'CuboidCfg'),
    'BottleCfg': ('blower.objects', 'BottleCfg'),
    'CigaretteCfg': ('blower.objects', 'CigaretteCfg'),
    'SIMPLE_BLOWER_CFG': ('blower.assets', 'SIMPLE_BLOWER_CFG'),
    'JetCfg': ('blower.configs', 'JetCfg'),
    'DragCfg': ('blower.configs', 'DragCfg'),
    'SamplingCfg': ('blower.configs', 'SamplingCfg'),
    'VizCfg': ('blower.configs', 'VizCfg'),
    'JetConeVisualizer': ('blower.viz', 'JetConeVisualizer'),
    'SampleMarkerVisualizer': ('blower.viz', 'SampleMarkerVisualizer'),
    'LaserRayVisualizer': ('blower.viz', 'LaserRayVisualizer'),
}


def __getattr__(name: str):
    if name not in _PUBLIC_EXPORTS:
        raise AttributeError(f"module 'blower' has no attribute {name!r}")
    module_name, attr_name = _PUBLIC_EXPORTS[name]
    return getattr(import_module(module_name), attr_name)


__all__ = list(_PUBLIC_EXPORTS)
