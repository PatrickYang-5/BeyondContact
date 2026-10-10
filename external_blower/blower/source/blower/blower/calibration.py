"""Validated, portable physical parameters shared by replay, bridge and RL."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path


PHYSICAL_KEYS = (
    "cube_size_m", "cube_mass_kg", "cube_static_friction", "cube_dynamic_friction",
    "ground_static_friction", "ground_dynamic_friction", "drag_coefficient",
    "base_link_world_z", "jet_U0", "jet_spread_k", "jet_xc", "jet_decay_k", "nozzle_radius",
)


def load_profile(path):
    profile = json.loads(Path(path).read_text())
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported calibration schema_version")
    if "fan_response" in profile:
        for key in ("delay_s", "rise_s", "fall_s"):
            value = profile["fan_response"][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid fan response: {key}")
    p = profile["physical"]
    for key in PHYSICAL_KEYS:
        value = p[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Invalid physical parameter: {key}")
        if key != "base_link_world_z" and value < 0:
            raise ValueError(f"Negative physical parameter: {key}")
    for key in ("cube_size_m", "cube_mass_kg", "drag_coefficient", "jet_U0", "nozzle_radius"):
        if p[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for prefix in ("cube", "ground"):
        if p[f"{prefix}_static_friction"] < p[f"{prefix}_dynamic_friction"]:
            raise ValueError(f"{prefix} static friction must be >= dynamic friction")
    return profile


def apply_to_rl(env_cfg, profile):
    """Opt-in deterministic nominal physics; avoids randomization undoing calibration.

    The support surface is Table in the Piper gripper-assist task. This does not
    calibrate reset geometry, tracking frames, contact offsets or robot assets.
    """
    from isaaclab.sim import RigidBodyMaterialCfg

    if any(profile.get("fan_response", {}).get(k, 0) > 0 for k in ("delay_s", "rise_s", "fall_s")):
        raise ValueError("Profile includes noninstantaneous fan dynamics; RL action currently supports nominal instantaneous profiles only")
    p = profile["physical"]
    if tuple(env_cfg.scene.box.spawn.size) != (p["cube_size_m"],) * 3:
        raise ValueError("Calibration cube size differs from RL geometry; align task geometry first")
    env_cfg.scene.box.spawn.mass_props.mass = p["cube_mass_kg"]
    def material(prefix):
        return RigidBodyMaterialCfg(
            static_friction=p[f"{prefix}_static_friction"],
            dynamic_friction=p[f"{prefix}_dynamic_friction"],
            restitution=0.0, friction_combine_mode="average", restitution_combine_mode="average",
        )
    env_cfg.scene.box.spawn.physics_material = material("cube")
    # UsdFileCfg does not consume a physics_material field. Bind explicitly to
    # the table's colliders after importing its USD, before physics starts.
    env_cfg.scene.table.spawn.calibration_support_material = material("ground")
    env_cfg.scene.table.spawn.func = spawn_calibrated_table
    action = env_cfg.actions.airflow
    action.jet_cfg = copy.deepcopy(action.jet_cfg)
    action.drag_cfg = copy.deepcopy(action.drag_cfg)
    action.drag_cfg.C_D = p["drag_coefficient"]
    for key in ("U0", "spread_k", "xc", "decay_k"):
        setattr(action.jet_cfg, key, p[f"jet_{key}"])
    action.jet_cfg.nozzle_radius = p["nozzle_radius"]
    action.jet_cfg.axial_model = "exponential"
    action.airflow_u0_range = None
    action.airflow_spread_k_range = None
    env_cfg.events.randomize_box_mass = None
    env_cfg.events.randomize_box_material = None


def spawn_calibrated_table(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    paths = sim_utils.find_matching_prim_paths(prim_path)
    if not paths:
        raise ValueError(f"No calibrated support colliders found: {prim_path}")
    for path in paths:
        material_path = path + "/CalibrationMaterial"
        cfg.calibration_support_material.func(material_path, cfg.calibration_support_material)
        sim_utils.bind_physics_material(path, material_path)
    return prim
