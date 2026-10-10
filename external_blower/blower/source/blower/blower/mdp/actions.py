"""Airflow action term for Isaac Lab manager-based environments."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from blower.configs.airflow_cfg import DragCfg, JetCfg, SamplingCfg, VizCfg
from blower.models.airflow_physics import AirflowModel, AirflowModelCfg, quat_to_axis
from blower.models.sampling import (
    build_mesh_surface_samples_for_prim,
    cylinder_surface_area,
    transform_samples_to_world,
)
from blower.objects.object_cfg import BallCfg, BottleCfg, BoxCfg, CigaretteCfg, CuboidCfg, build_target_object_samples


def _is_missing(value) -> bool:
    return value is MISSING or value is None or (isinstance(value, str) and value == '???')


def _cfg_get(cfg, name: str, default=MISSING):
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _coerce_sampling_cfg(sampling_cfg) -> SamplingCfg:
    if _is_missing(sampling_cfg):
        return SamplingCfg()
    if isinstance(sampling_cfg, Mapping):
        return SamplingCfg(**{key: value for key, value in sampling_cfg.items() if not _is_missing(value)})
    return sampling_cfg


def _infer_target_type(asset_name) -> str | None:
    name = str(asset_name).lower()
    for target_type in ('cuboid', 'box', 'ball', 'bottle', 'cigarette'):
        if target_type in name:
            return target_type
    return None


def _coerce_object_cfg(object_cfg, target_type: str):
    if _is_missing(object_cfg):
        return None
    if hasattr(object_cfg, 'build_body_samples'):
        return object_cfg
    if isinstance(object_cfg, Mapping):
        cfg_cls = {
            'ball': BallCfg,
            'box': BoxCfg,
            'cuboid': CuboidCfg,
            'bottle': BottleCfg,
            'cigarette': CigaretteCfg,
        }.get(str(target_type).lower())
        if cfg_cls is None:
            raise ValueError(f'Unsupported target_type for object config: {target_type}')
        fields = getattr(cfg_cls, '__dataclass_fields__', {})
        kwargs = {
            key: value
            for key, value in object_cfg.items()
            if key in fields and not _is_missing(value)
        }
        return cfg_cls(**kwargs)
    raise TypeError(f'Unsupported target_object_cfg type: {type(object_cfg).__name__}')


def _first_env_prim_path(env: ManagerBasedEnv) -> str | None:
    env_prim_paths = getattr(getattr(env, 'scene', None), 'env_prim_paths', None)
    try:
        if env_prim_paths:
            return str(env_prim_paths[0])
    except Exception:
        pass
    return '/World/envs/env_0'


def _get_usd_stage():
    try:
        import omni.usd

        return omni.usd.get_context().get_stage()
    except Exception:
        return None


def _asset_cfg_prim_path(asset) -> str | None:
    for owner in (asset, getattr(asset, 'cfg', None)):
        if owner is None:
            continue

        prim_path = getattr(owner, 'prim_path', None)
        if isinstance(prim_path, str) and prim_path:
            return prim_path

        prim_paths = getattr(owner, 'prim_paths', None)
        try:
            if prim_paths:
                first_prim_path = prim_paths[0]
                if isinstance(first_prim_path, str) and first_prim_path:
                    return first_prim_path
        except Exception:
            pass
    return None


def _is_concrete_prim_path(path: str) -> bool:
    return path.startswith('/') and not any(token in path for token in ('{', '}', '*', '[', ']'))


def _target_prim_suffixes(asset_name: str, cfg_prim_path: str | None) -> list[str]:
    suffixes = []
    if cfg_prim_path:
        if '{ENV_REGEX_NS}' in cfg_prim_path:
            suffix = cfg_prim_path.split('{ENV_REGEX_NS}', 1)[1]
            if suffix.startswith('/'):
                suffixes.append(suffix)
        tail = cfg_prim_path.rstrip('/').split('/')[-1]
        if tail and not any(token in tail for token in ('{', '}', '*', '[', ']')):
            suffixes.append(f'/{tail}')

    asset_tail = str(asset_name).strip('/')
    if asset_tail:
        suffixes.append(f'/{asset_tail}')
        suffixes.append(f'/{asset_tail[:1].upper()}{asset_tail[1:]}')

    unique = []
    for suffix in suffixes:
        if suffix not in unique:
            unique.append(suffix)
    return unique


def _resolve_scene_target_prim_path(env: ManagerBasedEnv, asset_name: str, asset) -> str | None:
    stage = _get_usd_stage()
    if stage is None:
        return None

    cfg_prim_path = _asset_cfg_prim_path(asset)
    env_root = _first_env_prim_path(env)
    candidates = []
    if cfg_prim_path:
        if env_root is not None and '{ENV_REGEX_NS}' in cfg_prim_path:
            candidates.append(cfg_prim_path.replace('{ENV_REGEX_NS}', env_root))
        if _is_concrete_prim_path(cfg_prim_path):
            candidates.append(cfg_prim_path)

    for candidate in candidates:
        try:
            if stage.GetPrimAtPath(candidate).IsValid():
                return candidate
        except Exception:
            pass

    suffixes = _target_prim_suffixes(str(asset_name), cfg_prim_path)
    if not suffixes:
        return None

    preferred_matches = []
    fallback_matches = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not any(prim_path.endswith(suffix) for suffix in suffixes):
            continue
        if env_root is not None and prim_path.startswith(f'{env_root}/'):
            preferred_matches.append(prim_path)
        else:
            fallback_matches.append(prim_path)

    matches = preferred_matches or fallback_matches
    if not matches:
        return None
    return sorted(matches)[0]


def _bottle_representative_area(object_cfg) -> float:
    bottle_cfg = object_cfg if object_cfg is not None else BottleCfg()
    radius = float(getattr(bottle_cfg, 'cylinder_radius', 0.034))
    height = float(getattr(bottle_cfg, 'cylinder_height', 0.20))
    return cylinder_surface_area(radius, height)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion product in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in (w, x, y, z) convention."""
    q_xyz = q[..., 1:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q[..., 0:1] * t + torch.cross(q_xyz, t, dim=-1)


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)


class AirflowActionTerm(ActionTerm):
    cfg: 'AirflowActionTermCfg'

    def __init__(self, cfg: 'AirflowActionTermCfg', env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._source_type = cfg.source_type.lower()
        self._source = None
        self._source_body_id = None
        self._source_asset_name = None
        self._source_body_name = None
        self._resolve_source()

        self._target_states = self._build_target_states(cfg, env)
        self._target_asset_name = self._target_states[0]['asset_name']
        self._target = self._target_states[0]['asset']
        self._target_object_cfg = self._target_states[0]['object_cfg']
        self._sample_batch = self._target_states[0]['sample_batch']

        self._airflow_model = AirflowModel(
            AirflowModelCfg(
                jet=cfg.jet_cfg,
                drag=cfg.drag_cfg,
                mode=cfg.mode,
                min_axial_distance=cfg.min_axial_distance,
                velocity_cutoff=cfg.velocity_cutoff,
                nozzle_local_axis=cfg.nozzle_local_axis,
            ),
            device=self.device,
        )

        self._nozzle_offset_pos = torch.tensor(cfg.nozzle_offset_pos, dtype=torch.float32, device=self.device)
        self._nozzle_offset_quat = _normalize_quat(
            torch.tensor(cfg.nozzle_offset_quat, dtype=torch.float32, device=self.device)
        )
        self._world_nozzle_pos = torch.tensor(cfg.world_nozzle_pos, dtype=torch.float32, device=self.device)
        self._world_nozzle_quat = _normalize_quat(
            torch.tensor(cfg.world_nozzle_quat, dtype=torch.float32, device=self.device)
        )

        self._action_dim = 1
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        for target_state in self._target_states:
            target_state['forces'] = torch.zeros(self.num_envs, 1, 3, device=self.device)
            target_state['torques'] = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._forces = self._target_states[0]['forces']
        self._torques = self._target_states[0]['torques']
        self._source_reaction_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._source_reaction_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        if self.cfg.apply_reaction_force and self._source is None:
            raise ValueError(
                "Airflow reaction force requires a physical source. "
                "Use source_type='articulation_body' or source_type='rigid_object'."
            )

        self._viz_enabled = cfg.viz_cfg is not None and (
            cfg.viz_cfg.show_jet_cone
            or cfg.viz_cfg.show_sample_markers
            or cfg.viz_cfg.show_laser_ray
            or cfg.viz_cfg.show_reaction_force_arrow
        )
        if cfg.viz_cfg is not None and not cfg.viz_cfg.show_jet_cone:
            self._remove_debug_prim('/World/Debug/JetCone')
        if self._viz_enabled:
            from blower.viz import (
                JetConeVisualizer,
                LaserRayVisualizer,
                ReactionForceArrowVisualizer,
                SampleMarkerVisualizer,
            )

            _prim_root = getattr(cfg.viz_cfg, 'debug_prim_root', '/World/Debug/Airflow')
            self._remove_debug_prim(_prim_root)
            self._remove_debug_prim('/World/Debug/JetCone')
            self._remove_debug_prim('/World/Debug/LaserRay')
            self._remove_debug_prim('/World/Debug/SampleMarkers')
            self._viz_env_ids = self._resolve_viz_env_ids()
            cone_renderer = JetConeVisualizer
            if cfg.viz_cfg.show_jet_cone:
                if cfg.viz_cfg.jet_cone_render_mode == 'mesh':
                    from blower.viz.jet_cone_mesh import MeshJetConeVisualizer
                    cone_renderer = MeshJetConeVisualizer
                elif cfg.viz_cfg.jet_cone_render_mode != 'wireframe':
                    raise ValueError(f"Unknown jet_cone_render_mode: {cfg.viz_cfg.jet_cone_render_mode!r}")
            self._jet_viz = (
                cone_renderer(cfg.viz_cfg, cfg.jet_cfg, prim_root=f'{_prim_root}/JetCones')
                if cfg.viz_cfg.show_jet_cone else None
            )
            self._reaction_arrow_viz = (
                ReactionForceArrowVisualizer(cfg.viz_cfg, prim_root=f'{_prim_root}/ReactionForceArrows')
                if cfg.viz_cfg.show_reaction_force_arrow else None
            )
            self._sample_viz = []
            self._laser_viz = []
            for env_id in self._viz_env_ids:
                env_root = f'{_prim_root}/env_{env_id:04d}'
                self._sample_viz.append(
                    SampleMarkerVisualizer(cfg.viz_cfg, prim_root=f'{env_root}/SampleMarkers')
                    if cfg.viz_cfg.show_sample_markers else None
                )
                self._laser_viz.append(
                    LaserRayVisualizer(cfg.viz_cfg, prim_root=f'{env_root}/LaserRay')
                    if cfg.viz_cfg.show_laser_ray else None
                )
        else:
            self._viz_env_ids = []
            self._jet_viz = None
            self._reaction_arrow_viz = None
            self._sample_viz = []
            self._laser_viz = []

    def _build_target_states(self, cfg: 'AirflowActionTermCfg', env: ManagerBasedEnv) -> list[dict]:
        target_cfgs = self._resolve_target_cfgs(cfg)
        target_states = []
        for target_cfg in target_cfgs:
            asset_name = _cfg_get(target_cfg, 'asset_name')
            target_type = _cfg_get(target_cfg, 'target_type')
            object_cfg = _cfg_get(target_cfg, 'target_object_cfg')
            if _is_missing(target_type) and not _is_missing(object_cfg):
                target_type = _cfg_get(object_cfg, 'target_type', MISSING)
            if _is_missing(target_type):
                target_type = _infer_target_type(asset_name)
            if _is_missing(target_type):
                raise ValueError(
                    f"Airflow target {asset_name!r} requires 'target_type' or an inferable asset name."
                )
            asset = env.scene[asset_name]

            sampling_cfg = _cfg_get(target_cfg, 'sampling_cfg')
            if _is_missing(sampling_cfg):
                sampling_cfg = cfg.sampling_cfg
            sampling_cfg = _coerce_sampling_cfg(sampling_cfg)
            object_cfg = _coerce_object_cfg(object_cfg, target_type)
            sample_batch = self._build_target_sample_batch(
                env,
                str(asset_name),
                str(target_type),
                asset,
                object_cfg,
                sampling_cfg,
            )

            target_states.append(
                {
                    'asset_name': asset_name,
                    'asset': asset,
                    'target_type': target_type,
                    'object_cfg': object_cfg,
                    'sample_batch': {
                        key: tensor.to(self.device)
                        for key, tensor in sample_batch.items()
                    },
                }
            )

        if not target_states:
            raise ValueError('AirflowActionTermCfg requires at least one target.')
        return target_states

    def _build_target_sample_batch(
        self,
        env: ManagerBasedEnv,
        asset_name: str,
        target_type: str,
        asset,
        object_cfg,
        sampling_cfg: SamplingCfg,
    ) -> dict:
        scene_sample_batch = self._try_build_scene_aligned_sample_batch(
            env,
            asset_name,
            target_type,
            asset,
            object_cfg,
            sampling_cfg,
        )
        if scene_sample_batch is not None:
            return scene_sample_batch

        if object_cfg is None:
            return build_target_object_samples(target_type, sampling_cfg)

        if object_cfg.target_type != target_type:
            raise ValueError(
                f'Object config type {object_cfg.target_type} does not match target_type {target_type}'
            )
        return object_cfg.build_body_samples(sampling_cfg)

    def _try_build_scene_aligned_sample_batch(
        self,
        env: ManagerBasedEnv,
        asset_name: str,
        target_type: str,
        asset,
        object_cfg,
        sampling_cfg: SamplingCfg,
    ) -> dict | None:
        if target_type.lower() != 'bottle':
            return None

        bottle_cfg = object_cfg if object_cfg is not None else BottleCfg()
        if not getattr(bottle_cfg, 'use_asset', True) or not getattr(bottle_cfg, 'use_mesh_samples', True):
            return None

        target_prim_path = _resolve_scene_target_prim_path(env, asset_name, asset)
        if target_prim_path is None:
            return None

        try:
            return build_mesh_surface_samples_for_prim(
                target_prim_path,
                sampling_cfg=sampling_cfg,
                representative_area=_bottle_representative_area(bottle_cfg),
            )
        except Exception as exc:
            warnings.warn(
                f'Bottle stage mesh sampling failed for {target_prim_path} ({exc}); '
                'falling back to object config sampling.',
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _resolve_target_cfgs(self, cfg: 'AirflowActionTermCfg') -> tuple['AirflowTargetCfg', ...]:
        if cfg.targets:
            return tuple(cfg.targets)

        target_asset_name = cfg.target_asset_name if not _is_missing(cfg.target_asset_name) else cfg.asset_name
        return (
            AirflowTargetCfg(
                asset_name=target_asset_name,
                target_type=cfg.target_type,
                target_object_cfg=cfg.target_object_cfg,
                sampling_cfg=cfg.sampling_cfg,
            ),
        )

    def _remove_debug_prim(self, prim_path: str) -> None:
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)
        except Exception:
            pass

    def _resolve_viz_env_ids(self) -> list[int]:
        max_visualized_envs = max(0, int(self.cfg.viz_cfg.max_visualized_envs))
        if max_visualized_envs == 0:
            return []

        requested_env_ids = list(getattr(self.cfg.viz_cfg, 'visualized_env_ids', ()) or ())
        if requested_env_ids:
            env_ids = [int(env_id) for env_id in requested_env_ids if 0 <= int(env_id) < self.num_envs]
            return env_ids[:max_visualized_envs]
        return list(range(min(self.num_envs, max_visualized_envs)))

    def _resolve_source(self):
        if self._source_type == 'articulation_body':
            if _is_missing(self.cfg.source_asset_name):
                raise ValueError("AirflowActionTermCfg with source_type='articulation_body' requires 'source_asset_name'.")
            if _is_missing(self.cfg.source_body_name):
                raise ValueError("AirflowActionTermCfg with source_type='articulation_body' requires 'source_body_name'.")
            self._source_asset_name = self.cfg.source_asset_name
            self._source = self._env.scene[self.cfg.source_asset_name]
            body_ids, body_names = self._source.find_bodies(self.cfg.source_body_name)
            if len(body_ids) != 1:
                raise ValueError(
                    f"Expected one source body for pattern {self.cfg.source_body_name!r}. "
                    f"Found {len(body_ids)}: {body_names}."
                )
            self._source_body_id = body_ids[0]
            self._source_body_name = body_names[0]
            return

        if self._source_type == 'rigid_object':
            source_asset_name = self.cfg.source_asset_name
            if _is_missing(source_asset_name) and not _is_missing(self.cfg.target_asset_name):
                source_asset_name = self.cfg.asset_name
            if _is_missing(source_asset_name):
                raise ValueError(
                    "AirflowActionTermCfg with source_type='rigid_object' requires 'source_asset_name'. "
                    "Legacy configs may pass asset_name=<blower> and target_asset_name=<target>."
                )
            self._source_asset_name = source_asset_name
            self._source = self._env.scene[source_asset_name]
            return

        if self._source_type == 'world_frame':
            return

        raise ValueError(
            f"Unsupported source_type: {self.cfg.source_type!r}. "
            "Expected 'articulation_body', 'rigid_object', or 'world_frame'."
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self.device)
        self._processed_actions[:, 0] = torch.clamp(self._raw_actions[:, 0], 0.0, 1.0)

    def _source_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._source_type == 'articulation_body':
            return (
                self._source.data.body_pos_w[:, self._source_body_id],
                self._source.data.body_quat_w[:, self._source_body_id],
            )

        if self._source_type == 'rigid_object':
            return self._source.data.root_pos_w, self._source.data.root_quat_w

        pos = self._world_nozzle_pos.unsqueeze(0).expand(self.num_envs, 3)
        quat = self._world_nozzle_quat.unsqueeze(0).expand(self.num_envs, 4)
        return pos, quat

    def _nozzle_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        source_pos, source_quat = self._source_pose_w()
        source_quat = _normalize_quat(source_quat)
        offset_pos = self._nozzle_offset_pos.unsqueeze(0).expand(self.num_envs, 3)
        offset_quat = self._nozzle_offset_quat.unsqueeze(0).expand(self.num_envs, 4)
        nozzle_pos = source_pos + _quat_apply(source_quat, offset_pos)
        nozzle_quat = _normalize_quat(_quat_mul(source_quat, offset_quat))
        return nozzle_pos, nozzle_quat

    def _apply_target_wrench(self, target, forces: torch.Tensor, torques: torch.Tensor) -> None:
        wrench_composer = getattr(target, 'instantaneous_wrench_composer', None)
        if wrench_composer is not None:
            wrench_composer.set_forces_and_torques(
                forces=forces,
                torques=torques,
                body_ids=[0],
                is_global=True,
            )
            return

        target.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=[0],
            is_global=True,
        )

    def _source_body_ids_for_wrench(self) -> list[int]:
        if self._source_type == 'articulation_body':
            return [self._source_body_id]
        if self._source_type == 'rigid_object':
            return [0]
        return []

    def _source_wrench_origin_w(self) -> torch.Tensor:
        if self._source_type == 'articulation_body':
            return self._source.data.body_pos_w[:, self._source_body_id]
        if self._source_type == 'rigid_object':
            return self._source.data.root_pos_w
        return self._world_nozzle_pos.unsqueeze(0).expand(self.num_envs, 3)

    def _reaction_force_enabled(self) -> bool:
        return (
            bool(self.cfg.apply_reaction_force)
            and float(self.cfg.reaction_force_max) > 0.0
            and self._source is not None
        )

    def _apply_source_reaction_wrench(
        self,
        nozzle_pos: torch.Tensor,
        nozzle_dir: torch.Tensor,
        flow_intensity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if not self._reaction_force_enabled():
            return

        if env_ids is None:
            active = slice(None)
            active_nozzle_pos = nozzle_pos
            active_nozzle_dir = nozzle_dir
            active_intensity = flow_intensity
            active_origin = self._source_wrench_origin_w()
        elif env_ids.numel() == 0:
            return
        else:
            active = env_ids
            active_nozzle_pos = nozzle_pos[env_ids]
            active_nozzle_dir = nozzle_dir[env_ids]
            active_intensity = flow_intensity[env_ids]
            active_origin = self._source_wrench_origin_w()[env_ids]

        if active_intensity.dim() == 2:
            active_intensity = active_intensity[:, 0]

        force_mag = torch.clamp(active_intensity, 0.0, 1.0) * max(
            float(self.cfg.reaction_force_max),
            0.0,
        )
        forces = -active_nozzle_dir * force_mag.unsqueeze(-1)
        torques = torch.cross(active_nozzle_pos - active_origin, forces, dim=-1)

        self._source_reaction_forces[active, 0, :] = forces
        self._source_reaction_torques[active, 0, :] = torques
        self._write_source_reaction_wrench()

    def _zero_source_reaction_wrench(self, env_ids: torch.Tensor | None = None) -> None:
        if not self._reaction_force_enabled():
            return

        if env_ids is None:
            self._source_reaction_forces[:] = 0.0
            self._source_reaction_torques[:] = 0.0
        elif env_ids.numel() > 0:
            self._source_reaction_forces[env_ids] = 0.0
            self._source_reaction_torques[env_ids] = 0.0
            self._clear_reaction_force_visualization(env_ids)
        else:
            return

        self._write_source_reaction_wrench()
        if env_ids is None:
            self._clear_reaction_force_visualization()

    def _write_source_reaction_wrench(self) -> None:
        body_ids = self._source_body_ids_for_wrench()
        if not body_ids:
            return

        wrench_composer = getattr(self._source, 'instantaneous_wrench_composer', None)
        if wrench_composer is not None:
            wrench_composer.set_forces_and_torques(
                forces=self._source_reaction_forces,
                torques=self._source_reaction_torques,
                body_ids=body_ids,
                is_global=True,
            )
            return

        self._source.set_external_force_and_torque(
            forces=self._source_reaction_forces,
            torques=self._source_reaction_torques,
            body_ids=body_ids,
            is_global=True,
        )

    def _update_reaction_force_visualization(self, nozzle_pos: torch.Tensor) -> None:
        if self._reaction_arrow_viz is None:
            return

        env_ids = [env_id for env_id in self._viz_env_ids if 0 <= env_id < self.num_envs]
        if not env_ids:
            self._reaction_arrow_viz.clear()
            return

        origins = [
            tuple(float(v) for v in nozzle_pos[env_id].detach().cpu().tolist())
            for env_id in env_ids
        ]
        reaction_forces = [
            tuple(float(v) for v in self._source_reaction_forces[env_id, 0].detach().cpu().tolist())
            for env_id in env_ids
        ]
        self._reaction_arrow_viz.update_many(env_ids, origins, reaction_forces)

    def _clear_reaction_force_visualization(self, env_ids: torch.Tensor | None = None) -> None:
        if self._reaction_arrow_viz is None:
            return
        if env_ids is None:
            self._reaction_arrow_viz.clear()
            return
        self._reaction_arrow_viz.clear([int(env_id) for env_id in env_ids.detach().cpu().tolist()])

    def apply_actions(self):
        nozzle_pos, nozzle_quat = self._nozzle_pose_w()
        flow_intensity = self._processed_actions[:, 0:1]
        visualized_sample_batches = []

        for target_state in self._target_states:
            target = target_state['asset']
            target_pos = target.data.root_pos_w
            target_quat = target.data.root_quat_w

            world_sample_batch = transform_samples_to_world(
                target_pos,
                target_quat,
                target_state['sample_batch'],
            )

            result = self._airflow_model.compute_force_result(
                blower_pos=nozzle_pos,
                blower_quat=nozzle_quat,
                target_pos=target_pos,
                sample_positions_world=world_sample_batch['positions_world'],
                sample_normals_world=world_sample_batch['normals_world'],
                sample_areas=world_sample_batch['areas'],
                sample_is_omni=world_sample_batch['is_omni'],
                flow_intensity=flow_intensity,
            )

            forces = target_state['forces']
            torques = target_state['torques']
            forces[:, 0, :] = result['force_vec'][:, 0, :]
            torques[:, 0, :] = result['torque_vec'][:, 0, :]
            self._apply_target_wrench(target, forces, torques)

            if self._viz_enabled and self.cfg.viz_cfg.show_sample_markers:
                visualized_sample_batches.append(world_sample_batch['positions_world'])

        nozzle_dir = None
        if self._reaction_force_enabled() or self._viz_enabled:
            nozzle_dir = quat_to_axis(nozzle_quat, self.cfg.nozzle_local_axis)

        if self._reaction_force_enabled():
            self._apply_source_reaction_wrench(nozzle_pos, nozzle_dir, flow_intensity)
        if self._reaction_arrow_viz is not None:
            self._update_reaction_force_visualization(nozzle_pos)

        if self._viz_enabled:
            sample_positions = (
                torch.cat(visualized_sample_batches, dim=1)
                if visualized_sample_batches else None
            )
            self._update_visualization(
                nozzle_pos,
                nozzle_dir,
                sample_positions,
                flow_intensity,
            )

    def _update_visualization(
        self,
        nozzle_pos: torch.Tensor,
        nozzle_dir: torch.Tensor,
        sample_positions: torch.Tensor | None,
        flow_intensity: torch.Tensor,
    ) -> None:
        if self._jet_viz is not None:
            nozzle_positions = [
                tuple(float(v) for v in nozzle_pos[env_id].tolist())
                for env_id in self._viz_env_ids
            ]
            nozzle_dirs = [
                tuple(float(v) for v in nozzle_dir[env_id].tolist())
                for env_id in self._viz_env_ids
            ]
            intensities = [float(flow_intensity[env_id, 0].item()) for env_id in self._viz_env_ids]
            self._jet_viz.update_many(nozzle_positions, nozzle_dirs, intensities)

        for index, env_id in enumerate(self._viz_env_ids):
            env_nozzle_pos = nozzle_pos[env_id].tolist()
            env_nozzle_dir = nozzle_dir[env_id].tolist()

            if self._sample_viz[index] is not None and sample_positions is not None:
                env_sample_positions = sample_positions[env_id].cpu().tolist()
                self._sample_viz[index].update([tuple(p) for p in env_sample_positions])
            if self._laser_viz[index] is not None:
                self._laser_viz[index].update(env_nozzle_pos, env_nozzle_dir)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            for target_state in self._target_states:
                target_state['forces'][:] = 0.0
                target_state['torques'][:] = 0.0
            self._zero_source_reaction_wrench()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            for target_state in self._target_states:
                target_state['forces'][env_ids] = 0.0
                target_state['torques'][env_ids] = 0.0
            self._zero_source_reaction_wrench(env_ids)


@configclass
class AirflowTargetCfg:
    asset_name: str = MISSING
    target_type: str = MISSING
    target_object_cfg: object = MISSING
    sampling_cfg: SamplingCfg | None = None


@configclass
class AirflowActionTermCfg(ActionTermCfg):
    class_type: type = AirflowActionTerm

    # Isaac Lab's ActionTerm base requires asset_name. In the new virtual-nozzle
    # interface this should normally be the target asset that receives airflow.
    asset_name: str = MISSING
    target_asset_name: str | None = None

    # Virtual nozzle pose source.
    source_type: str = 'articulation_body'
    source_asset_name: str | None = None
    source_body_name: str | None = None
    nozzle_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    nozzle_offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    nozzle_local_axis: str = 'x'
    world_nozzle_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_nozzle_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    target_type: str = MISSING
    target_object_cfg: object = MISSING
    targets: tuple[AirflowTargetCfg, ...] = ()
    mode: str = 'surface'
    min_axial_distance: float = 0.0
    velocity_cutoff: float = 0.5
    jet_cfg: JetCfg = JetCfg()
    drag_cfg: DragCfg = DragCfg()
    viz_cfg: VizCfg | None = None
    sampling_cfg: SamplingCfg = SamplingCfg()

    # Optional recoil force applied to the physical source body. The force is
    # linear in the processed airflow action: 0 -> 0 N, 1 -> reaction_force_max N.
    apply_reaction_force: bool = False
    reaction_force_max: float = 0.0
