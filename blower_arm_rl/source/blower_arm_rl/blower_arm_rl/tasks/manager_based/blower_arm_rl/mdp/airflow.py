"""Airflow helpers for Franka blower tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from blower.models.airflow_physics import quat_to_axis
from blower.models.sampling import transform_samples_to_world
from blower.mdp import AirflowActionTerm, AirflowActionTermCfg


class DelayedAirflowActionTerm(AirflowActionTerm):
    """Apply airflow automatically after a per-episode delay."""

    cfg: "DelayedAirflowActionTermCfg"

    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = 0.0

    def apply_actions(self):
        elapsed_s = self._env.episode_length_buf.to(self.device) * self._env.step_dt
        enabled = elapsed_s >= self.cfg.delay_s
        self._processed_actions[:, 0] = torch.where(
            enabled,
            torch.full_like(elapsed_s, self.cfg.intensity),
            torch.zeros_like(elapsed_s),
        )
        super().apply_actions()

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)


@configclass
class DelayedAirflowActionTermCfg(AirflowActionTermCfg):
    class_type: type = DelayedAirflowActionTerm

    delay_s: float = 1.0
    intensity: float = 1.0


class SmoothedAirflowActionTerm(AirflowActionTerm):
    """Policy-controlled airflow with action mapping, smoothing, and reset settling.

    Isaac Lab calls action terms immediately after an environment reset. The box can still be settling during
    those first few frames, so this term masks freshly-reset environments and applies zero wrench without
    evaluating the airflow model for them.
    """

    cfg: "SmoothedAirflowActionTermCfg"

    def __init__(self, cfg: "SmoothedAirflowActionTermCfg", env):
        super().__init__(cfg, env)
        self._target_actions = torch.zeros_like(self._processed_actions)
        self._previous_processed_actions = torch.zeros_like(self._processed_actions)
        self._mount_pose_viz = self._make_mount_pose_visualizer() if cfg.show_mount_pose_marker else None
        self._base_jet_u0 = float(getattr(self.cfg.jet_cfg, "U0", 0.0))
        self._base_jet_spread_k = float(getattr(self.cfg.jet_cfg, "spread_k", 0.0))
        self._airflow_bucket_u0, self._airflow_bucket_spread_k = self._make_airflow_randomization_buckets()
        self._airflow_bucket_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._airflow_u0 = torch.full((self.num_envs,), self._base_jet_u0, dtype=torch.float32, device=self.device)
        self._airflow_spread_k = torch.full(
            (self.num_envs,), self._base_jet_spread_k, dtype=torch.float32, device=self.device
        )
        self._reaction_force_max = torch.full(
            (self.num_envs,), max(float(self.cfg.reaction_force_max), 0.0), dtype=torch.float32, device=self.device
        )
        self._sample_airflow_randomization(None)

    @property
    def previous_processed_actions(self) -> torch.Tensor:
        return self._previous_processed_actions

    def target_force_w(self, asset) -> torch.Tensor:
        """Last applied airflow force in world coordinates, excluding contact forces."""
        for state in self._target_states:
            if state["asset"] is asset:
                return state["forces"].sum(dim=1)
        raise ValueError("Requested asset is not an airflow target")

    def target_torque_w(self, asset) -> torch.Tensor:
        """Last applied jet torque about the target COM, excluding contacts."""
        for state in self._target_states:
            if state["asset"] is asset:
                return state["torques"].sum(dim=1)
        raise ValueError("Requested asset is not an airflow target")

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self.device)
        self._previous_processed_actions[:] = self._processed_actions

        if self.cfg.action_mapping == "tanh_01":
            self._target_actions[:, 0] = 0.5 * (torch.tanh(self._raw_actions[:, 0]) + 1.0)
        elif self.cfg.action_mapping == "clamp_01":
            self._target_actions[:, 0] = torch.clamp(self._raw_actions[:, 0], 0.0, 1.0)
        elif self.cfg.action_mapping == "clip_signed_01":
            self._target_actions[:, 0] = 0.5 * (torch.clamp(self._raw_actions[:, 0], -1.0, 1.0) + 1.0)
        else:
            raise ValueError(
                f"Unsupported airflow action_mapping={self.cfg.action_mapping!r}. "
                "Expected 'tanh_01', 'clamp_01', or 'clip_signed_01'."
            )

        if self.cfg.smoothing_tau_s > 0.0:
            alpha = self._env.step_dt / (self.cfg.smoothing_tau_s + self._env.step_dt)
            self._processed_actions[:] = (
                alpha * self._target_actions + (1.0 - alpha) * self._previous_processed_actions
            )
        else:
            self._processed_actions[:] = self._target_actions

        ready = self._ready_mask()
        self._processed_actions[:, 0] = torch.where(
            ready,
            self._processed_actions[:, 0],
            torch.zeros_like(self._processed_actions[:, 0]),
        )

        # self._processed_actions[:, 0] = torch.where(
        #     ready,
        #     torch.ones_like(self._processed_actions[:, 0]),
        #     torch.zeros_like(self._processed_actions[:, 0]),
        # )

    def apply_actions(self):
        if self._mount_pose_viz is not None:
            nozzle_pos, nozzle_quat = self._nozzle_pose_w()
            self._update_mount_pose_visualization(nozzle_pos, nozzle_quat)

        ready = self._ready_mask()
        if not torch.any(ready):
            self._processed_actions[:] = 0.0
            self._zero_target_wrenches()
            return

        not_ready_ids = (~ready).nonzero(as_tuple=False).squeeze(-1)
        if not_ready_ids.numel() > 0:
            self._processed_actions[not_ready_ids] = 0.0
            self._zero_target_wrenches(env_ids=not_ready_ids)
        self._apply_ready_airflow(ready.nonzero(as_tuple=False).squeeze(-1))

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        self._sample_airflow_randomization(env_ids)
        if env_ids is None:
            self._target_actions[:] = 0.0
            self._previous_processed_actions[:] = 0.0
        else:
            self._target_actions[env_ids] = 0.0
            self._previous_processed_actions[env_ids] = 0.0

    def _ready_mask(self) -> torch.Tensor:
        elapsed_s = self._env.episode_length_buf.to(self.device) * self._env.step_dt
        return elapsed_s >= self.cfg.reset_settle_s

    def _zero_target_wrenches(self, env_ids: torch.Tensor | None = None):
        for target_state in self._target_states:
            forces = target_state["forces"]
            torques = target_state["torques"]
            if env_ids is None:
                forces[:] = 0.0
                torques[:] = 0.0
            elif env_ids.numel() > 0:
                forces[env_ids] = 0.0
                torques[env_ids] = 0.0
            self._apply_target_wrench(target_state["asset"], forces, torques)
        self._zero_source_reaction_wrench(env_ids)

    def _make_airflow_randomization_buckets(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_buckets = max(int(self.cfg.airflow_randomization_buckets), 1)
        u0_range = self.cfg.airflow_u0_range
        spread_k_range = self.cfg.airflow_spread_k_range

        if u0_range is None:
            u0_low = u0_high = self._base_jet_u0
        else:
            u0_low, u0_high = (float(u0_range[0]), float(u0_range[1]))
        if spread_k_range is None:
            spread_low = spread_high = self._base_jet_spread_k
        else:
            spread_low, spread_high = (float(spread_k_range[0]), float(spread_k_range[1]))

        u0 = torch.empty(num_buckets, dtype=torch.float32, device=self.device).uniform_(u0_low, u0_high)
        spread_k = torch.empty(num_buckets, dtype=torch.float32, device=self.device).uniform_(spread_low, spread_high)
        return u0.clamp_min(1e-6), spread_k.clamp_min(0.0)

    def _sample_airflow_randomization(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        bucket_ids = torch.randint(
            0,
            self._airflow_bucket_u0.numel(),
            (env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )
        self._airflow_bucket_ids[env_ids] = bucket_ids
        self._airflow_u0[env_ids] = self._airflow_bucket_u0[bucket_ids]
        self._airflow_spread_k[env_ids] = self._airflow_bucket_spread_k[bucket_ids]

        if self._base_jet_u0 > 1e-6:
            u0_ratio_sq = (self._airflow_u0[env_ids] / self._base_jet_u0) ** 2
            self._reaction_force_max[env_ids] = max(float(self.cfg.reaction_force_max), 0.0) * u0_ratio_sq
        else:
            self._reaction_force_max[env_ids] = max(float(self.cfg.reaction_force_max), 0.0)

    def _apply_ready_airflow(self, ready_ids: torch.Tensor):
        nozzle_pos, nozzle_quat = self._nozzle_pose_w()
        flow_intensity = self._processed_actions[:, 0:1]
        visualized_sample_batches = []

        for target_state in self._target_states:
            target = target_state["asset"]
            target_pos = target.data.root_pos_w
            target_quat = target.data.root_quat_w

            world_sample_batch = transform_samples_to_world(
                target_pos[ready_ids],
                target_quat[ready_ids],
                target_state["sample_batch"],
            )

            forces = target_state["forces"]
            torques = target_state["torques"]
            old_u0 = float(self._airflow_model.cfg.jet.U0)
            old_spread_k = float(self._airflow_model.cfg.jet.spread_k)
            try:
                unique_bucket_ids = torch.unique(self._airflow_bucket_ids[ready_ids])
                if unique_bucket_ids.numel() == 1:
                    bucket_id = unique_bucket_ids[0]
                    self._airflow_model.cfg.jet.U0 = float(self._airflow_bucket_u0[bucket_id])
                    self._airflow_model.cfg.jet.spread_k = float(self._airflow_bucket_spread_k[bucket_id])
                    result = self._airflow_model.compute_force_result(
                        blower_pos=nozzle_pos[ready_ids],
                        blower_quat=nozzle_quat[ready_ids],
                        target_pos=target_pos[ready_ids],
                        sample_positions_world=world_sample_batch["positions_world"],
                        sample_normals_world=world_sample_batch["normals_world"],
                        sample_areas=world_sample_batch["areas"],
                        sample_is_omni=world_sample_batch["is_omni"],
                        flow_intensity=flow_intensity[ready_ids],
                    )
                    forces[ready_ids, 0, :] = result["force_vec"][:, 0, :]
                    torques[ready_ids, 0, :] = result["torque_vec"][:, 0, :]
                else:
                    for bucket_id in unique_bucket_ids:
                        bucket_ready_mask = self._airflow_bucket_ids[ready_ids] == bucket_id
                        bucket_rows = torch.nonzero(bucket_ready_mask, as_tuple=False).squeeze(-1)
                        bucket_ready_ids = ready_ids[bucket_rows]
                        self._airflow_model.cfg.jet.U0 = float(self._airflow_bucket_u0[bucket_id])
                        self._airflow_model.cfg.jet.spread_k = float(self._airflow_bucket_spread_k[bucket_id])

                        result = self._airflow_model.compute_force_result(
                            blower_pos=nozzle_pos[bucket_ready_ids],
                            blower_quat=nozzle_quat[bucket_ready_ids],
                            target_pos=target_pos[bucket_ready_ids],
                            sample_positions_world=world_sample_batch["positions_world"][bucket_rows],
                            sample_normals_world=world_sample_batch["normals_world"][bucket_rows],
                            sample_areas=world_sample_batch["areas"],
                            sample_is_omni=world_sample_batch["is_omni"][bucket_rows],
                            flow_intensity=flow_intensity[bucket_ready_ids],
                        )

                        forces[bucket_ready_ids, 0, :] = result["force_vec"][:, 0, :]
                        torques[bucket_ready_ids, 0, :] = result["torque_vec"][:, 0, :]
            finally:
                self._airflow_model.cfg.jet.U0 = old_u0
                self._airflow_model.cfg.jet.spread_k = old_spread_k
            self._apply_target_wrench(target, forces, torques)

            if self._viz_enabled and self.cfg.viz_cfg.show_sample_markers:
                sample_positions = torch.zeros(
                    self.num_envs,
                    world_sample_batch["positions_world"].shape[1],
                    3,
                    dtype=world_sample_batch["positions_world"].dtype,
                    device=self.device,
                )
                sample_positions[ready_ids] = world_sample_batch["positions_world"]
                visualized_sample_batches.append(sample_positions)

        nozzle_dir = None
        if self._reaction_force_enabled() or self._viz_enabled:
            nozzle_dir = quat_to_axis(nozzle_quat, self.cfg.nozzle_local_axis)

        if self._reaction_force_enabled():
            self._apply_source_reaction_wrench(
                nozzle_pos,
                nozzle_dir,
                flow_intensity,
                env_ids=ready_ids,
            )
        if self._reaction_arrow_viz is not None:
            self._update_reaction_force_visualization(nozzle_pos)

        if self._viz_enabled:
            sample_positions = (
                torch.cat(visualized_sample_batches, dim=1)
                if visualized_sample_batches else None
            )
            self._update_visualization(nozzle_pos, nozzle_dir, sample_positions, flow_intensity)

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
            active_reaction_force_max = self._reaction_force_max
        elif env_ids.numel() == 0:
            return
        else:
            active = env_ids
            active_nozzle_pos = nozzle_pos[env_ids]
            active_nozzle_dir = nozzle_dir[env_ids]
            active_intensity = flow_intensity[env_ids]
            active_origin = self._source_wrench_origin_w()[env_ids]
            active_reaction_force_max = self._reaction_force_max[env_ids]

        if active_intensity.dim() == 2:
            active_intensity = active_intensity[:, 0]

        force_mag = torch.clamp(active_intensity, 0.0, 1.0) * active_reaction_force_max
        forces = -active_nozzle_dir * force_mag.unsqueeze(-1)
        torques = torch.cross(active_nozzle_pos - active_origin, forces, dim=-1)

        self._source_reaction_forces[active, 0, :] = forces
        self._source_reaction_torques[active, 0, :] = torques
        self._write_source_reaction_wrench()

    def _make_mount_pose_visualizer(self) -> VisualizationMarkers:
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/AirflowMountPose",
            markers={
                "mount": sim_utils.SphereCfg(
                    radius=self.cfg.mount_pose_marker_sphere_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=self.cfg.mount_pose_marker_sphere_color,
                    ),
                ),
                "axis": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(
                        self.cfg.mount_pose_marker_arrow_length,
                        self.cfg.mount_pose_marker_arrow_radius,
                        self.cfg.mount_pose_marker_arrow_radius,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=self.cfg.mount_pose_marker_arrow_color,
                    ),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def _update_mount_pose_visualization(self, nozzle_pos: torch.Tensor, nozzle_quat: torch.Tensor) -> None:
        max_envs = min(int(self.cfg.mount_pose_marker_max_envs), self.num_envs)
        if max_envs <= 0:
            return

        nozzle_pos = nozzle_pos[:max_envs]
        nozzle_dir = quat_to_axis(nozzle_quat[:max_envs], self.cfg.nozzle_local_axis)
        nozzle_dir = nozzle_dir / torch.linalg.norm(nozzle_dir, dim=1, keepdim=True).clamp_min(1e-6)

        marker_pos = torch.cat([nozzle_pos, nozzle_pos], dim=0)
        identity_quat = torch.zeros(max_envs, 4, dtype=nozzle_pos.dtype, device=nozzle_pos.device)
        identity_quat[:, 0] = 1.0
        arrow_quat = self._quat_from_local_x_to_axis(nozzle_dir)
        marker_quat = torch.cat([identity_quat, arrow_quat], dim=0)
        marker_indices = torch.cat(
            [
                torch.zeros(max_envs, dtype=torch.long, device=nozzle_pos.device),
                torch.ones(max_envs, dtype=torch.long, device=nozzle_pos.device),
            ],
            dim=0,
        )
        self._mount_pose_viz.visualize(marker_pos, marker_quat, marker_indices=marker_indices)

    def _quat_from_local_x_to_axis(self, axis_w: torch.Tensor) -> torch.Tensor:
        axis_w = axis_w / torch.linalg.norm(axis_w, dim=1, keepdim=True).clamp_min(1e-6)
        local_x = torch.zeros_like(axis_w)
        local_x[:, 0] = 1.0
        rot_axis = torch.cross(local_x, axis_w, dim=1)
        rot_axis_norm = torch.linalg.norm(rot_axis, dim=1, keepdim=True)
        fallback_axis = torch.zeros_like(axis_w)
        fallback_axis[:, 2] = 1.0
        rot_axis = torch.where(
            rot_axis_norm > 1e-6,
            rot_axis / rot_axis_norm.clamp_min(1e-6),
            fallback_axis,
        )
        dot = torch.sum(local_x * axis_w, dim=1).clamp(-1.0, 1.0)
        angle = torch.atan2(rot_axis_norm.squeeze(1), dot)
        return math_utils.quat_from_angle_axis(angle, rot_axis)


@configclass
class SmoothedAirflowActionTermCfg(AirflowActionTermCfg):
    class_type: type = SmoothedAirflowActionTerm

    action_mapping: str = "clip_signed_01"
    """How to map the policy's scalar airflow action into [0, 1]."""

    smoothing_tau_s: float = 0.20
    """First-order low-pass filter time constant in seconds. Set to 0 to disable smoothing."""

    reset_settle_s: float = 0.35
    """Seconds after reset during which airflow computation is skipped for that environment."""

    show_mount_pose_marker: bool = False
    """Whether to visualize the airflow mount position and local nozzle axis."""

    mount_pose_marker_max_envs: int = 8
    """Maximum number of environments visualized with airflow mount markers."""

    mount_pose_marker_sphere_radius: float = 0.025
    """Radius of the airflow mount position marker."""

    mount_pose_marker_sphere_color: tuple[float, float, float] = (0.0, 0.9, 1.0)
    """RGB color of the airflow mount position marker."""

    mount_pose_marker_arrow_length: float = 0.16
    """Length scale of the airflow axis arrow marker."""

    mount_pose_marker_arrow_radius: float = 0.08
    """Radius scale of the airflow axis arrow marker."""

    mount_pose_marker_arrow_color: tuple[float, float, float] = (0.0, 1.0, 0.25)
    """RGB color of the airflow axis arrow marker."""

    airflow_u0_range: tuple[float, float] | None = None
    """Per-episode range for the jet exit velocity U0. None keeps the configured jet value."""

    airflow_spread_k_range: tuple[float, float] | None = None
    """Per-episode range for the jet spread coefficient. None keeps the configured jet value."""

    airflow_randomization_buckets: int = 16
    """Number of random airflow parameter buckets shared across environments."""
