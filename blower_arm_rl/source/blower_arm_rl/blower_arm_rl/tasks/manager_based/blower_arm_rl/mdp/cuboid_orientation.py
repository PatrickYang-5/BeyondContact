# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Position-and-heading additions to the shared gripper-assisted box-push MDP.

Local Z is the cuboid's undirected long axis. Each episode draws a heading
command for that axis, and the policy must bring the cuboid into the goal
square aligned with it. The tracked quantity is the planar long-axis heading
modulo pi, so the requirement is invariant under the two ways the cuboid can
end up on a different long face: rolling about the long axis by a multiple of
90 degrees, and flipping end over end. Any of the four long side faces may be
the one resting on the table.

Contact is allowed; no airflow qualification, contact latch or reward budgets
gate task progress or success. The fixed robot base has the same XY
origin/axes as the inherited scene frame.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul

from .box_push_common import (
    _box_fully_inside_goal, _box_goal_distance, _ensure_box_push_state,
    _goal_center_xy_local, _reset_goal_hold_state,
)
from .box_push_observation import _add_scheduled_uniform_noise
from .box_push_reward import arm_action_deviation_l2
from .box_push_visual import update_goal_region_visual, update_heading_line_visual


_BOX = SceneEntityCfg("box")


def _state(env):
    cfg = env.cfg.cuboid_orientation
    _ensure_box_push_state(env, cfg.goal_half_extent)
    if not hasattr(env, "_cuboid_orientation"):
        def zeros():
            return torch.zeros(env.num_envs, device=env.device)
        env._cuboid_orientation = SimpleNamespace(
            level=int(cfg.initial_level),
            episode_level=torch.full((env.num_envs,), cfg.initial_level, device=env.device, dtype=torch.long),
            tolerance=zeros(), hold_required=zeros(), hold=zeros(), prev_error=zeros(),
            command=zeros(),
            ever_success=zeros().bool(), stable=zeros().bool(), progress=zeros(),
            support_face=torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
            last_step=torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
            completed=0, successes=0, last_update=0, last_rate=0.0,
            rate_stage=int(cfg.initial_level), rate_samples=0, last_evaluation_rate=0.0,
        )
        state = env._cuboid_orientation
        state.tolerance.fill_(math.radians(cfg.tolerance_deg[state.level]))
        state.hold_required.fill_(cfg.hold_s[state.level])
        # Before the first reset every env carries the command-range center, so
        # geometry queries made during scene setup stay well defined.
        state.command.fill_(math.radians(cfg.command_center_deg))
    return env._cuboid_orientation


def _ids(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


def _support_face(rotation_w):
    """Index of the long face currently resting down: 0/1 = +/-X, 2/3 = +/-Y.

    The four long faces have outward normals +/-body_x and +/-body_y, so the
    supporting one is whichever normal points most steeply downward.
    """
    z = rotation_w[:, 2, :2]
    return torch.stack((z[:, 0], -z[:, 0], z[:, 1], -z[:, 1]), dim=1).argmin(dim=1)


def goal_center_range(cfg, level):
    """Goal-center band for a stage, interpolated start -> final.

    Interpolating rather than tabulating one band per stage keeps the band
    growing smoothly if the stage count changes, and keeps the near edge fixed
    wherever start and final agree on it.
    """
    blend = level / max(len(cfg.tolerance_deg) - 1, 1)
    return tuple(lo + (hi - lo) * blend
                 for lo, hi in zip(cfg.goal_center_range_start, cfg.goal_center_range_final))


def heading_command_embedding(command):
    """Encode a mod-pi heading command as (cos 2t, sin 2t).

    A command and that command turned by 180 degrees ask for the same cuboid
    pose, so the doubled angle is the unique embedding that is also continuous
    across the wrap. It pairs directly with the u u^T components the policy
    already observes, which makes the heading error a bilinear form of the two.
    """
    doubled = 2 * command
    return torch.stack((doubled.cos(), doubled.sin()), dim=1)


def undirected_heading_error(long_axis_b, command=None):
    """Signed turn from the planar long axis to the commanded heading.

    Axis and command both live on the mod-pi circle: rolling the cuboid onto
    another long face leaves the long axis untouched, and flipping it end over
    end negates the axis, so neither may change the error. Comparing doubled
    angles makes that exact and cancels the axis magnitude, so an
    un-normalized (in particular a non-horizontal) axis needs no special case.
    Result is wrapped to (-pi/2, pi/2]; ``command=None`` keeps the original
    fixed target perpendicular to the base X axis.
    """
    x, y = long_axis_b[:, 0], long_axis_b[:, 1]
    axis_sin, axis_cos = 2 * x * y, x.square() - y.square()
    if command is None:
        command_sin, command_cos = torch.zeros_like(x), -torch.ones_like(x)
    else:
        doubled = 2 * torch.as_tensor(command, dtype=x.dtype, device=x.device).expand_as(x)
        command_sin, command_cos = doubled.sin(), doubled.cos()
    sine = command_sin * axis_cos - command_cos * axis_sin
    cosine = command_cos * axis_cos + command_sin * axis_sin
    error = 0.5 * torch.atan2(sine, cosine)
    # Exactly 90 degrees off: atan2 straddles +-pi, so pick one side.
    tie = (sine.abs() < 1e-7) & (cosine < 0)
    return torch.where(tie, torch.full_like(error, math.pi / 2), error)


def cuboid_geometry(env):
    cfg = env.cfg.cuboid_orientation
    command = _state(env).command
    box, robot = env.scene["box"], env.scene["robot"]
    rotation_w = matrix_from_quat(box.data.root_quat_w)
    world_to_base = matrix_from_quat(robot.data.root_quat_w).transpose(1, 2)
    rotation_b = world_to_base @ rotation_w
    pos_b = (world_to_base @ (box.data.root_pos_w - robot.data.root_pos_w).unsqueeze(-1)).squeeze(-1)
    half_size = torch.tensor(cfg.size, device=env.device) * 0.5
    extent = rotation_w.abs() @ half_size
    pos_local = box.data.root_pos_w - env.scene.env_origins
    lo = torch.tensor([cfg.workspace_bounds[0], cfg.workspace_bounds[2]], device=env.device)
    hi = torch.tensor([cfg.workspace_bounds[1], cfg.workspace_bounds[3]], device=env.device)
    inside = ((pos_local[:, :2] - extent[:, :2] >= lo - 1e-6)
              & (pos_local[:, :2] + extent[:, :2] <= hi + 1e-6)).all(dim=1)
    long_axis = rotation_b[:, :, 2]
    # A long face is down exactly when one of the two short body axes points up.
    # ``flat`` is the tight success check; ``on_face`` is the loose one that only
    # fails once the cuboid is genuinely rolling off its face rather than
    # wobbling on it, so it can drive a termination without being trigger-happy.
    upright = rotation_w[:, 2, :2].abs().amax(dim=1)
    flat = upright >= math.cos(math.radians(cfg.flat_tolerance_deg))
    on_face = upright >= math.cos(math.radians(cfg.tumble_tolerance_deg))
    face = _support_face(rotation_w)
    supported = (pos_local[:, 2] - extent[:, 2] - cfg.table_height).abs() <= cfg.support_height_tolerance
    goal = env.cfg.box_push.geometry.goal_center
    half = tuple(s / 2 for s in cfg.size)
    in_goal = _box_fully_inside_goal(env, _BOX, goal, cfg.goal_half_extent, half)
    distance = _box_goal_distance(env, _BOX, goal, cfg.goal_half_extent, half)
    return dict(pos=pos_b, rotation=rotation_b, axis=long_axis, command=command, on_face=on_face, face=face,
                error=undirected_heading_error(long_axis, command), inside=inside, flat=flat, supported=supported,
                valid=inside & flat & supported, in_goal=in_goal, goal_distance=distance,
                velocity=box.data.root_lin_vel_w, omega=box.data.root_ang_vel_w,
                world_to_base=world_to_base)


def reset_cuboid_orientation(env, env_ids):
    cfg, state = env.cfg.cuboid_orientation, _state(env)
    ids = _ids(env, env_ids)
    n, level = len(ids), state.level
    if n == 0:
        return
    zero = torch.zeros(n, device=env.device)
    yaw = (2 * torch.rand(n, device=env.device) - 1) * math.pi
    lying = quat_from_euler_xyz(zero, zero + math.pi / 2, yaw)
    face_roll = quat_from_euler_xyz(zero, zero, torch.randint(0, 4, (n,), device=env.device) * (math.pi / 2))
    quat = quat_mul(lying, face_roll)
    extent = matrix_from_quat(quat).abs() @ (torch.tensor(cfg.size, device=env.device) * 0.5)

    gx0, gx1, gy0, gy1 = goal_center_range(cfg, level)
    goal_lo = torch.tensor([gx0, gy0], device=env.device)
    goal_hi = torch.tensor([gx1, gy1], device=env.device)
    work_lo = torch.tensor([cfg.workspace_bounds[0], cfg.workspace_bounds[2]], device=env.device)
    work_hi = torch.tensor([cfg.workspace_bounds[1], cfg.workspace_bounds[3]], device=env.device)
    if torch.any(goal_lo - cfg.goal_half_extent < work_lo) or torch.any(goal_hi + cfg.goal_half_extent > work_hi):
        raise ValueError("Goal square must fit inside the workspace")
    goals = goal_lo + (goal_hi - goal_lo) * torch.rand(n, 2, device=env.device)
    if not hasattr(env, "_box_push_goal_center_xy"):
        env._box_push_goal_center_xy = _goal_center_xy_local(env, env.cfg.box_push.geometry.goal_center).clone()
    env._box_push_goal_center_xy[ids] = goals
    env._box_push_goal_radius[ids] = cfg.goal_half_extent
    env._box_push_pending_goal_radius.fill_(cfg.goal_half_extent)

    x0, x1, y0, y1 = cfg.spawn_bounds[level]
    lo = torch.maximum(torch.tensor([x0, y0], device=env.device), work_lo + extent[:, :2] + cfg.spawn_clearance)
    hi = torch.minimum(torch.tensor([x1, y1], device=env.device), work_hi - extent[:, :2] - cfg.spawn_clearance)
    # Entire footprint lies on the robot side of the goal's near edge. This
    # guarantees disjoint rectangles even if the initial heading is correct.
    hi[:, 0] = torch.minimum(hi[:, 0], goals[:, 0] - cfg.goal_half_extent - extent[:, 0] - cfg.spawn_clearance)
    if torch.any(hi <= lo):
        raise ValueError("No feasible cuboid spawn strictly before the goal; check ranges")
    xy = lo + (hi - lo) * torch.rand(n, 2, device=env.device)
    pos = env.scene.env_origins[ids].clone()
    pos[:, :2] += xy
    pos[:, 2] += cfg.table_height + extent[:, 2] + cfg.spawn_height_clearance
    box = env.scene["box"]
    box.write_root_pose_to_sim(torch.cat((pos, quat), dim=1), env_ids=ids)
    box.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
    # Draw the heading command independently of the spawn pose, so a stage can
    # widen the command range without also making the spawn easier to align.
    half_range = math.radians(cfg.command_half_range_deg[level])
    command = math.radians(cfg.command_center_deg) + (2 * torch.rand(n, device=env.device) - 1) * half_range

    state.episode_level[ids] = level
    state.tolerance[ids] = math.radians(cfg.tolerance_deg[level])
    state.hold_required[ids] = cfg.hold_s[level]
    state.command[ids] = command
    spawn_rotation = matrix_from_quat(quat)
    state.support_face[ids] = _support_face(spawn_rotation)
    state.prev_error[ids] = undirected_heading_error(spawn_rotation[:, :, 2], command).abs()
    state.hold[ids] = 0
    state.ever_success[ids] = False
    state.stable[ids] = False
    state.progress[ids] = 0
    state.last_step[ids] = env.common_step_counter
    # Use sampled poses directly: avoid relying on PhysX's deferred pose cache.
    residual = (xy - goals).abs() + extent[:, :2] - cfg.goal_half_extent
    env._box_push_prev_goal_distance[ids] = residual.clamp_min(0).norm(dim=1)
    _reset_goal_hold_state(env, ids)
    update_goal_region_visual(env, env.cfg.box_push.geometry.goal_center,
                              goal_radius=cfg.goal_half_extent, env_ids=ids)
    # The command is a base-frame heading and the marker lives in env-local
    # coordinates; this scene's base shares the env origin's XY axes, as the
    # module docstring notes, so the angle carries over unchanged. The marker
    # rides on the cuboid (see _update_orientation); seed it from the sampled
    # spawn XY rather than the pose cache, which PhysX updates only later.
    marker_xy = box.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    marker_xy[ids] = xy
    update_heading_line_visual(env, marker_xy, state.command,
                               center_z=cfg.heading_line_height, env_ids=ids)


def _update_orientation(env):
    cfg, state = env.cfg.cuboid_orientation, _state(env)
    geom = cuboid_geometry(env)
    fresh = state.last_step != env.common_step_counter
    error = geom["error"].abs()
    still = (geom["velocity"].norm(dim=1) <= cfg.linear_speed_tolerance)
    still &= geom["omega"].norm(dim=1) <= cfg.angular_speed_tolerance
    pose_ok = geom["valid"] & geom["in_goal"] & (error <= state.tolerance)
    stable = pose_ok & still
    state.hold[fresh] = torch.where(stable[fresh], state.hold[fresh] + env.step_dt, 0.0)
    state.stable[fresh] = stable[fresh]
    state.ever_success[fresh] |= state.hold[fresh] + 1e-6 >= state.hold_required[fresh]
    # Immediate signed improvement, no contact/flow/history-best gate. The
    # angular term becomes more important near the destination, continuously.
    near = torch.exp(-geom["goal_distance"] / cfg.angle_near_goal_scale)
    state.progress[fresh] = ((state.prev_error - error) / cfg.angle_progress_std * near * geom["valid"])[fresh]
    state.prev_error[fresh] = error[fresh]
    state.last_step[fresh] = env.common_step_counter

    # Shared PLAY tooling reads these buffers. Publish the complete pose goal,
    # never the position-only proxy, without calling the old position-only hold.
    env._box_push_goal_inside[:] = pose_ok
    env._box_push_goal_spatial_inside[:] = geom["in_goal"]
    env._box_push_goal_held_success[:] = state.hold + 1e-6 >= state.hold_required
    env._box_push_goal_hold_steps[:] = (state.hold / env.step_dt).round().long()
    env._box_push_goal_max_hold_steps[:] = torch.maximum(
        env._box_push_goal_max_hold_steps, env._box_push_goal_hold_steps)
    log = env.extras.setdefault("log", {})
    log["Metrics/cuboid_heading_error_deg"] = torch.rad2deg(error).mean()
    log["Metrics/cuboid_in_goal"] = geom["in_goal"].float().mean()
    log["Metrics/cuboid_aligned_fraction"] = pose_ok.float().mean()
    log["Metrics/cuboid_success_rate_step"] = env._box_push_goal_held_success.float().mean()
    log["Metrics/cuboid_held_success"] = log["Metrics/cuboid_success_rate_step"]
    log["Metrics/cuboid_in_workspace"] = geom["inside"].float().mean()
    log["Metrics/cuboid_on_long_face"] = geom["on_face"].float().mean()

    # Carry the command marker along under the cuboid. Drawn at the goal center
    # instead, it is unreadable: the goal is 60 cm wide but only the heading is
    # commanded, so a successful cuboid sits anywhere in the square and can be
    # a quarter metre from the line -- which looks like a miss. Riding on the
    # object, "is the long axis parallel to the line" is directly visible.
    # Gated on the GUI first: this runs from reward terms, several times per
    # step, and headless training must not pay for the index math.
    if env.sim.has_gui() or env.sim.has_rtx_sensors():
        update_heading_line_visual(
            env,
            env.scene["box"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2],
            state.command,
            center_z=cfg.heading_line_height,
            # Empty on the repeat calls within a step, so the writes happen once.
            env_ids=fresh.nonzero(as_tuple=False).flatten(),
        )
    return geom, state


def cuboid_orientation_dense(env):
    geom, _ = _update_orientation(env)
    near = torch.exp(-geom["goal_distance"] / env.cfg.cuboid_orientation.angle_near_goal_scale)
    return near * (1 - geom["error"].abs() / (math.pi / 2)) * geom["valid"]


def cuboid_orientation_progress(env):
    return _update_orientation(env)[1].progress


def cuboid_pose_inside(env):
    geom, state = _update_orientation(env)
    return (geom["valid"] & geom["in_goal"] & (geom["error"].abs() <= state.tolerance)).float()


def cuboid_orientation_stable(env):
    geom, state = _update_orientation(env)
    gate = geom["valid"] & geom["in_goal"] & (geom["error"].abs() <= state.tolerance)
    return gate.float() * torch.exp(-(geom["velocity"].square().sum(1) / 0.04**2
                                     + geom["omega"].square().sum(1) / 0.25**2))


def cuboid_arm_action_deviation(env):
    cfg = env.cfg.box_push.rewards
    penalty = arm_action_deviation_l2(
        env, action_term_name="arm_action", penalty_warmup_steps=cfg.arm_action_deviation_warmup_steps,
        penalty_ramp_steps=cfg.arm_action_deviation_ramp_steps, penalty_start_scale=cfg.arm_action_deviation_start_scale)
    return penalty * cuboid_pose_inside(env)


def cuboid_tumbled(env):
    """Terminate once the cuboid leaves the long face it was reset on.

    Two independent signals, because neither alone is enough. The tilt test
    catches a cuboid that is not resting on any long face — tipped onto an end
    face, or balanced up on an edge. It cannot catch a roll on its own: the
    tilt measure has 90 degree period, so a pure roll only dips below the
    threshold inside a narrow window around 45 degrees and a fast roll can step
    straight over it. Comparing the supporting face against the one recorded at
    reset closes that hole, and lands exactly when the roll commits.

    The success test stays invariant to *which* long face is down; this only
    forbids swapping faces mid-episode, which is what separates reorienting the
    cuboid in the plane from tumbling it around.
    """
    cfg, state = env.cfg.cuboid_orientation, _state(env)
    geom = cuboid_geometry(env)
    settled = env.episode_length_buf.float() * env.step_dt > cfg.tumble_ignore_initial_s
    return (~geom["on_face"] | (geom["face"] != state.support_face)) & settled


def cuboid_tumbled_penalty(env):
    return cuboid_tumbled(env).float()


def cuboid_out_of_workspace(env):
    geom = cuboid_geometry(env)
    cfg = env.cfg.cuboid_orientation
    height = env.scene["box"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - cfg.table_height
    return ~geom["inside"] | (height < 0.0) | (height > cfg.max_center_height)


def cuboid_out_of_workspace_penalty(env):
    return cuboid_out_of_workspace(env).float()


def cuboid_policy_pose(env):
    geom, cfg = cuboid_geometry(env), env.cfg.cuboid_orientation
    pos = _add_scheduled_uniform_noise(env, geom["pos"], cfg.position_noise_range, 30000, 60000)
    u, v, _ = geom["axis"].unbind(1)
    # The controlled quantity is the undirected heading of the long axis in
    # the base XY plane.  Its doubled-angle representation is invariant to an
    # end-for-end axis flip and directly matches the heading command encoding.
    # Do not normalize the XY projection: the embedding magnitude u^2 + v^2
    # then also falls below one when the long axis tilts out of the table plane.
    planar_axis = torch.stack((u.square() - v.square(), 2 * u * v), dim=1)
    return torch.cat((pos, planar_axis), dim=1)


def cuboid_goal_observation(env):
    """Undirected heading command for the policy as ``(cos 2t, sin 2t)``."""
    state = _state(env)
    return heading_command_embedding(state.command)


def cuboid_privileged_state(env):
    geom, state = _update_orientation(env)
    goal_delta = _goal_center_xy_local(env, env.cfg.box_push.geometry.goal_center) - (
        env.scene["box"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2])
    # The signed error saves the value function from recovering the command
    # from the full rotation matrix, which is the part it needs most.
    goal_half_extent = torch.full_like(state.tolerance, env.cfg.cuboid_orientation.goal_half_extent)
    return torch.cat((geom["velocity"], geom["omega"], geom["rotation"].reshape(env.num_envs, 9), goal_delta,
                      geom["goal_distance"][:, None], geom["in_goal"].float()[:, None],
                      (state.hold / state.hold_required.clamp_min(1e-6)).clamp(max=1)[:, None],
                      heading_command_embedding(state.command), geom["error"][:, None],
                      state.tolerance[:, None], goal_half_extent[:, None]), dim=1)


def cuboid_orientation_curriculum(env, env_ids):
    cfg, state = env.cfg.cuboid_orientation, _state(env)
    ids = _ids(env, env_ids)
    ids = ids[(env.episode_length_buf[ids] > 0) & (state.episode_level[ids] == state.level)]
    if len(ids):
        success = ((state.hold[ids] + 1e-6 >= state.hold_required[ids])
                   & state.stable[ids] & ~env.termination_manager.terminated[ids])
        state.completed += len(ids)
        state.successes += int(success.sum().item())
        state.last_rate = state.successes / state.completed
        state.rate_stage, state.rate_samples = state.level, state.completed
    enough = (state.completed >= cfg.min_completed_episodes
              and env.common_step_counter >= cfg.curriculum_warmup_steps
              and env.common_step_counter - state.last_update >= cfg.curriculum_interval_steps)
    if enough and not cfg.evaluation:
        rate = state.successes / state.completed
        state.last_evaluation_rate = rate
        if rate >= cfg.advance_success_rate:
            state.level = min(state.level + 1, len(cfg.tolerance_deg) - 1)
        elif rate < cfg.retreat_success_rate:
            state.level = max(state.level - 1, 0)
        state.last_update = env.common_step_counter
        state.completed = state.successes = 0
        env._box_push_robot_reset_range_scale.fill_(1 - state.level / max(len(cfg.tolerance_deg) - 1, 1))
    return {"stage": state.level, "completed_episodes": state.completed,
            "success_rate": state.last_rate, "success_rate_stage": state.rate_stage,
            "success_rate_samples": state.rate_samples, "last_evaluation_success_rate": state.last_evaluation_rate,
            "next_tolerance_deg": cfg.tolerance_deg[state.level],
            "next_command_half_range_deg": cfg.command_half_range_deg[state.level],
            "robot_reset_range_scale": env._box_push_robot_reset_range_scale}
