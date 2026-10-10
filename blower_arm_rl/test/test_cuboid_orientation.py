# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU behavioral checks of actual task functions, without launching Kit."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

TASKS = Path(__file__).resolve().parents[1] / "source/blower_arm_rl/blower_arm_rl/tasks/manager_based"


@pytest.fixture
def task():
    root = Path(importlib.util.find_spec("isaaclab").origin).parent
    spec = importlib.util.spec_from_file_location("cuboid_test_math", root / "utils/math.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ns = dict(torch=torch, math=math, SimpleNamespace=NS, math_utils=m,
              SceneEntityCfg=lambda name: NS(name=name, body_ids=slice(None)),
              _BOX=NS(name="box"), matrix_from_quat=m.matrix_from_quat,
              quat_from_euler_xyz=m.quat_from_euler_xyz, quat_mul=m.quat_mul,
              _add_scheduled_uniform_noise=lambda env, value, *args: value,
              update_goal_region_visual=lambda *args, **kwargs: None,
              update_heading_line_visual=lambda *args, **kwargs: None)
    # Preserve postponed annotations while loading actual functions. No copies
    # of the geometry/reward implementation live in this fixture.
    for filename, selected in (("box_push_common.py", None),
                               ("box_push_reward.py", {"box_goal_distance_tanh", "box_goal_progress"}),
                               ("cuboid_orientation.py", None)):
        source = TASKS / "blower_arm_rl/mdp" / filename
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and (selected is None or n.name in selected)]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(tree, str(source), "exec"), ns)
    source = TASKS / "piper_blower_rl/blower_gripper_assist_cuboid_orientation_cfg.py"
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "CuboidOrientationTaskCfg")
    cls.decorator_list = []
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), ns)
    return NS(**ns), m


def make_env(task, n=4):
    code, _ = task
    identity = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    robot = NS(data=NS(root_quat_w=identity.clone(), root_pos_w=torch.zeros(n, 3)))
    data = NS(root_quat_w=identity.clone(), root_pos_w=torch.zeros(n, 3),
              root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3))
    box = NS(data=data)
    def pose(value, env_ids):
        data.root_pos_w[env_ids], data.root_quat_w[env_ids] = value[:, :3], value[:, 3:]
    def velocity(value, env_ids):
        data.root_lin_vel_w[env_ids], data.root_ang_vel_w[env_ids] = value[:, :3], value[:, 3:]
    box.write_root_pose_to_sim, box.write_root_velocity_to_sim = pose, velocity
    class Scene(dict):
        pass
    scene = Scene(robot=robot, box=box)
    scene.env_origins = torch.zeros(n, 3)
    env = NS(cfg=NS(cuboid_orientation=code.CuboidOrientationTaskCfg(),
                    box_push=NS(geometry=NS(goal_center=(1.00, 0., 0.006)))),
             num_envs=n, device="cpu", scene=scene, step_dt=1/30, common_step_counter=0,
             episode_length_buf=torch.zeros(n, dtype=torch.long), extras={},
             # Headless: the marker update is gated on this before it touches USD.
             sim=NS(has_gui=lambda: False, has_rtx_sensors=lambda: False),
             termination_manager=NS(terminated=torch.zeros(n, dtype=torch.bool)))
    code.reset_cuboid_orientation(env, None)
    return env


def set_heading(env, m, heading):
    angles = torch.as_tensor(heading, dtype=torch.float32).expand(env.num_envs)
    zero = torch.zeros_like(angles)
    env.scene["box"].data.root_quat_w[:] = m.quat_from_euler_xyz(zero, zero + math.pi/2, angles)


def put_in_goal(env, m, heading=None, code=None):
    """Center the cuboid in its goal square, by default at the commanded heading."""
    env.scene["box"].data.root_pos_w[:, :2] = env._box_push_goal_center_xy + env.scene.env_origins[:, :2]
    env.scene["box"].data.root_pos_w[:, 2] = env.scene.env_origins[:, 2] + 0.035
    set_heading(env, m, env._cuboid_orientation.command if heading is None else heading)


def pin_command(env, angle=math.pi/2):
    """Freeze the sampled command so a test can use fixed expected angles."""
    env._cuboid_orientation.command.fill_(angle)
    return env


def position_reward(code, env, progress=False):
    fn = code.box_goal_progress if progress else code.box_goal_distance_tanh
    return fn(env, std=0.25, goal_center=(1.00, 0., 0.006), goal_radius=0.2,
              box_half_extent=(0.035, 0.035, 0.07))


def test_undirected_heading_and_shortest_turn(task):
    code, _ = task
    angles = torch.deg2rad(torch.tensor([90., -90., 0., 180., 60., 120., 240., 300.]))
    axis = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=1)
    error = code.undirected_heading_error(axis)
    torch.testing.assert_close(torch.rad2deg(error), torch.tensor([0., 0., 90., 90., 30., -30., 30., -30.]), atol=2e-5, rtol=0)
    torch.testing.assert_close(code.undirected_heading_error(-axis), error)
    # The default target is the pi/2 case of the general command.
    torch.testing.assert_close(code.undirected_heading_error(axis, math.pi/2), error)


def test_heading_error_tracks_arbitrary_command(task):
    code, _ = task
    headings = torch.deg2rad(torch.tensor([0., 30., 100., 175.]))
    axis = torch.stack((headings.cos(), headings.sin(), torch.zeros_like(headings)), dim=1)
    command = torch.deg2rad(torch.tensor([0., 0., 130., 5.]))
    error = code.undirected_heading_error(axis, command)
    torch.testing.assert_close(torch.rad2deg(error), torch.tensor([0., -30., 30., 10.]), atol=2e-5, rtol=0)
    # A command is mod pi, and so is the axis: both flips leave the error alone.
    torch.testing.assert_close(code.undirected_heading_error(axis, command + math.pi), error, atol=2e-5, rtol=0)
    torch.testing.assert_close(code.undirected_heading_error(-axis, command), error, atol=2e-5, rtol=0)
    # Turning axis and command together is a no-op; error never exceeds 90 deg.
    for turn in torch.deg2rad(torch.tensor([17., 63., 155.])):
        turned = headings + turn
        rotated = torch.stack((turned.cos(), turned.sin(), torch.zeros_like(turned)), dim=1)
        torch.testing.assert_close(code.undirected_heading_error(rotated, command + turn), error, atol=2e-5, rtol=0)
    dense = code.undirected_heading_error(axis, torch.rand(4) * 10 - 5)
    assert (dense.abs() <= math.pi/2 + 1e-6).all()


@pytest.mark.parametrize("level", range(4))
def test_reset_arbitrary_heading_all_faces_strictly_outside_goal(task, level):
    code, m = task
    env = make_env(task, n=2048)
    env.scene.env_origins[:] = torch.tensor([2., -3., 1.])
    env.scene["robot"].data.root_pos_w[:] = torch.tensor([2., -3., 1.03])
    code._state(env).level = level
    code.reset_cuboid_orientation(env, None)
    geom = code.cuboid_geometry(env)
    assert geom["valid"].all() and not geom["in_goal"].any()
    assert (geom["axis"][:, 2].abs() < 1e-5).all()
    err = torch.rad2deg(geom["error"].abs())
    assert err.min() < 2 and err.max() > 88
    rotation = m.matrix_from_quat(env.scene["box"].data.root_quat_w)
    assert len(torch.unique(rotation[:, 2, :2].round(), dim=0)) == 4
    extent = rotation.abs() @ torch.tensor([0.035, 0.035, 0.07])
    far_x = env.scene["box"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] + extent[:, 0]
    cfg = env.cfg.cuboid_orientation
    near_edge = env._box_push_goal_center_xy[:, 0] - cfg.goal_half_extent - cfg.spawn_clearance
    assert (far_x <= near_edge + 1e-6).all()
    torch.testing.assert_close(position_reward(code, env, True), torch.zeros(env.num_envs), atol=3e-6, rtol=0)
    env.common_step_counter += 1
    assert not code.cuboid_pose_inside(env).any()
    assert not code.cuboid_orientation_stable(env).any()
    assert not code._state(env).ever_success.any()


def test_partial_reset_preserves_other_envs_and_no_fake_progress(task):
    code, _ = task
    env = make_env(task)
    state = code._state(env)
    state.hold[:] = 0.3
    pose = env.scene["box"].data.root_pos_w.clone()
    goal = env._box_push_goal_center_xy.clone()
    command = state.command.clone()
    code.reset_cuboid_orientation(env, slice(0, 4, 2))
    torch.testing.assert_close(env.scene["box"].data.root_pos_w[1::2], pose[1::2])
    torch.testing.assert_close(env._box_push_goal_center_xy[1::2], goal[1::2])
    torch.testing.assert_close(state.command[1::2], command[1::2])
    torch.testing.assert_close(state.hold, torch.tensor([0., 0.3, 0., 0.3]))
    torch.testing.assert_close(position_reward(code, env, True), torch.zeros(4), atol=1e-6, rtol=0)


def test_no_motion_cannot_succeed_even_when_initial_heading_is_correct(task):
    code, m = task
    env = make_env(task)
    set_heading(env, m, env._cuboid_orientation.command)
    code._state(env).prev_error.zero_()
    for step in range(1, 301):
        env.common_step_counter = step
        # Holding the commanded heading is not progress; only closing the angle
        # is. Quaternion round-off keeps this at float noise, never positive.
        progress = code.cuboid_orientation_progress(env)
        assert (progress <= 0).all() and progress.abs().max() < 1e-6
        assert not code.cuboid_orientation_stable(env).any()
        assert not code.cuboid_pose_inside(env).any()
    assert not code._state(env).ever_success.any()


def test_position_dense_and_progress_available_without_airflow_or_contact_gate(task):
    code, _ = task
    env = make_env(task)
    initial = position_reward(code, env)
    # env deliberately has no action manager or contact sensor; a physical push
    # must earn the same task progress as the same displacement from airflow.
    env.scene["box"].data.root_pos_w[:, 0] += 0.05
    assert (position_reward(code, env) > initial).all()
    assert (position_reward(code, env, True) > 0).all()
    assert not position_reward(code, env, True).any()
    env.scene["box"].data.root_pos_w[:, 0] -= 0.05
    assert (position_reward(code, env, True) < 0).all()


def test_heading_progress_signed_and_position_shaping_continuous(task):
    code, m = task
    env = pin_command(make_env(task))
    put_in_goal(env, m, math.pi/4)
    code._state(env).prev_error[:] = math.pi/4
    set_heading(env, m, math.pi/3)
    env.common_step_counter = 1
    assert (code.cuboid_orientation_progress(env) > 0).all()
    near = code.cuboid_orientation_dense(env)
    set_heading(env, m, math.pi/4)
    env.common_step_counter += 1
    assert (code.cuboid_orientation_progress(env) < 0).all()
    set_heading(env, m, math.pi/3)
    env.scene["box"].data.root_pos_w[:, 0] -= 0.4
    far = code.cuboid_orientation_dense(env)
    assert ((far > 0) & (far < near)).all()


@pytest.mark.parametrize("face", range(4))
def test_combined_goal_requires_hold_and_accepts_all_long_faces(task, face):
    code, m = task
    env = make_env(task)
    put_in_goal(env, m)
    zero = torch.zeros(env.num_envs)
    roll = m.quat_from_euler_xyz(zero, zero, zero + face * math.pi/2)
    box = env.scene["box"].data
    box.root_quat_w[:] = m.quat_mul(box.root_quat_w, roll)
    state = code._state(env)
    env.common_step_counter = 1
    for _ in range(20):
        code.cuboid_orientation_stable(env)
    torch.testing.assert_close(state.hold, torch.full((4,), env.step_dt))
    assert env.extras["log"]["Metrics/cuboid_success_rate_step"] == 0
    for step in range(2, 17):
        env.common_step_counter = step
        code.cuboid_orientation_stable(env)
    assert env.extras["log"]["Metrics/cuboid_success_rate_step"] == 1
    box.root_ang_vel_w[:2, 2] = 1
    env.common_step_counter += 1
    code.cuboid_orientation_stable(env)
    assert env.extras["log"]["Metrics/cuboid_success_rate_step"] == 0.5
    assert not env.termination_manager.terminated.any()


@pytest.mark.parametrize("face", range(4))
@pytest.mark.parametrize("flip", [False, True])
def test_any_long_face_and_end_flip_satisfy_the_same_command(task, face, flip):
    code, m = task
    env = make_env(task, n=64)
    command = env._cuboid_orientation.command
    put_in_goal(env, m, command + (math.pi if flip else 0.0))
    box = env.scene["box"].data
    zero = torch.zeros(env.num_envs)
    # Rolling onto another long face turns the cuboid about its own long axis,
    # which must leave both the heading error and the flat/support check alone.
    box.root_quat_w[:] = m.quat_mul(box.root_quat_w, m.quat_from_euler_xyz(zero, zero, zero + face * math.pi/2))
    geom = code.cuboid_geometry(env)
    assert geom["valid"].all() and geom["in_goal"].all()
    torch.testing.assert_close(geom["error"], torch.zeros(env.num_envs), atol=2e-5, rtol=0)
    env.common_step_counter = 1
    assert code.cuboid_pose_inside(env).all()
    assert (code.cuboid_orientation_dense(env) > 0).all()


@pytest.mark.parametrize("widen", [False, True])
def test_goal_center_band_grows_outward_from_a_fixed_near_edge(task, widen):
    code, _ = task
    env = make_env(task, n=2048)
    cfg = env.cfg.cuboid_orientation
    # Cover both the released fixed range and an explicitly widening curriculum.
    if widen:
        x0, x1, y0, y1 = cfg.goal_center_range_start
        far_edge = (x1 + cfg.workspace_bounds[1] - cfg.goal_half_extent) / 2
        cfg.goal_center_range_final = (x0, far_edge, y0, y1)
    stages = len(cfg.tolerance_deg)
    start, final = cfg.goal_center_range_start, cfg.goal_center_range_final
    assert code.goal_center_range(cfg, 0) == pytest.approx(start)
    assert code.goal_center_range(cfg, stages - 1) == pytest.approx(final)
    reach = []
    for level in range(stages):
        x0, x1, y0, y1 = code.goal_center_range(cfg, level)
        code._state(env).level = level
        code.reset_cuboid_orientation(env, None)
        goals = env._box_push_goal_center_xy
        assert (goals[:, 0] >= x0 - 1e-6).all() and (goals[:, 0] <= x1 + 1e-6).all()
        assert (goals[:, 1] >= y0 - 1e-6).all() and (goals[:, 1] <= y1 + 1e-6).all()
        assert goals[:, 0].max() > x1 - 0.02 and goals[:, 0].min() < x0 + 0.02
        # A widening band must never drop the short transports it already had.
        assert x0 == pytest.approx(start[0])
        reach.append(x1)
        assert (env._box_push_goal_radius == cfg.goal_half_extent).all()
        # The goal square still fits the workspace at every stage.
        assert x0 - cfg.goal_half_extent >= cfg.workspace_bounds[0] - 1e-6
        assert x1 + cfg.goal_half_extent <= cfg.workspace_bounds[1] + 1e-6
    assert reach == sorted(reach)
    if widen:
        assert reach[-1] > reach[0]
    else:
        assert reach == pytest.approx([start[1]] * stages)


def test_command_range_widens_with_stage_and_stays_uniform(task):
    code, _ = task
    env = make_env(task, n=4096)
    cfg = env.cfg.cuboid_orientation
    center = math.radians(cfg.command_center_deg)
    spans = []
    for level in range(len(cfg.tolerance_deg)):
        code._state(env).level = level
        code.reset_cuboid_orientation(env, None)
        offset = env._cuboid_orientation.command - center
        half = math.radians(cfg.command_half_range_deg[level])
        assert (offset.abs() <= half + 1e-6).all()
        assert offset.abs().max() > 0.9 * half
        spans.append(half)
        # The spawn heading stays fully random and independent of the command.
        error = code.cuboid_geometry(env)["error"].abs()
        assert torch.rad2deg(error).min() < 2 and torch.rad2deg(error).max() > 88
    assert spans == sorted(spans) and spans[-1] > spans[0]
    # Capped at the two diagonals: a half range above 90 deg would wrap on the
    # mod-pi command circle and cover some headings twice.
    assert spans[-1] == pytest.approx(math.pi/4)
    assert spans[-1] <= math.pi/2 + 1e-9
    # Commanded headings therefore never run along the base X axis.
    command = env._cuboid_orientation.command % math.pi
    assert (command > math.pi/4 - 1e-6).all() and (command < 3*math.pi/4 + 1e-6).all()


def test_tumbling_off_the_long_face_terminates_but_a_settled_face_does_not(task):
    code, m = task
    env = pin_command(make_env(task, n=4))
    put_in_goal(env, m)
    box = env.scene["box"].data
    zero = torch.zeros(env.num_envs)
    state = code._state(env)
    env.episode_length_buf[:] = 60  # Past tumble_ignore_initial_s.
    flat_quat = m.quat_from_euler_xyz(zero, zero + math.pi/2, zero + math.pi/2)

    # Whichever of the four long faces an episode starts on, resting squarely
    # on it is never a tumble, and the recorded face round-trips.
    for face in range(4):
        box.root_quat_w[:] = m.quat_mul(flat_quat, m.quat_from_euler_xyz(zero, zero, zero + face * math.pi/2))
        state.support_face[:] = code._support_face(m.matrix_from_quat(box.root_quat_w))
        assert code.cuboid_geometry(env)["on_face"].all()
        assert (code.cuboid_geometry(env)["face"] == state.support_face).all()
        assert not code.cuboid_tumbled(env).any()
        assert not code.cuboid_tumbled_penalty(env).any()

    # Reset on face 0, then roll away from it by increasing amounts.
    box.root_quat_w[:] = flat_quat
    state.support_face[:] = code._support_face(m.matrix_from_quat(box.root_quat_w))
    for roll_deg, tumbled in ((0.0, False), (20.0, False), (44.0, True), (50.0, True),
                              (90.0, True), (180.0, True)):
        roll = m.quat_from_euler_xyz(zero, zero, zero + math.radians(roll_deg))
        box.root_quat_w[:] = m.quat_mul(flat_quat, roll)
        assert bool(code.cuboid_tumbled(env).all()) is tumbled, f"roll {roll_deg} deg"

    # A completed roll is caught by the face comparison even though the cuboid
    # is squarely on a face again -- the tilt test alone would miss it.
    box.root_quat_w[:] = m.quat_mul(flat_quat, m.quat_from_euler_xyz(zero, zero, zero + math.pi/2))
    assert code.cuboid_geometry(env)["on_face"].all()
    assert code.cuboid_tumbled(env).all()

    # Standing on an end face (long axis vertical) is a tumble too.
    box.root_quat_w[:] = torch.tensor([1., 0., 0., 0.])
    assert not code.cuboid_geometry(env)["on_face"].any()
    assert code.cuboid_tumbled(env).all()

    # Freshly reset envs are exempt while the drop settles.
    env.episode_length_buf[:] = 0
    assert not code.cuboid_tumbled(env).any()


def test_reset_records_the_spawn_support_face(task):
    code, m = task
    env = make_env(task, n=1024)
    state = code._state(env)
    geom = code.cuboid_geometry(env)
    # Reset spawns all four faces uniformly, and each env's recorded face is
    # the one it actually landed on, so no episode starts already tumbled.
    assert (state.support_face == geom["face"]).all()
    assert sorted(torch.bincount(state.support_face, minlength=4).tolist())[0] > 150
    env.episode_length_buf[:] = 60
    assert not code.cuboid_tumbled(env).any()


def test_correct_position_wrong_angle_is_not_complete_goal(task):
    code, m = task
    env = make_env(task)
    put_in_goal(env, m, env._cuboid_orientation.command + math.pi/2)
    env.common_step_counter = 1
    geom = code.cuboid_geometry(env)
    assert geom["in_goal"].all()
    torch.testing.assert_close(torch.rad2deg(geom["error"].abs()), torch.full((4,), 90.), atol=2e-4, rtol=0)
    assert not code.cuboid_pose_inside(env).any()
    assert not code.cuboid_orientation_stable(env).any()


def test_vertical_edge_balance_airborne_and_footprint_rejected(task):
    code, m = task
    # Pin the heading: env 3 is pushed out along Y, so the rejection must come
    # from the 0.07 long-axis footprint rather than from whichever command the
    # current stage happened to draw.
    env = pin_command(make_env(task))
    put_in_goal(env, m)
    box = env.scene["box"].data
    box.root_quat_w[0] = torch.tensor([1., 0., 0., 0.])
    zero = torch.zeros(1)
    roll = m.quat_from_euler_xyz(zero, zero, zero + math.pi/4)
    box.root_quat_w[1:2] = m.quat_mul(box.root_quat_w[1:2], roll)
    box.root_pos_w[2, 2] = 0.10
    # Far enough out that the 0.07 long-axis footprint crosses the Y boundary.
    box.root_pos_w[3, 1] = env.cfg.cuboid_orientation.workspace_bounds[3] - 0.035
    assert not code.cuboid_geometry(env)["valid"].any()
    assert code.cuboid_out_of_workspace(env)[3]


def test_full_footprint_distance_and_legacy_center_distance(task):
    code, m = task
    # Pin the heading: the expected distance below is the long axis lying along
    # Y, so the Y footprint is the 0.07 half length rather than the 0.035 one.
    env = pin_command(make_env(task))
    put_in_goal(env, m)
    half = env.cfg.cuboid_orientation.goal_half_extent
    # Shift Y until the 0.07 long-axis footprint pokes 0.05 m past the goal edge.
    env.scene["box"].data.root_pos_w[:, 1] += half - 0.02
    geom = code.cuboid_geometry(env)
    assert not geom["in_goal"].any()
    torch.testing.assert_close(geom["goal_distance"], torch.full((4,), 0.05), atol=1e-6, rtol=0)
    # The legacy center-only distance ignores the footprint, so it is still zero.
    assert not code._box_goal_distance(env, code._BOX, (1.00, 0., 0.), half).any()
    scalar = code._box_fully_inside_goal(env, code._BOX, (1.00, 0., 0.), half, 0.01)
    assert scalar.all()


def test_curriculum_counts_finished_episodes_and_freezes_active_thresholds(task):
    code, _ = task
    env = make_env(task)
    cfg, state = env.cfg.cuboid_orientation, code._state(env)
    cfg.min_completed_episodes = 4
    env.common_step_counter = 4000
    state.hold[:] = state.hold_required
    state.stable[:] = True
    code.cuboid_orientation_curriculum(env, None)
    assert state.level == 0
    env.episode_length_buf[:] = 200
    code.cuboid_orientation_curriculum(env, torch.tensor([0]))
    assert state.level == 0
    old_tolerance = state.tolerance.clone()
    code.cuboid_orientation_curriculum(env, torch.tensor([1, 2, 3]))
    assert state.level == 1
    torch.testing.assert_close(state.tolerance, old_tolerance)
    env.common_step_counter += 3000
    code.cuboid_orientation_curriculum(env, None)
    assert state.completed == 0  # Old-stage episodes cannot promote again.
    code.reset_cuboid_orientation(env, None)
    state.hold[:] = state.hold_required
    state.stable[:] = True
    env.termination_manager.terminated[:] = True
    code.cuboid_orientation_curriculum(env, None)
    assert state.level == 0


def test_policy_axis_sign_invariance_and_goal_observability(task):
    code, m = task
    env = make_env(task)
    set_heading(env, m, math.pi/3)
    before = code.cuboid_policy_pose(env)
    set_heading(env, m, math.pi/3 + math.pi)
    torch.testing.assert_close(code.cuboid_policy_pose(env), before, atol=1e-6, rtol=0)
    assert before.shape == (4, 5)
    # Heading 60 deg is encoded as (cos 120 deg, sin 120 deg).
    expected_axis = torch.tensor([math.cos(2 * math.pi/3), math.sin(2 * math.pi/3)]).repeat(4, 1)
    torch.testing.assert_close(before[:, 3:], expected_axis, atol=2e-5, rtol=0)

    privileged = code.cuboid_privileged_state(env)
    assert privileged.shape == (4, 25)
    torch.testing.assert_close(privileged[:, -2], code._state(env).tolerance)
    assert (privileged[:, -1] == env.cfg.cuboid_orientation.goal_half_extent).all()

    # Only the commanded heading is policy-visible. Curriculum tolerance and
    # goal half extent are critic-only privileged inputs.
    goal = code.cuboid_goal_observation(env)
    assert goal.shape == (4, 2)
    command = env._cuboid_orientation.command
    torch.testing.assert_close(goal, torch.stack(((2*command).cos(), (2*command).sin()), dim=1))
    pin_command(env, math.pi/3)
    assert not torch.allclose(code.cuboid_goal_observation(env), goal)


@pytest.mark.parametrize("evaluation", [False, True])
def test_success_rate_updates_before_curriculum_thresholds(task, evaluation):
    code, _ = task
    env = make_env(task)
    env.cfg.cuboid_orientation.evaluation = evaluation
    state = code._state(env)
    env.common_step_counter = 10
    env.episode_length_buf[:] = 10
    state.hold[:2] = state.hold_required[:2]
    state.stable[:2] = True
    result = code.cuboid_orientation_curriculum(env, torch.tensor([0]))
    assert result["success_rate"] == 1 and result["success_rate_samples"] == 1
    result = code.cuboid_orientation_curriculum(env, torch.tensor([1, 2, 3]))
    assert result["success_rate"] == 0.5 and result["success_rate_samples"] == 4
    assert state.level == 0
    if evaluation:
        env.common_step_counter = 10000
        env.cfg.cuboid_orientation.min_completed_episodes = 1
        code.cuboid_orientation_curriculum(env, None)
        assert state.level == 0


def test_contact_geometry_uses_rotated_long_side(task):
    _, m = task
    source = TASKS / "blower_arm_rl/mdp/box_push_termination.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
                and n.name == "body_local_boxes_box_contact_force")
    ns = dict(torch=torch, math_utils=m, ManagerBasedRLEnv=object, Sequence=list,
              SceneEntityCfg=lambda name: NS(name=name, body_ids=slice(None)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    env = make_env(task, n=2)
    env.episode_length_buf[:] = 10
    set_heading(env, m, math.pi/2)
    env.scene["box"].data.root_pos_w[:] = torch.tensor([0.5, 0., 0.035])
    env.scene["robot"].data.body_pos_w = torch.tensor([[[0.5, 0.06, 0.035]], [[0.56, 0., 0.035]]])
    env.scene["robot"].data.body_quat_w = torch.tensor([[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]])
    env.scene.sensors = {"contact_forces": NS(data=NS(net_forces_w_history=torch.ones(2, 1, 1, 3)))}
    result = ns["body_local_boxes_box_contact_force"](
        env, force_threshold=0.01, box_half_extent=0.04,
        box_centers=((0., 0., 0.),), box_half_extents=((0.005, 0.005, 0.005),),
        target_box_half_extents=(0.035, 0.035, 0.07))
    assert result.tolist() == [True, False]
