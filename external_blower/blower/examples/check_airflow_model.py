#!/usr/bin/env python3
"""Minimal numerical smoke test for the reusable airflow model.

Run this with the Python interpreter that has Isaac Lab installed:

    python examples/check_airflow_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "blower"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from blower.configs.airflow_cfg import DragCfg, JetCfg, SamplingCfg  # noqa: E402
from blower.models.airflow_physics import AirflowModel, AirflowModelCfg, compute_jet_velocity  # noqa: E402
from blower.models.sampling import build_default_samples_for_type, transform_samples_to_world  # noqa: E402


def check_cuboid_surface_samples() -> None:
    """Check surface coverage and area integration at several resolutions."""
    half_size = torch.tensor([0.035, 0.035, 0.07])
    face_areas = (0.0098, 0.0098, 0.0049)
    for target_area in (6e-5, 2e-5, 1e-3):
        samples = build_default_samples_for_type("cuboid", SamplingCfg(target_area=target_area))
        positions = samples["positions_body"]
        normals = samples["normals_body"]
        areas = samples["areas"]
        assert torch.all(areas > 0)
        assert torch.all(positions.abs() <= half_size + 1e-7)
        assert torch.allclose(areas.sum(), torch.tensor(0.049))
        assert torch.allclose((normals * areas[:, None]).sum(0), torch.zeros(3), atol=1e-7)
        for axis, face_area in enumerate(face_areas):
            for sign in (-1.0, 1.0):
                mask = normals[:, axis] == sign
                assert mask.any()
                assert torch.allclose(positions[mask, axis], sign * half_size[axis])
                assert torch.allclose(areas[mask].sum(), torch.tensor(face_area))
                # Uniform face pressure must act through the face center.
                center = (positions[mask] * areas[mask, None]).sum(0) / areas[mask].sum()
                expected_center = torch.zeros(3)
                expected_center[axis] = sign * half_size[axis]
                assert torch.allclose(center, expected_center, atol=1e-7)
        if target_area == 6e-5:
            assert len(areas) == 860
    print("OK: cuboid surface coverage, representative areas, and pressure centers")


def main() -> None:
    check_cuboid_surface_samples()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_envs = 3
    blower_pos = torch.zeros(num_envs, 3, device=device)
    blower_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device)

    target_pos = torch.tensor(
        [
            [0.50, 0.00, 0.00],   # centered in the jet
            [0.50, 0.04, 0.00],   # radially offset
            [-0.50, 0.00, 0.00],  # behind the nozzle; should receive no force
        ],
        device=device,
    )
    target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device)
    flow_intensity = torch.tensor([[1.0], [0.5], [1.0]], device=device)

    sample_batch = build_default_samples_for_type("ball", SamplingCfg())
    sample_batch = {key: value.to(device) for key, value in sample_batch.items()}
    world_samples = transform_samples_to_world(target_pos, target_quat, sample_batch)

    model = AirflowModel(
        AirflowModelCfg(
            jet=JetCfg(),
            drag=DragCfg(),
            mode="surface",
            min_axial_distance=0.0,
            velocity_cutoff=0.0,
        ),
        device=device,
    )
    result = model.compute_force_result(
        blower_pos=blower_pos,
        blower_quat=blower_quat,
        target_pos=target_pos,
        sample_positions_world=world_samples["positions_world"],
        sample_normals_world=world_samples["normals_world"],
        sample_areas=world_samples["areas"],
        sample_is_omni=world_samples["is_omni"],
        flow_intensity=flow_intensity,
    )

    assert result["force_vec"].shape == (num_envs, 1, 3)
    assert result["torque_vec"].shape == (num_envs, 1, 3)
    assert torch.isfinite(result["force_vec"]).all()
    assert torch.isfinite(result["torque_vec"]).all()
    assert result["force_vec"][0, 0, 0] > 0.0
    assert torch.linalg.norm(result["force_vec"][2, 0]) == 0.0

    exp_velocity = compute_jet_velocity(
        axial=torch.tensor([0.05, 0.20, 0.30], device=device),
        radial=torch.zeros(3, device=device),
        jet_cfg=JetCfg(axial_model="exponential", U0=10.0, xc=0.20, decay_k=2.0),
    )
    exp_expected = torch.tensor([10.0, 10.0, 10.0 * torch.exp(torch.tensor(-0.2)).item()], device=device)
    assert torch.allclose(exp_velocity, exp_expected, atol=1e-6)

    force_x = result["force_vec"][:, 0, 0].detach().cpu().tolist()
    force_mag = result["force_mag"].detach().cpu().tolist()
    print(f"device={device}")
    print(f"force_x={force_x}")
    print(f"force_mag={force_mag}")
    print("OK: airflow model smoke test passed")


if __name__ == "__main__":
    main()
