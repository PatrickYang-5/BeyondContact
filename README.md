# BeyondContact

Anonymous research release for airflow-assisted manipulation using Piper-L.

## Included components

- `external_blower/blower`: reusable airflow extension, force/torque model, target objects and visualisation.
- `blower_arm_rl`: PPO training and evaluation for **gripper-assisted box pushing** and **rectangular reorientation**.
- `docs`: static project website for GitHub Pages.

Only the following tasks are registered (plus their `-Play-v0` variants):

| Task | Gym ID |
| --- | --- |
| Gripper-assisted box pushing | `Isaac-Blower-Piper-GripperAssistBoxPush-v0` |
| Rectangular reorientation | `Isaac-Blower-Piper-GripperAssistCuboidOrientation-v0` |

The reorientation task uses the original `CuboidOrientation` identifier. Shared base configurations and MDP utilities are retained as dependencies; they do not register additional tasks.

## Installation

Use an isolated environment with Isaac Sim 5.1.0, a compatible Isaac Lab installation, Python 3.11, PyTorch and RSL-RL. Simulator dependencies are not bundled. Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
Run from this repository root, using that environment's Python:

```bash
python -m pip install -e external_blower/blower/source/blower
python -m pip install -e blower_arm_rl/source/blower_arm_rl
python blower_arm_rl/scripts/list_envs.py
```

Editable installs are required: robot and object assets are resolved relative to the source checkout. Keep this checkout separate from other projects and use a separate environment if those projects install packages with the same names.

## Train

```bash
python blower_arm_rl/scripts/rsl_rl/train.py --task Isaac-Blower-Piper-GripperAssistBoxPush-v0 --headless
python blower_arm_rl/scripts/rsl_rl/train.py --task Isaac-Blower-Piper-GripperAssistCuboidOrientation-v0 --headless
```

## Evaluate a trained checkpoint

```bash
python blower_arm_rl/scripts/rsl_rl/play.py --task Isaac-Blower-Piper-GripperAssistBoxPush-Play-v0 --num_envs 1 --checkpoint /path/to/model.pt
python blower_arm_rl/scripts/rsl_rl/play.py --task Isaac-Blower-Piper-GripperAssistCuboidOrientation-Play-v0 --num_envs 1 --checkpoint /path/to/model.pt
```

No pretrained checkpoints, real-robot deployment code or experiment logs are included.

## Checks

```bash
python external_blower/blower/examples/check_airflow_model.py
python -m pytest blower_arm_rl/test -q
```

## Website

```bash
python -m http.server 8000 --directory docs
```

Visit http://localhost:8000. On GitHub Pages, publish the `docs` folder of `main`.

## Third-party notices

Existing third-party copyright and SPDX notices are preserved. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This release does not assign a new blanket license to third-party code or assets.
