# Blower — Reusable Airflow Component Library

An Isaac Lab extension providing reusable airflow physics components for downstream RL environments. This package implements a batched PyTorch airflow force model, an Isaac Lab `ActionTerm` that applies jet forces to rigid-body targets, configurable target object presets, and optional debug visualisers.

This package does **not** register any Gym/Isaac Lab task environments. It is designed to be imported by downstream projects that define their own observations, rewards, terminations, and RL wrappers.

![External Blower package architecture](../../fig/External_package.jpeg)

## What This Package Provides

| Module | Contents |
|--------|----------|
| `blower.models` | Batched PyTorch airflow force model (`AirflowModel`) |
| `blower.mdp` | Isaac Lab `AirflowActionTerm` and `AirflowActionTermCfg` |
| `blower.objects` | Rigid-body target configs: ball, box, cuboid, bottle, cigarette |
| `blower.assets` | Optional kinematic blower proxy asset config |
| `blower.configs` | Jet, drag, sampling, and visualisation dataclasses |
| `blower.viz` | Debug visualisers: jet cone, laser ray, sample markers, reaction-force arrow |

## Installation

Use the Python interpreter that has Isaac Lab installed:

```bash
python -m pip install -e source/blower
```

If Isaac Lab is managed via `isaaclab.sh`:

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e source/blower
```

## Verification

### Numerical smoke test (no Isaac Sim required)

Verifies the airflow velocity model, surface sampling, body-to-world sample transforms, and force/torque output shapes using plain PyTorch:

```bash
python examples/check_airflow_model.py
```


## Target Object Configurations

All object configs are defined in `blower.objects` and implement a common interface:

| Method / Property | Description |
|---|---|
| `.target_type` | String identifier used by `AirflowActionTermCfg` (`"ball"`, `"box"`, `"cuboid"`, `"bottle"`, `"cigarette"`) |
| `.to_rigid_object_cfg(prim_path)` | Returns an Isaac Lab `RigidObjectCfg` ready to be placed in a scene |
| `.build_body_samples(sampling_cfg)` | Returns pre-computed surface sample tensors in the object body frame |

### BallCfg

A solid sphere spawned as a procedural Isaac Lab primitive.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius` | `0.025` m | Sphere radius |
| `mass` | `0.05` kg | Rigid-body mass |
| `color` | `(1.0, 0.2, 0.2)` | RGB diffuse colour |
| `spawn_z` | `0.03` m | Initial spawn height above ground |

**Surface sampling:** Fibonacci (golden-ratio) lattice over the sphere surface. Sample count is determined by dividing total surface area ($4\pi r^2$) by `SamplingCfg.target_area`. All sample normals point radially outward from the centre.

```python
from blower.objects import BallCfg

ball = BallCfg(radius=0.03, mass=0.08)
scene_cfg.target = ball.to_rigid_object_cfg(prim_path="{ENV_REGEX_NS}/Target")
```

---

### BoxCfg

An axis-aligned cuboid spawned as a procedural Isaac Lab primitive.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `size` | `(0.05, 0.05, 0.05)` m | Box dimensions (x, y, z) |
| `mass` | `0.05` kg | Rigid-body mass |
| `color` | `(1.0, 0.2, 0.2)` | RGB diffuse colour |
| `spawn_z` | `0.03` m | Initial spawn height above ground |

**Surface sampling:** Uniform grid over all six faces. The number of sample points per face is computed from the face area divided by `SamplingCfg.target_area`, then distributed to match the face aspect ratio. Sample normals are the outward face normals (`±x̂`, `±ŷ`, `±ẑ`).

```python
from blower.objects import BoxCfg

box = BoxCfg(size=(0.06, 0.04, 0.05), mass=0.03)
scene_cfg.target = box.to_rigid_object_cfg(prim_path="{ENV_REGEX_NS}/Target")
```

---

### CuboidCfg (7 × 7 × 14 cm)

A `BoxCfg` preset with `size=(0.07, 0.07, 0.14)` m and `target_type="cuboid"`.
The 14 cm side is along local Z. The default center height is `0.075` m,
leaving 5 mm ground clearance. Mass (`0.05` kg) and color inherit the box defaults;
set `mass` to match the physical object.

Spawning, collision, airflow surface primitives, and force integration use the
same implementation as the cube. Each primitive stores its face-cell center,
outward normal, and representative area (`face_area / points_on_face`). With
the unchanged `SamplingCfg.target_area=6e-5` m²:

| Faces | Grid per face | Primitives per face | Area per primitive |
|---|---|---|---|
| ±X, ±Y (7 × 14 cm) | 10 × 17 | 170 | `0.0098 / 170` m² ≈ `5.76471e-5` m² |
| ±Z (7 × 7 cm) | 10 × 9 | 90 | `0.0049 / 90` m² ≈ `5.44444e-5` m² |

There are **860 primitives** in total, whose areas sum to **0.049 m²**.
As with the cube, `target_area` sets the desired resolution; actual areas depend
on integer grid counts so each face's area is preserved exactly.

```python
from blower.objects import CuboidCfg
from blower.configs import SamplingCfg
from blower.mdp import AirflowTargetCfg

cuboid = CuboidCfg(mass=0.05)  # Replace with the measured mass in kg.
scene_cfg.cuboid = cuboid.to_rigid_object_cfg("{ENV_REGEX_NS}/Cuboid")
samples = cuboid.build_body_samples(SamplingCfg())

# Append to AirflowActionTermCfg.targets to receive airflow and sample markers.
cuboid_target = AirflowTargetCfg(
    asset_name="cuboid",
    target_type=cuboid.target_type,
    target_object_cfg=cuboid,
)
```

---

### BottleCfg

A plastic water bottle. By default spawned from a GLB mesh asset (`plastic_water_bottle.glb`) with `convexDecomposition` collision. Its airflow samples are generated directly from the mesh surface when `trimesh` is available, with the old procedural cylinder approximation retained as a fallback or when `use_asset=False`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cylinder_radius` | `0.034` m | Radius of the cylindrical approximation used for surface sampling |
| `cylinder_height` | `0.20` m | Height of the cylindrical approximation |
| `mass` | `0.05` kg | Rigid-body mass |
| `spawn_z` | `0.1068` m | Initial spawn height (accounts for upright bottle CoM) |
| `offset_body` | `(0.021, 0.0, 0.11)` m | Body-frame offset applied to samples (centres the cylinder on the bottle body) |
| `use_asset` | `True` | Use the GLB mesh asset; set `False` for a plain cylinder primitive |
| `asset_filename` | `plastic_water_bottle.glb` | Filename relative to the package `assets/` directory |
| `asset_path` | `None` | Override to an absolute path for a custom bottle mesh |
| `scale` | `(1.0, 1.0, 1.0)` | Uniform scale applied to the mesh |
| `rotate_xyz` | `(0.0, 0.0, 90.0)` deg | Initial Euler rotation of the mesh (spawns upright) |
| `collision_approximation` | `convexDecomposition` | PhysX collision approximation for the mesh |
| `use_mesh_samples` | `True` | Sample the GLB mesh surface for airflow primitives; if disabled or unavailable, use the cylinder approximation |

**Surface sampling:** By default, points are placed approximately evenly on the triangle mesh surface, but the number of points and per-point area use the same representative-area convention as the primitive samplers. For the bottle, that representative area is the cylindrical outer envelope defined by `cylinder_radius` and `cylinder_height`, so mesh details do not artificially increase sample density or total force. If mesh sampling fails, the same cylinder is sampled directly (lateral surface + both caps), then translated by `offset_body` in the body frame.

```python
from blower.objects import BottleCfg

bottle = BottleCfg(mass=0.04)
scene_cfg.bottle = bottle.to_rigid_object_cfg(prim_path="{ENV_REGEX_NS}/Bottle")
```

---

### CigaretteCfg

A thin cylindrical rod. Spawned as a procedural cylinder primitive.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `length` | `0.085` m | Rod length (cylinder height, oriented along Y) |
| `radius` | `0.003` m | Rod cross-section radius |
| `mass` | `0.01` kg | Rigid-body mass |
| `spawn_z` | `0.05` m | Initial spawn height |
| `color` | `(0.8, 0.8, 0.2)` | RGB diffuse colour |

**Surface sampling:** A 1-D line sampler distributes equally-spaced points along the rod axis. Each point carries a **zero normal vector**, which marks it as *omnidirectional* in the force model. An omnidirectional sample receives the full drag pressure regardless of the nozzle direction — this accurately models the all-around exposure of a thin rod to the jet.

```python
from blower.objects import CigaretteCfg

cigarette = CigaretteCfg(length=0.09, mass=0.008)
scene_cfg.target = cigarette.to_rigid_object_cfg(prim_path="{ENV_REGEX_NS}/Target")
```

---

## Surface Sampling in Detail

The airflow force model integrates pressure over the exposed surface of each target object. Before training, each object's surface is discretised into a finite set of sample points stored in the object body frame. These samples are transformed to world coordinates each simulation step and passed to the force model.

A sample batch is a dictionary with three tensors:

| Key | Shape | Description |
|-----|-------|-------------|
| `positions_body` | `(N, 3)` | Sample point coordinates in the object body frame |
| `normals_body` | `(N, 3)` | Outward surface normal at each point; zero vector → omnidirectional |
| `areas` | `(N,)` | Area element represented by each point (m²) |

### Sampler summary

| Object | Sampler | Normal type |
|--------|---------|-------------|
| Box | Uniform grid on 6 faces | Directional (`±x̂`, `±ŷ`, `±ẑ`) |
| Ball | Fibonacci lattice on sphere surface | Directional (radial outward) |
| Bottle | GLB mesh surface sampling; cylinder fallback | Directional (mesh face normals; radial + axial for fallback) |
| Cigarette | 1-D line along rod axis | Omnidirectional (zero vector) |

### Controlling sample density — `SamplingCfg`

All samplers accept a `SamplingCfg` instance. Its primary parameter is `target_area`, which sets the desired surface area per sample point and thus controls the total sample count.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_area` | `6×10⁻⁵` m² | Target area per sample point. Smaller values → more samples → higher accuracy, higher compute cost |
| `min_n` / `max_n` | `2` / `64` | Minimum and maximum sample count per face edge (box sampler) |
| `min_ntheta` / `max_ntheta` | `8` / `128` | Angular sample count bounds for cylinder and sphere samplers |
| `min_nz` / `max_nz` | `2` / `128` | Axial sample count bounds for the cylinder sampler |

```python
from blower.configs.airflow_cfg import SamplingCfg

fine_sampling = SamplingCfg(target_area=1e-5)   # ~6× more samples than default
coarse_sampling = SamplingCfg(target_area=5e-4)  # fewer samples, faster
```

---

## Airflow Physics Configuration

### `JetCfg` — Nozzle and Velocity Field

Controls the shape and intensity of the free-jet velocity field.

| Parameter | Default | Physical meaning |
|-----------|---------|-----------------|
| `U0` | `130.0` m/s | Peak exit velocity at the nozzle orifice |
| `B` | `5.9` | Axial decay constant (empirical round-jet value ≈ 5.7–6.2) |
| `nozzle_radius` | `0.01` m | Nozzle exit radius $r_0$; sets the initial jet width |
| `spread_k` | `0.08` | Linear spreading coefficient $k_s$; the Gaussian half-width grows as $\sigma_r = r_0 + k_s x$ |
| `eps` | `0.01` m | Denominator offset preventing singularity at $x = 0$ |
| `axial_model` | `"inverse"` | Centerline velocity model: `"inverse"` for the original capped inverse-distance model, or `"exponential"` for the piecewise plateau/decay model |
| `xc` | `0.1` m | Plateau length for the `"exponential"` model |
| `decay_k` | `1.0` 1/m | Exponential decay rate after `xc` for the `"exponential"` model |

The velocity at axial distance $x$ and radial offset $r$ is:

$$U(x, r) = \min\!\left(U_0,\; \frac{B \cdot U_0 \cdot 2r_0}{x + \varepsilon}\right) \cdot \exp\!\left(-\frac{r^2}{2\sigma_r^2}\right), \quad \sigma_r = r_0 + k_s x$$

The $\min(\cdot)$ term caps the velocity at $U_0$ in the near-field (where the far-field decay formula would otherwise exceed the exit velocity).

Set `axial_model="exponential"` to use the alternative centerline profile:

$$U_c(x)=\begin{cases}U_0,&x\le x_c\\U_0e^{-k(x-x_c)},&x>x_c\end{cases}$$

The radial profile remains Gaussian for both models:

$$U(x,r)=U_c(x)\exp\!\left(-\frac{r^2}{2\sigma_r^2}\right)$$

```python
from blower.configs.airflow_cfg import JetCfg

jet = JetCfg(U0=80.0, nozzle_radius=0.015, spread_k=0.10)
exp_jet = JetCfg(axial_model="exponential", U0=80.0, xc=0.20, decay_k=3.0)
```

### `DragCfg` — Aerodynamic Drag

| Parameter | Default | Physical meaning |
|-----------|---------|-----------------|
| `rho` | `1.225` kg/m³ | Air density at sea level and 15 °C |
| `C_D` | `1.0` | Effective drag coefficient (lumps shape and skin-friction effects) |

The dynamic pressure applied to each surface sample is:

$$q = \frac{1}{2} \rho C_D U^2$$

Increasing `C_D` linearly scales all forces without changing the spatial force distribution.

```python
from blower.configs.airflow_cfg import DragCfg

drag = DragCfg(rho=1.2, C_D=1.2)
```

### `AirflowModelCfg` — Integration Mode and Filters

Passed to `AirflowModel` and `AirflowActionTermCfg`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"surface"` | Force integration mode: `"simple"` (CoM-based) or `"surface"` (per-sample integration) |
| `min_axial_distance` | `0.0` m | Samples with axial distance $x < $ this value are ignored (useful to suppress back-facing application) |
| `velocity_cutoff` | `0.5` m/s | Objects whose centre-of-mass jet velocity falls below this threshold receive no force (prevents spurious forces at long range) |
| `nozzle_local_axis` | `"x"` | Local axis of the nozzle body that points along the jet direction (`"x"`, `"y"`, `"z"`, or negated variants) |

**Mode comparison:**

| Mode | Force | Torque | Computation | Best used for |
|------|-------|--------|-------------|--------------|
| `"simple"` | Jet at CoM, projected area | None | Fastest | Large objects, debugging, initial prototyping |
| `"surface"` | Integrated over all sample points | Yes (moment arms) | Moderate | Realistic tumbling, thin or elongated objects |

---

## Visualisation Configuration — `VizCfg`

`VizCfg` controls all optional debug overlays rendered during training or play. Pass it to `AirflowActionTermCfg` via the `viz_cfg` field.

### General

| Parameter | Default | Description |
|-----------|---------|-------------|
| `show_jet_cone` | `True` | Render a translucent cone representing the jet velocity field |
| `show_sample_markers` | `True` | Render small spheres at each surface sample point |
| `show_laser_ray` | `True` | Render a line from the nozzle tip along the jet axis |
| `show_reaction_force_arrow` | `False` | Render an arrow indicating blower recoil force |
| `max_visualized_envs` | `16` | Maximum number of environments for which overlays are drawn |
| `visualized_env_ids` | `()` | Specific environment indices to visualise; empty means first `max_visualized_envs` |
| `debug_prim_root` | `"/World/Debug/Airflow"` | Root USD path for all debug prims; override to avoid conflicts when multiple blowers are active |

### Jet Cone

The cone represents the Gaussian velocity profile: wider and more transparent at long range, brighter and narrower near the nozzle.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `jet_cone_render_mode` | `"wireframe"` | `"mesh"` uses airflow_project translucent shell bands; `"wireframe"` uses lightweight debug lines |
| `jet_cone_num_slices` | `24` | Number of axial cross-section rings along the cone |
| `jet_cone_num_segments` | `48` | Number of angular segments per ring |
| `jet_cone_max_distance` | `5.0` m | Maximum cone length |
| `jet_cone_velocity_threshold` | `0.5` m/s | Visual cone boundary speed threshold |
| `jet_cone_sigma_scale` | `1.0` | Legacy fallback multiplier on spreading width when `jet_cone_velocity_threshold <= 0` |
| `jet_cone_min_alpha` | `0.015` | Minimum opacity (outermost / farthest regions) |
| `jet_cone_max_alpha` | `0.12` | Maximum opacity (innermost / nearest region) |
| `jet_cone_alpha_gamma` | `0.55` | Gamma correction applied to the alpha ramp |
| `jet_cone_emissive_strength` | `0.45` | Emissive brightness multiplier |
| `jet_cone_start_distance` | `0.01` m | Distance from nozzle at which the cone begins |
| `jet_cone_close_caps` | `False` | Whether to close the cone with end-caps |
| `jet_cone_update_interval` | `4` | Update cone geometry every N simulation steps (reduces GPU overhead) |
| `jet_cone_line_thickness` | `2.0` | Line thickness for wireframe cone rendering |
| `jet_cone_use_flow_intensity` | `False` | Scale cone size by the current policy action magnitude |

### Sample Markers

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_marker_radius` | `0.002` m | Radius of each sample point sphere |
| `sample_marker_color` | `(0.2, 0.8, 1.0)` | RGB colour of the sample markers |

### Laser Ray

| Parameter | Default | Description |
|-----------|---------|-------------|
| `laser_ray_length` | `1.5` m | Length of the ray line segment drawn from the nozzle |
| `laser_ray_radius` | `0.005` m | Tube radius of the ray line |
| `laser_ray_color` | `(0.0, 0.9, 1.0)` | RGB colour of the ray |
| `laser_point_radius` | `0.015` m | Radius of the hit-point indicator sphere |
| `laser_point_color` | `(1.0, 0.1, 0.02)` | RGB colour of the hit-point sphere |
| `laser_use_hit_plane` | `False` | Extend the ray to intersect a horizontal plane at `laser_hit_plane_z` |
| `laser_hit_plane_z` | `0.0` m | Z-coordinate of the hit plane |
| `laser_point_surface_offset` | `0.0` m | Additional offset applied to the hit-point indicator along the normal |

### Reaction Force Arrow

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reaction_force_arrow_length_per_newton` | `0.05` m/N | Arrow length scale |
| `reaction_force_arrow_min_force` | `1×10⁻⁴` N | Force magnitude below which the arrow is hidden |
| `reaction_force_arrow_min_length` | `0.02` m | Minimum rendered arrow length |
| `reaction_force_arrow_max_length` | `0.6` m | Maximum rendered arrow length |
| `reaction_force_arrow_shaft_radius` | `0.008` m | Shaft radius |
| `reaction_force_arrow_head_radius` | `0.025` m | Arrowhead cone radius |
| `reaction_force_arrow_head_length` | `0.06` m | Arrowhead cone length |
| `reaction_force_arrow_color` | `(1.0, 0.55, 0.05)` | RGB colour (orange) |

---

## Package Structure

```
external_blower/blower/
├── examples/
│   ├── check_airflow_model.py      # Pure-PyTorch numerical verification
│
├── source/blower/blower/
│   ├── models/
│   │   ├── airflow_physics.py      # Core jet velocity field and force integration
│   │   └── sampling.py             # Surface sampler (box, ball, mesh, cylinder fallback, line)
│   ├── mdp/
│   │   └── actions.py              # AirflowActionTerm / AirflowActionTermCfg
│   ├── objects/
│   │   └── object_cfg.py           # BallCfg, BoxCfg, BottleCfg, CigaretteCfg
│   ├── assets/
│   │   └── blower_cfg.py           # Kinematic blower proxy asset config
│   ├── configs/
│   │   └── airflow_cfg.py          # JetCfg, DragCfg, SamplingCfg, VizCfg
│   └── viz/
│       ├── jet_cone.py             # Airflow cone visualiser
│       ├── laser_ray.py            # Nozzle laser-ray debug line
│       ├── sample_markers.py       # Surface sample point markers
│       └── reaction_force_arrow.py # Blower recoil force arrow
│
└── README.md
```

---

## Downstream Integration

### Minimal environment wiring

```python
from blower.mdp import AirflowActionTermCfg
from blower.objects import BallCfg
from blower.configs.airflow_cfg import JetCfg, DragCfg, SamplingCfg, VizCfg

target_cfg = BallCfg(radius=0.03, mass=0.06)

class MySceneCfg(InteractiveSceneCfg):
    robot  = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    target = target_cfg.to_rigid_object_cfg(prim_path="{ENV_REGEX_NS}/Target")

class ActionsCfg:
    airflow = AirflowActionTermCfg(
        asset_name="target",
        source_type="articulation_body",
        source_asset_name="robot",
        source_body_name="ee_link",
        nozzle_offset_pos=(0.12, 0.0, 0.0),
        nozzle_offset_quat=(1.0, 0.0, 0.0, 0.0),
        nozzle_local_axis="x",
        target_type=target_cfg.target_type,
        target_object_cfg=target_cfg,
        mode="surface",
        jet_cfg=JetCfg(U0=100.0, nozzle_radius=0.012),
        drag_cfg=DragCfg(C_D=1.2),
        sampling_cfg=SamplingCfg(target_area=4e-5),
        viz_cfg=VizCfg(show_jet_cone=True, max_visualized_envs=4),
    )
```

### Multiple targets from one nozzle

```python
from blower.mdp import AirflowActionTermCfg, AirflowTargetCfg
from blower.objects import BallCfg, BottleCfg

target_cfg = BallCfg()
bottle_cfg = BottleCfg()

airflow = AirflowActionTermCfg(
    asset_name="target",
    source_type="articulation_body",
    source_asset_name="robot",
    source_body_name="ee_link",
    nozzle_offset_pos=(0.12, 0.0, 0.0),
    nozzle_local_axis="x",
    targets=(
        AirflowTargetCfg(asset_name="target",  target_type="ball",   target_object_cfg=target_cfg),
        AirflowTargetCfg(asset_name="bottle",  target_type="bottle", target_object_cfg=bottle_cfg),
    ),
    mode="surface",
)
```

### Nozzle source types

| `source_type` | Description |
|---|---|
| `"articulation_body"` | Nozzle bound to a body on an articulated robot; transform computed as $T_\text{world}^\text{nozzle} = T_\text{world}^\text{body} \cdot T_\text{body}^\text{nozzle}$ |
| `"rigid_object"` | Nozzle bound to an independent kinematic blower proxy (`SIMPLE_BLOWER_CFG`) |
| `"world_frame"` | Fixed nozzle at a specified world-space pose |

### Reaction force

Set `apply_reaction_force=True` and `reaction_force_max=<newtons>` to apply blower recoil to the source body (opposite the nozzle axis, linear in action magnitude). Set `viz_cfg.show_reaction_force_arrow=True` to render a scaled arrow indicating the recoil direction and magnitude.
