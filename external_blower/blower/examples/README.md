# Blower Examples

A numerical verification script for the `blower` extension. Run from the `external_blower/blower/` directory.

## check_airflow_model.py

Verifies the numerical airflow model without launching Isaac Sim. Tests covered:

- Gaussian jet velocity field computation across a range of axial and radial distances.
- Surface sampler output (positions, normals, areas) for box, sphere, cylinder, and line primitives.
- Body-to-world coordinate transforms of sample points.
- Force and torque tensor shapes for both `simple` and `surface` interaction modes.

```bash
python examples/check_airflow_model.py
```

No Omniverse runtime is required; only PyTorch and the installed `blower` package are needed.
