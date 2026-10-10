# Release validation

- 73 existing task behavior checks passed (including fixed and widening goal-range cases).
- Airflow model numerical smoke test passed on CUDA.
- 65 Python files parsed successfully; local import targets resolve in this checkout.
- Exactly two training and two Play task registrations; all configuration entry points exist.
- XML mesh paths and all six USD layers' external references resolve within the release.
- Website local links resolve; the static site has no original repository links or external tracking scripts.
- Known original author/account, institution, repository and home-directory strings were absent from the released files and decoded USD layers.
- No symlinks, original Git history, experiment logs or pretrained checkpoints are included.

The existing goal-range test assumed an expanding curriculum although the released configuration uses a fixed range. In this copy, the test explicitly covers both configurations; task training parameters are unchanged.

Full Isaac Sim environment startup, rollout and training have not been validated for this isolated release. Static import checks and CPU task tests do not substitute for a simulator run. Numerical checks used an existing compatible Isaac Lab environment without installing this copy into that environment.

Third-party asset creator names remain as attribution. See THIRD_PARTY_NOTICES.md for asset provenance and the outstanding robot-asset license documentation gap. Anonymisation removes direct identifiers checked here; code similarity to other public work can still reveal provenance.
