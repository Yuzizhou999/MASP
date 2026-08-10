# Command index

Tools are named by operation and responsibility.

## Build and validation

| Command | Purpose |
|---|---|
| `build_map_model.py` | Convert a source SMAP into a single-fleet map model |
| `build_unified_map_model.py` | Merge fork and jack topology |
| `build_conflict_resources.py` | Generate geometric conflict resources |
| `build_runtime_assets.py` | Generate workstations and runtime metadata |
| `build_repository.py` | Rebuild and validate the main runtime assets |
| `validate_repository.py` | Validate schemas and cross-file invariants |

## Simulation and evaluation

| Command | Purpose |
|---|---|
| `simulate_explicit_plan.py` | Deterministically replay a supplied plan |
| `simulate_planning.py` | Assign tasks, plan routes and simulate execution |
| `simulate_dispatch.py` | Run rolling-horizon coordination and benchmarks |
| `simulate_online_dispatch.py` | Submit tasks online and install acknowledged plans |
| `validate_recovery.py` | Run deadlock and reverse-recovery acceptance cases |

## Training and visualization

| Command | Purpose |
|---|---|
| `train_priority_policy.py` | Capture rolling conflict states and train PPO |
| `build_dispatch_dashboard.py` | Build a standalone replay from a run directory |
| `build_unified_scene_model.py` | Build compact geometry for visualization |

All commands are run from the repository root. Experiment output belongs in
`runs/` or `tmp/`; both directories are ignored by Git.
