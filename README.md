# MASP

MASP is a deterministic multi-vehicle warehouse dispatch reference implementation.
It combines task assignment, rolling-horizon priority coordination, continuous-time
SIPP planning, reservation-based safety checks, online plan acknowledgement,
deadlock recovery and an optional RL priority policy.

## Repository layout

| Path | Responsibility |
|---|---|
| `masp/` | Runtime domain modules and scheduling algorithms |
| `tools/` | Executable build, validation, simulation, training and visualization commands |
| `config/` | Runtime configuration and commented examples |
| `schemas/` | JSON Schema contracts for tasks, plans, scenarios and configuration |
| `scenarios/` | Small validation cases, benchmarks and multi-fleet workloads |
| `generated/` | Reproducible map, conflict-resource and workstation assets |
| `visualization/` | Standalone dispatch replay template |
| `tests/` | Unit, integration and end-to-end regression tests |
| `docs/` | Architecture, safety boundaries and implementation history |
| `runs/`, `tmp/` | Local experiment outputs; ignored by Git |

Core modules use domain names rather than implementation-stage names:

- `planning.py`: task assignment and base task planning;
- `coordination.py`: rolling-horizon candidate ordering and local conflict planning;
- `online.py`: task submission, plan proposal, acknowledgement and telemetry runtime;
- `rl_priority.py`: observation encoder, oracle-supervised priority policy, optional PPO fine-tuning and checkpoint loading;
- `sipp.py`, `reservations.py`: collision-free timing and resource ownership;
- `deadlock.py`, `recovery.py`: wait-graph supervision and recovery execution;
- `recovery_scenario.py`: deterministic recovery acceptance scenario.

Online dispatch uses real RH-PP commitment windows: only the complete segment
prefix ending at `safeUntilMs` is acknowledged and reserved. The vehicle-task
binding and completed service phase survive that boundary, while the uncommitted
route tail is discarded and replanned from the safe node in a continuation plan.

## Quick start

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python tools/validate_repository.py
python -m pytest -q
```

Run a quick online dispatch simulation:

```powershell
python tools/simulate_online_dispatch.py `
  scenarios/interactive-multi-fleet.json `
  --policy congestion `
  --output-dir runs/online-dispatch-quick
```

Build its replay dashboard:

```powershell
python tools/build_dispatch_dashboard.py runs/online-dispatch-quick
```

Open `runs/online-dispatch-quick/dispatch-dashboard.html` in a browser.

## Main workflows

| Goal | Command |
|---|---|
| Rebuild runtime assets | `python tools/build_repository.py` |
| Validate repository inputs | `python tools/validate_repository.py` |
| Replay an explicit plan | `python tools/simulate_explicit_plan.py` |
| Plan a continuous task stream | `python tools/simulate_planning.py` |
| Run rolling-horizon dispatch | `python tools/simulate_dispatch.py` |
| Run online dispatch | `python tools/simulate_online_dispatch.py` |
| Validate recovery behavior | `python tools/validate_recovery.py` |
| Train the RL priority policy | `python tools/train_priority_policy.py` |
| Build a replay dashboard | `python tools/build_dispatch_dashboard.py <run-dir>` |

See [scenarios/README.md](scenarios/README.md), [config/README.md](config/README.md),
[tools/README.md](tools/README.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for detailed inputs, safety boundaries and experiment commands.

## Repository rules

- Name source files after domain responsibility, not project milestones.
- Keep executable entry points in `tools/`; core modules must not depend on them.
- Put reproducible inputs in `config/`, `schemas/` or `scenarios/`.
- Do not hand-edit `generated/`; rebuild it with the corresponding tool.
- Write local results to `runs/` or `tmp/`, never beside source files.
- Treat RL as a priority proposal mechanism only. SIPP, reservations and plan
  validation remain mandatory safety boundaries.
