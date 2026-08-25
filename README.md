# VLA Recovery Bench

VLA Recovery Bench is an experiment scaffold for measuring whether a frozen
vision-language-action policy can detect and recover from controlled failures in
long-horizon manipulation.

The repository currently provides:

- deterministic, phase-aware fault schedules;
- a policy/environment/monitor interface that keeps the base policy frozen;
- JSONL event logs with enough information to reproduce every rollout;
- recovery-aware metrics, including precision, recall, detection delay, and
  post-fault task success;
- a dependency-free dummy environment for local development;
- a RoboCasa smoke test and a headless AutoDL setup path.

## Local validation

Python 3.11 or newer is required.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m vla_recovery_bench.cli dummy \
  --config configs/local_dummy.json \
  --output outputs/local_dummy
```

The dummy experiment is not a robotics result. It verifies scheduling, logging,
fault matching, recovery decisions, and metric aggregation without CUDA or
MuJoCo.

## AutoDL smoke test

Keep all large files on `/root/autodl-tmp`. Follow [docs/AUTODL.md](docs/AUTODL.md),
then run:

```bash
source /root/autodl-tmp/VLA/env.sh
conda activate /root/autodl-tmp/VLA/envs/robocasa
python scripts/smoke_test_robocasa.py \
  --output /root/autodl-tmp/VLA/outputs/smoke_test
```

Success requires a second clean run and a non-blank `first_frame.png`.

## Repository boundary

Model weights, datasets, videos, Conda environments, and experiment outputs are
intentionally excluded from Git. A generic SmolVLA base checkpoint is not assumed
to be compatible with RoboCasa. Policy integration requires a checkpoint whose
observation keys, state vector, action representation, and normalization statistics
match the target RoboCasa embodiment.

See [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) for the current protocol
and [docs/SMOLVLA_INTEGRATION.md](docs/SMOLVLA_INTEGRATION.md) for the policy gate.

