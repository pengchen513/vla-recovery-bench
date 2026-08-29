# Diagnostic Probe v1.1

This document describes the implementation of the bounded diagnostic-probe
runner. It is an engineering and identifiability artifact, not a recovery
result. The runner never trains a policy, chooses a recovery action, or reads
fault labels in its online monitor channel.

## Frozen contract

- Environment: `robocasa/PickPlaceCounterToCabinet`, `split=target`.
- Protocol: `configs/diagnostic_probe_v1_1.json` (research protocol 1.4).
- Policy: the audited frozen GR00T RoboCasa manifest supplied on the command
  line.
- Monitor: the passed monitor checkpoint supplied on the command line.
- Arms: `passive_only` and `passive_plus_probe`, paired by scene seed and
  condition.
- Probe budget: four environment transitions: repeat the previous requested
  action, force one fresh policy query, continue two actions from that chunk,
  then discard the remainder and force a fresh query before normal operation.
- Online monitor events contain observations, requested actions/chunks, chunk
  metadata, posterior diagnostics, and declared latency only. Rewards, fault
  schedules, executed actions, outcomes, and seeds are written to the separate
  `privileged_audit.jsonl` stream.

## Required order

Load the RoboCasa environment before every command:

```bash
source /home/pc/VLA/env.sh
conda activate /home/pc/VLA/envs/robocasa
cd /home/pc/VLA/recovery-bench
```

The original v1.1 attempt and its failed holdout artifact are historical. For
the reviewed relock, use the independent v1.2 protocol and the commands in
[MONITOR_RELOCK_V1_2.md](MONITOR_RELOCK_V1_2.md). Do not reuse the old
`700-749`/`800-849` shards for a new lock.

For reference, the original command was:

```bash
python scripts/lock_diagnostic_probe.py \
  --calibration-data /home/pc/VLA/outputs/monitor_v1_0_formal_calibration_* \
  --validation-data /home/pc/VLA/outputs/monitor_v1_0_formal_validation_* \
  --output /home/pc/VLA/outputs/diagnostic_probe_v1_1_lock
```

The command is write-once. A successful run creates
`probe_lock.json` with `status=locked`. A failed holdout gate creates only
`probe_lock_failed.json` and a failed `artifact_validation.json`; it never
creates a lock that the runner can consume. Use a new output directory for a
new reviewed calibration attempt.

Only a successful lock permits the non-scored three-seed debug run:

```bash
python scripts/run_diagnostic_probe.py \
  --stage debug \
  --lock /home/pc/VLA/outputs/diagnostic_probe_v1_1_lock/probe_lock.json \
  --output /home/pc/VLA/outputs/diagnostic_probe_v1_1_debug
```

After reviewing the debug artifact, run the 12-seed, 72-episode pilot:

```bash
python scripts/run_diagnostic_probe.py \
  --stage pilot \
  --lock /home/pc/VLA/outputs/diagnostic_probe_v1_1_lock/probe_lock.json \
  --output /home/pc/VLA/outputs/diagnostic_probe_v1_1_pilot
```

Analyze a completed collection in a separate immutable tree:

```bash
python scripts/analyze_diagnostic_probe.py \
  --source /home/pc/VLA/outputs/diagnostic_probe_v1_1_pilot \
  --output /home/pc/VLA/outputs/diagnostic_probe_v1_1_pilot_analysis
```

The analysis reports the paired change in mechanism log loss, a scene-seed
cluster bootstrap interval, mechanism diagnosis metrics, clean joint-trigger
rate, probe cost caps, and attrition. A positive point estimate alone is not a
recovery claim; the pilot gate also requires the clean operating-point and cost
constraints.

The `final` stage is intentionally unavailable until a separately issued
final-lock artifact and an explicit final plan exist. Final-test seeds cannot
be reached through the pilot configuration.

## Current machine state

The current formal monitor and shard integrity gates pass, but the entropy lock
is blocked by the untouched validation holdout: calibration locks a threshold
of `0.999950842110984` (`2/50` joint clean triggers), while validation has
`4/50 = 8%` joint clean triggers against the pre-registered `5%` cap. The
auditable failure is:

`/home/pc/VLA/outputs/diagnostic_probe_v1_1_lock/probe_lock_failed.json`

A second write-once replay after implementation hardening is retained at
`/home/pc/VLA/outputs/diagnostic_probe_v1_1_lock_v2/probe_lock_failed.json`.

Consequently no diagnostic-probe debug or pilot process was started. The
minimal next scientific action is to review and, if justified, recalibrate the
monitor/entropy operating point under a newly versioned protocol; do not lower
or disable the validation gate post hoc.
