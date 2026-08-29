# Monitor Relock v1.2

This is a versioned operating-point relock for the v1.1 diagnostic probe. It is
not a new monitor architecture and it is not a scientific result. The frozen
GR00T policy, monitor checkpoint, feature schema, information boundary, fault
mechanisms, and `risk OR normalized-posterior-entropy` trigger are unchanged.

## Why a relock exists

The original v1.1 lock used calibration seeds `700-749` and an untouched
validation holdout `800-849`. Its locked union trigger was `4/50 = 8%` on the
validation set, exceeding the pre-registered `5%` clean budget. That lock is
retained as a failed artifact. No threshold is lowered after seeing that
failure. The v1.2 protocol assigns independent clean calibration seeds
`1000-1049` and validation seeds `1100-1149`, then applies the same rule and
the same point gate.

The parent protocol is recorded by path and SHA256 in
`configs/monitor_relock_v1_2.json`. The validator rejects any change outside
the two seed lists and explicit relock metadata.

## Data collection

Each partition contains 50 scene seeds, and each seed is collected under the
three paired conditions `clean`, `actuator_fault`, and `observation_fault`.
Formal shards contain ten seeds (30 episodes) and must be written to new,
empty, write-once output directories. Collect shards sequentially with the
frozen policy server; do not run the probe or any recovery arm during this
phase.

```bash
source /home/pc/VLA/env.sh
conda activate /home/pc/VLA/envs/robocasa
cd /home/pc/VLA/recovery-bench

python scripts/collect_monitor_dataset.py \
  --protocol configs/monitor_relock_v1_2.json \
  --partition calibration --collection-role formal_shard \
  --max-scene-seeds 10 --seed-offset 0 \
  --output /home/pc/VLA/outputs/monitor_relock_v1_2_calibration_1000_1009
```

Repeat with offsets `10, 20, 30, 40`, changing the output suffix to the
corresponding ten-seed range. Repeat the same five commands for validation
seeds `1100-1149`. The formal gate requires a clean repository snapshot,
full 750-step horizon, exact seed coverage, three conditions per seed,
cross-shard disjointness, identical provenance, and SHA256-sealed artifacts.

## Lock and report

After all ten shards pass their aggregate gates, issue a new lock:

```bash
python scripts/lock_diagnostic_probe.py \
  --protocol configs/monitor_relock_v1_2.json \
  --probe configs/diagnostic_probe_v1_1.json \
  --calibration-data /home/pc/VLA/outputs/monitor_relock_v1_2_calibration_1000_1009 /home/pc/VLA/outputs/monitor_relock_v1_2_calibration_1010_1019 /home/pc/VLA/outputs/monitor_relock_v1_2_calibration_1020_1029 /home/pc/VLA/outputs/monitor_relock_v1_2_calibration_1030_1039 /home/pc/VLA/outputs/monitor_relock_v1_2_calibration_1040_1049 \
  --validation-data /home/pc/VLA/outputs/monitor_relock_v1_2_validation_1100_1109 /home/pc/VLA/outputs/monitor_relock_v1_2_validation_1110_1119 /home/pc/VLA/outputs/monitor_relock_v1_2_validation_1120_1129 /home/pc/VLA/outputs/monitor_relock_v1_2_validation_1130_1139 /home/pc/VLA/outputs/monitor_relock_v1_2_validation_1140_1149 \
  --output /home/pc/VLA/outputs/diagnostic_probe_v1_2_lock
```

`probe_lock.json` contains the calibration order statistic, the validation
point estimate, and a two-sided Clopper–Pearson 95% interval. The interval is
report-only; the fail-closed gate is the point estimate
`validation_joint_trigger_rate <= 0.05`. A blocked attempt writes only a
write-once `probe_lock_failed.json` and must not be consumed by the runner.

Only a `status=locked` artifact permits the existing v1.1 debug/pilot runner.
Until that artifact exists, do not enable `actuator_dropout`,
`observation_occlusion`, or any recovery intervention.
