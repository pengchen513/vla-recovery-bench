# Monitor Relock v1.3

This relock addresses the blocked v1.2 diagnostic-probe lock without changing
the monitor feature schema, policy contract, fault timing, or the strict clean
episode false-alarm budget.

## Why v1.2 was blocked

The v1.2 fixed monitor checkpoint produced three clean risk-alarm episodes in a
50-episode calibration set.  The registered five-percent episode budget allows
at most two episodes, so selecting an entropy threshold could not make the
joint risk-or-entropy trigger valid.  The failed artifact remains at
`/home/pc/VLA/outputs/diagnostic_probe_v1_2_lock` and is never overwritten.

## Registered v1.3 sources

The monitor architecture is unchanged and is retrained only from the original
v1.0 train shards.  Its risk threshold is calibrated only from clean episodes
in the v1.2 calibration shards.  The validation holdout is fresh and has never
been used for development: scene seeds 1150 through 1199.

| Partition | Protocol | Scene seeds | Use |
| --- | --- | --- | --- |
| train | `configs/monitor_training_v1_0.json` | 600-699 | fit monitor weights |
| calibration | `configs/monitor_relock_v1_2.json` | 1000-1049 | fit clean threshold and entropy lock |
| validation | `configs/monitor_relock_v1_3.json` (`self`) | 1150-1199 | one-time holdout gate |

The source protocol hashes are recorded in
`configs/monitor_relock_v1_3.json` and rechecked by
`validate_mixed_source_shard_set`.  Every source must have the same immutable
environment, information boundary, feature, fault, and storage contract;
cross-partition scene-seed overlap blocks training or locking.

## Fixed training settings

The v1.3 manifest fixes 80 epochs, learning rate `0.08`, L2 `1e-4`, and random
seed `14042026`.  Validation is never used to select the threshold.  The policy
checkpoint remains frozen and no policy parameters enter the monitor.

## Collection

After the precollection commit, collect the five validation shards with
`scripts/collect_monitor_dataset.py`, using the v1.3 protocol and seed blocks
1150-1159, 1160-1169, 1170-1179, 1180-1189, and 1190-1199.  Each output path is
write-once.  Run `scripts/validate_monitor_shards.py` over all five paths and
require exact 50-seed/150-episode coverage before training.

## Training and lock

Train into the new write-once directory
`/home/pc/VLA/outputs/monitor_v1_3_formal_model`, then lock into
`/home/pc/VLA/outputs/diagnostic_probe_v1_3_lock`.  Both commands record all
source protocol paths and hashes.  A lock is issued only when the fresh
validation joint trigger is at most 5%; otherwise the failed artifact is
preserved and the diagnostic probe remains disabled.

No final-test data, recovery intervention, VLA checkpoint download, or model
training is part of this relock.
