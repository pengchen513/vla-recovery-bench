# Monitor threshold relock v1.4

This relock is a threshold-only operating-point decision for the frozen
`groot_n1_5_robocasa_atomic_seen_30p` monitor checkpoint.  It does not retrain
the VLA or the monitor.

* calibration: clean episodes from scene seeds `1200..1249`;
* validation: clean episodes from scene seeds `1250..1299`;
* threshold rule: the pre-registered risk-or-normalized-entropy union rule;
* clean budget: at most 5% of validation episodes with any alarm;
* pilot seeds `500..511` and all earlier relock artifacts are excluded from
  threshold selection.

The protocol is frozen in
`configs/monitor_relock_v1_4.json`.  The lock command refuses overlapping
calibration/validation seeds and records the complete seed sets, source shard
hashes, and an explicit `pilot_data_used_for_threshold: false` assertion in
`probe_lock.json`.  A lock is not issued when the fresh validation budget is
exceeded; lowering the threshold cap or inspecting pilot outcomes cannot
repair that failure.

Before collecting the shards, commit the protocol and runner.  Collect five
10-seed formal shards for each partition, validate them with the formal gate,
then run `scripts/lock_diagnostic_probe.py` into a new write-once output
directory.  Pilot or final-test outputs must not be passed as either data
argument.
