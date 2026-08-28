# Scientific readiness gate

The governing protocol is
[`RESEARCH_SPECIFICATION.md`](RESEARCH_SPECIFICATION.md) version 1.4. This
readiness note records infrastructure status only; it is not evidence that the
scientific detector or recovery method has passed.

The RoboCasa installation and interface baseline are prerequisites, not
scientific results. The current target contract is
`robocasa/PickPlaceCounterToCabinet`, `split=target`, with fixed episode seeds.

Completed locally:

- headless EGL and MuJoCo import checks;
- official RoboCasa assets and two non-blank smoke tests;
- recursive observation contract probe;
- structured Gymnasium Dict action validation;
- deterministic ZeroPolicy and RandomPolicy interface baselines;
- strict policy adapter validation of observation keys/shapes and actions;
- clean-baseline runner with per-episode seeds, rewards, steps, action shapes,
  action hashes, and policy latency;
- unit tests and Ruff checks.

The integration gate for the frozen policy, monitor firewall, typed intervention
protocol, hard-fault adapters, and immutable artifact writer have passed their
local checks. The v1.4 three-seed debug and 36-episode pilot have also completed,
but the scientific gate remains closed. The versioned power simulation has been
executed and written to
`/home/pc/VLA/outputs/power_analysis_v1_4.json` (selected `168` independent
units; minimum simulated power `0.8372` across the declared scenarios). A
user-provided frozen policy
checkpoint must have a metadata manifest matching every field in
`docs/SMOLVLA_INTEGRATION.md`. In particular, a generic SmolVLA checkpoint is
not accepted without evidence that its embodiment, three camera streams,
proprioception, 12-dimensional structured action semantics, control mode, and
normalization match RoboCasa. The manifest checker also verifies the local
checkpoint SHA-256 and requires published clean-baseline metadata.

The current 30-episode clean baseline is historical checkpoint provenance. It
must not be used to train or tune a monitor and is not large enough for a
confirmatory superiority or calibration claim.

## Formal monitor-shard integrity gate

The formal data path is now fail-closed. Each `formal_shard` or
`full_partition` collection seals these source artifacts in a write-once
`shard_integrity.json` containing file sizes and SHA256 values. Before monitor
training creates an output directory, the read-only gate verifies:

- the current protocol and policy-manifest SHA256 values;
- full-horizon, non-debug collection from a clean repository snapshot;
- exact, disjoint seed coverage for train `600..699`, calibration `700..749`,
  and validation `800..849`;
- exactly one clean, actuator-fault, and observation-fault episode per seed;
- deterministic episode IDs, pair IDs, factor rows, fault schedules, tokens,
  exposure masks, and contiguous control/observation timestamps;
- agreement among HDF5 channels, dataset index, episode JSONL, audit JSONL,
  metrics, and run manifest;
- finite and in-range requested and executed structured actions;
- unchanged frozen-policy parameter hashes and identical environment/software
  provenance across every shard;
- byte size and SHA256 agreement for every sealed source artifact.

The standalone check is read-only:

```bash
python scripts/validate_monitor_shards.py \
  --partition train \
  --shard /path/to/train-shard-* \
  --json
```

`scripts/train_fault_monitor.py` invokes the same gate for train, calibration,
and validation and refuses to fit a monitor if any partition is incomplete.
Synthetic two-shard coverage, missing/duplicate-seed rejection, checksum-tamper
rejection, re-sealed semantic-tamper rejection, and the real training-entry
fail-closed path pass locally. The existing
single-seed debug directories are intentionally rejected: they use a reduced
horizon, predate `shard_integrity.json`, and do not cover their declared
partitions. No formal monitor shard has been collected yet, so this engineering
gate is implemented but the scientific gate remains closed.

## Phase 0 execution record

The initial non-scored three-seed debug is retained at
`/home/pc/VLA/outputs/identifiability_pilot_v1_4_debug_3seed/`. The final
chunk-audited three-seed debug artifacts are in:

```text
/home/pc/VLA/outputs/identifiability_pilot_v1_4_debug_chunk_audited/
```

The first immutable 36-episode run is retained at
`/home/pc/VLA/outputs/identifiability_pilot_v1_4/`. It predates the complete
action-chunk audit field and is not the final analysis source. The final
chunk-audited pilot artifacts are in:

```text
/home/pc/VLA/outputs/identifiability_pilot_v1_4_chunk_audited/
```

The final run completed all 36 episodes: 12 clean, 12 actuator-fault, and 12
observation-fault episodes over seeds 500--511. Eight episodes from each fault
mechanism reached the declared exposure window; four from each mechanism ended
successfully before onset and remain recorded as `not_exposed` attrition. The
artifact validator passed, the monitor stream contained no forbidden top-level
fields, every monitor record contained the complete requested 16-step action
chunk, all actions were finite and in range, and the frozen GR00T parameter hash
remained
`facb9d875a6e590e429bc2724b31d6af9f6346b36db825898acbe4ef3a364a08`.

The source-preserving analysis is:

```text
/home/pc/VLA/outputs/identifiability_pilot_v1_4_chunk_audited_analysis_v2.json
```

The fixed transparent diagnostic rule produced balanced accuracy `0.7500`
(scene-seed cluster-bootstrap 95% interval `[0.5625, 0.9375]`) and macro-F1
`0.7500` (interval `[0.5608, 0.9373]`) on 16 hard exposed episodes. It raised at
least one alarm in 11 of 12 clean episodes and detected only 1 of 8 actuator
faults within their declared exposure windows, versus 6 of 8 observation
faults. All three conditions had 7/12 task success, and every scene seed had the
same binary success outcome under clean, actuator, and observation conditions.
These are exploratory pilot observations, not superiority or recovery evidence.

The Phase 0 collection gate passed, but the identifiability gate did not. The
transparent rule is an uncalibrated single-score diagnostic, not a held-out
fault-conditioned monitor; its clean alarm behavior is unusable; the completed
source stream now contains the complete future GR00T chunk and strict
exposure-window delay accounting, but no diagnostic-probe arm or calibrated
operating point was evaluated. The initial pre-fix source remains immutable and
is not mixed into the final analysis.

Before any confirmatory fault or recovery result, complete the following gates:

1. freeze a diagnostic-probe protocol and explicit monitor train, calibration,
   validation, and final-test seed splits without reusing pilot labels for final
   evaluation;
2. train and evaluate the fault-conditioned monitor at a controlled held-out
   clean operating point, including full requested action chunks;
3. quantify whether the predeclared diagnostic probe adds information at an
   acceptable component-wise cost;
4. implement and test common-prefix state branching, or freeze the weaker
   episode-randomized intention-to-treat fallback claim;
5. only then collect the powered confirmatory paired/randomized branches.

The in-process runner now omits transition `info` and reward from
`MonitorContext`, keeps executed actions/outcomes in a separate audit record,
and refuses to overwrite non-empty JSONL artifacts. These structural firewall
checks pass locally. The readiness status remains **engineering-ready,
science-blocked**. The pilot artifact manifest is complete, but passive
identifiability at a usable clean operating point, diagnostic-probe value,
calibration, and randomized/common-prefix recovery evaluation remain unresolved.
The generated power result is planning evidence, not a scientific result.

After the manifest passes, a clean integration check may be run with no faults:

```bash
source /home/pc/VLA/env.sh
conda activate /home/pc/VLA/envs/robocasa
python scripts/check_policy_contract.py \
  --manifest /path/to/policy_manifest.json \
  --probe /home/pc/VLA/outputs/robocasa_contract.json
python scripts/run_robocasa_baseline.py \
  --episodes 20 --horizon 1000 --seed 0 \
  --output /home/pc/VLA/outputs/robocasa_clean_baseline
```

`actuator_fault` and `observation_fault` remain enabled only for declared pilot,
monitor-training, and calibration work; confirmatory recovery collection is
still blocked. The all-zero/all-channel-zero conditions are diagnostic checks,
not the full scientific fault model. Object displacement and grasp release
remain blocked until task-specific MuJoCo body semantics are documented in a
new specification version. No policy-agnostic or general RoboCasa claim is
allowed with only one task and one frozen policy.
