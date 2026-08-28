# v1.4 Identifiability Pilot Execution Report

**Protocol:** `RESEARCH_SPECIFICATION.md` version 1.4
**Environment:** `robocasa/PickPlaceCounterToCabinet`, `split=target`
**Policy:** frozen GR00T N1.5 RoboCasa `atomic_seen_30p`
**Scientific role:** exploratory Phase 0 gate; not superiority or recovery evidence

## Executed runs

The initial three-seed debug is retained at
`/home/pc/VLA/outputs/identifiability_pilot_v1_4_debug_3seed/`. After the action
chunk audit correction, the final three-seed debug completed 9 episodes and
passed its artifact checks:

```text
/home/pc/VLA/outputs/identifiability_pilot_v1_4_debug_chunk_audited/
```

The first full pilot completed 36 episodes in a condition-shuffled order and is
retained as historical provenance:

```text
/home/pc/VLA/outputs/identifiability_pilot_v1_4/
```

After correcting the monitor stream to expose the complete requested GR00T
action chunk and restricting delay scoring to the declared exposure window, a
fresh immutable 36-episode run was completed:

```text
/home/pc/VLA/outputs/identifiability_pilot_v1_4_chunk_audited/
```

The source-preserving audit analysis for the final run is:

```text
/home/pc/VLA/outputs/identifiability_pilot_v1_4_chunk_audited_analysis_v2.json
```

No policy parameter was trained or modified. The model was in evaluation mode,
all parameters had `requires_grad=False`, and the parameter SHA-256 before and
after the full run was identical:

```text
facb9d875a6e590e429bc2724b31d6af9f6346b36db825898acbe4ef3a364a08
```

## Population and contracts

| Item | Result |
|---|---:|
| Total episodes | 36 |
| Clean / actuator / observation | 12 / 12 / 12 |
| Seeds per condition | 500--511 |
| Exposed actuator episodes | 8 |
| Exposed observation episodes | 8 |
| Not-exposed attrition per fault mechanism | 4 |
| Monitor records | 16,565 |
| Forbidden top-level monitor leaks | 0 |
| Complete requested 16-step chunks | 100% of monitor records |
| Artifact validation | passed |
| Finite, in-range actions | passed |

The four not-exposed episodes per mechanism completed successfully before the
shared onset at step 240. They were retained as attrition and were not silently
replaced.

## Exploratory findings

The fixed, untrained transparent rule used a pre-collection threshold: predict
observation fault when maximum observation evidence in the strict exposure
window is at least `0.3`, otherwise predict actuator fault.

| Metric | Result |
|---|---:|
| Balanced accuracy | 0.7500 |
| 95% scene-seed cluster-bootstrap interval | [0.5625, 0.9375] |
| Macro-F1 | 0.7500 |
| 95% scene-seed cluster-bootstrap interval | [0.5608, 0.9373] |
| Actuator exposure-window detection | 1/8 |
| Observation exposure-window detection | 6/8 |
| Clean episodes with any alarm | 11/12 |
| Clean alarm events per 1,000 steps | 41.12 |

The mechanism confusion counts were:

| Actual | Predicted actuator | Predicted observation |
|---|---:|---:|
| Actuator | 6 | 2 |
| Observation | 2 | 6 |

Every condition had 7/12 task success. For all 12 scene seeds, clean, actuator,
and observation conditions produced the same binary success outcome. This pilot
therefore shows no observed task-success effect from these short faults, though
it was not powered as a recovery comparison.

## Gate decision

The result indicates a weak passive mechanism signal, but it does not establish
the proposed identifiability claim:

- the pilot rule is not a held-out learned monitor despite exceeding the
  planning reference in this small exploratory sample;
- the transparent rule false-alarms on 91.7% of clean episodes;
- it is a single-score diagnostic rather than a held-out fault-conditioned monitor;
- no predeclared diagnostic-probe comparison was run;
- no calibrated clean operating point was evaluated.

The Phase 0 data-collection and engineering gates passed. The scientific
identifiability gate is **not passed**, so confirmatory active-recovery
experiments remain blocked. The next experiment must freeze a diagnostic probe
and monitor train/calibration/held-out split, then test the fault-conditioned
monitor at a controlled clean operating point.

## Reproduction commands

```bash
source /home/pc/VLA/env.sh
conda activate /home/pc/VLA/envs/robocasa
cd /home/pc/VLA/recovery-bench

python scripts/analyze_identifiability_pilot.py \
  --source /home/pc/VLA/outputs/identifiability_pilot_v1_4_chunk_audited \
  --output /home/pc/VLA/outputs/identifiability_pilot_v1_4_chunk_audited_analysis_v2.json
```

The analysis output is write-once. Re-running it requires a new output path;
the immutable source directory must not be modified.
