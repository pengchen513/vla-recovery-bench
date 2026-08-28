# Response to the Research Specification Review

**Reviewed document:** `RESEARCH_SPECIFICATION.md` version 1.0
**Response version:** 1.4
**Date:** 2026-08-26
**Disposition:** accepted recommendations are incorporated in specification
version 1.4; deferred items remain explicit gates and are not represented as
completed capabilities.

## Executive decision

The review is substantively correct. The previous specification was strong as
an engineering and provenance document, but it made four scientific mistakes:

1. it treated mechanism, exposure, uncertainty, and recoverability as one
   latent class;
2. it used counterfactual language for matched seeds without specifying a
   randomized branch design;
3. it treated a 30-episode run as if it could support confirmatory calibration,
   superiority, and transfer claims;
4. it specified active interventions and a privileged-information firewall that
   the original runner did not implement.

The revised plan therefore narrows the primary claim to a single-fault,
non-privileged mechanism-diagnosis and cost-constrained recovery study. It
separates mechanism labels from online exposure and hindsight outcome labels,
removes online `irrecoverable` classification, excludes reward from the primary
track, replaces the single 5% event budget with a declared cost vector, and
makes implementation of the firewall and intervention protocol a hard gate.

The review's conclusion that the project is not yet a top-tier paper is also
accepted. No document change can substitute for the missing pilot, ablations,
sample-size evidence, and independent evaluation.

Version 1.4 records an additional implementation fact: naming a firewall in the
specification does not mean that the runtime automatically enforces it. The
repository now has structural in-process checks for privileged `info`/reward,
typed intervention/chunk metadata, a separate audit record, and refusal to
overwrite non-empty JSONL artifacts. The scientific gate remains closed until
the full artifact manifest, randomized branch protocol, and identifiability
pilot pass. The versioned power analysis is complete as planning evidence at
`/home/pc/VLA/outputs/power_analysis_v1_4.json`; it does not unlock fault
execution or constitute a scientific result.

## v1.4 audit amendment

A follow-up audit found that the checked-in v1.3 pilot was not merely sparse:
its assignment encoded mechanism through the experimental factors themselves
(all actuator rows used onset 80, while all observation rows used onset 240;
the variant distributions also differed). A classifier could therefore obtain
apparently strong mechanism scores without learning a mechanism signature. This
is a design invalidation, not a result. The v1.3 file remains untouched as
historical provenance and is blocked from collection.

The adopted v1.4 correction is a crossed 12-scene-seed pilot. Each seed produces
one clean, one actuator, and one observation episode; the two fault episodes
reuse the same onset, duration, camera, and scene-factor row. Factor counts are
balanced across rows, and episode IDs are distinct from the scene-seed pair ID.
The primary recovery estimand is also the sole confirmatory primary endpoint;
mechanism macro-F1 is a key secondary identifiability gate. This avoids both
mechanism confounding and unplanned co-primary multiplicity.

The review's proposal to use a full factorial was not adopted. A crossed,
balanced pilot followed by a pre-registered fractional-factorial confirmatory
matrix gives factor balance without pretending that sparse pilot cells support
all interaction claims.

The audit also fixed a timing-contract ambiguity. Observation corruption is
applied by the current adapter after the environment step, so its first altered
policy input is one control step after the injection call. v1.4 records the
injection step and first-affected-input step separately; detection delay is
measured from the latter. This prevents an implementation detail from being
silently counted as early warning.

## Disposition by finding

| Review item | Decision | Change in version 1.4 |
|---|---|---|
| F1: latent mode is not identifiable | **Accept** | Split episode mechanism `M`, exposure `E_t`, potential outcome `Y(u)`, and online risk `Q_t`; remove `policy/OOD` and `irrecoverable` from the primary mechanism class. |
| F2: recovery comparison is not causal | **Accept with qualification** | Define intention-to-treat estimands and randomized common-prefix decision branches as the preferred design. Matched seeds without branching are labeled descriptive only. |
| F3: 30 episodes are underpowered | **Accept** | Keep the existing 30 episodes as historical provenance; add a primary endpoint, a power-analysis requirement, independent seeds, hierarchical paired analysis, and a confirmatory minimum sample plan. |
| F4: one binary intervention budget is inadequate | **Accept** | Record time, compute, human, and risk costs as a vector; report per-episode and per-step rates plus a pre-registered scalar sensitivity analysis. |
| F5: matched seeds are not automatically counterfactuals | **Accept** | Replace unqualified `counterfactual` wording with matched-seed/common-random-number comparison; reserve causal wording for randomized branches. |
| M1: novelty is still a component combination | **Accept with reframing** | Make mechanism-conditioned prediction and active diagnosis the falsifiable contribution; treat uncertainty, latent prediction, and calibration as non-novel ingredients. |
| M2: all-zero occlusion creates a shortcut | **Accept** | Add a corruption ladder: trivial diagnostic, hard synthetic, and naturalistic/stale/correlated corruption; final mechanism claim uses the hard set. |
| M3: zeroing all actuator channels is too narrow/incorrect | **Accept** | Define channel-specific, hold-last, intermittent, and noise variants; do not call the current all-field zero transform the canonical physical fault without validation. |
| M4: reward timing leaks information | **Accept** | Exclude reward from the primary monitor input; retain reward-enabled monitoring only as a timing-audited ablation. |
| M5: `irrecoverable` lacks a label | **Accept** | Remove it from online diagnosis; report recoverability only as a protocol-relative potential outcome/hindsight analysis. |
| M6: intervention enum/context does not match code | **Accept as implementation gate** | Add a pre-science gate requiring a typed intervention enum, chunk metadata, and explicit context firewall before any active-recovery result. |
| M7: runner/artifacts do not enforce the contract | **Accept as implementation gate** | Require a monitor-safe context, offline audit channel, immutable output handling, and complete artifact set before pilot results are scored. |
| M8: one task/checkpoint is not externally valid | **Accept** | Introduce claim tiers: GR00T/RoboCasa case study first; generalization claims require a second task and preferably a second frozen policy. |

## Recommendations deliberately not adopted verbatim

### “Use `irrecoverable` as a separate online state”

Not adopted. The review correctly identifies that irrecoverability is a future
reachability property, not an ordinary sensor-level mechanism. Without a formal
reachable-set or intervention-relative label, exposing it online would create a
new form of hindsight leakage. It is retained only as an offline analysis of
`Y(u)` under the declared protocol.

### “Use a single weighted cost number immediately”

Modified. A scalar cost with arbitrary weights can hide important trade-offs.
The revised protocol records a cost vector first and reports a Pareto frontier.
A scalar constrained analysis is permitted only after weights and sensitivity
ranges are frozen before final evaluation.

### “Require two policies before any useful experiment”

Modified. The current official checkpoint and clean integration are sufficient
for a controlled GR00T/RoboCasa case study and for the identifiability pilot.
They are not sufficient for policy-agnostic claims. A second policy is a
generalization gate, not a prerequisite for every engineering milestone.

### “Run a full factorial over every factor”

Deferred and replaced by a staged design. A full factorial would create sparse
cells and waste simulation budget. The revised plan uses a pilot, a balanced
fractional-factorial confirmatory matrix, and held-out factor transfer tests.

## Resulting priority order

1. Implement the monitor-safe information firewall and typed intervention
   protocol; current runner outputs must not be treated as compliant yet.
2. Run the Phase 0 identifiability pilot with hard observation corruptions and
   validated actuator variants.
3. Decide from the pilot whether passive mechanism diagnosis is identifiable or
   whether a diagnostic probe is necessary.
4. Freeze one primary diagnosis endpoint and one recovery endpoint, then run an
   explicit power calculation before collecting confirmatory data.
5. Build the black-box predictive monitor and calibration split.
6. Evaluate active recovery using randomized common-prefix branches when the
   implementation supports them; otherwise report a weaker matched-seed study.
7. Add a second task and second frozen policy only before making transfer,
   policy-agnostic, or general RoboCasa claims.

The immediate engineering sequence is therefore: firewall and immutable
artifact tests, a three-seed non-scored debug run, the predeclared
identifiability pilot, and a simulation-based power analysis. Detector training
and active recovery remain disabled until these gates pass.

## What the review changes scientifically

The main claim is no longer “a general failure detector with many uncertainty
signals.” It is:

> Under a single declared fault and a non-privileged information boundary, a
> mechanism-conditioned action predictor plus an explicitly costed diagnostic
> intervention can improve recovery utility over fixed retry at a controlled
> clean-intervention cost.

This claim can fail. If hard corruptions are not distinguishable, the paper must
report non-identifiability. If active diagnosis does not beat fixed retry at
matched cost, the paper must drop the active-recovery claim. If the evaluation
remains one task and one policy, the title and abstract must call it a
GR00T/RoboCasa case study.
