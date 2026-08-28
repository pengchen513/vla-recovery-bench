# Experiment design

This short document is the operational index for the more detailed and
versioned research contract in
[`RESEARCH_SPECIFICATION.md`](RESEARCH_SPECIFICATION.md). If the two documents
disagree, the versioned specification wins; this file must then be updated
before running a new experiment.

The governing specification is version 1.4. This index deliberately separates
engineering checks, an identifiability pilot, and confirmatory science. A
successful smoke test, clean baseline, or pilot does not by itself establish a
recovery improvement.

## Current scientific estimand

For a frozen, action-chunked VLA, estimate whether a non-privileged,
mechanism-conditioned monitor and active intervention selector improve
autonomous recovery over fixed retry at a pre-registered intervention-cost
budget. The primary protocol has exactly one episode-level mechanism:
`none`, `actuator_fault`, or `observation_fault`.

Mechanism (`M`), active exposure (`E_t`), and intervention-relative outcome
(`Y(u)`) are separate variables. A scene pair ID links matched rollouts, while
each rollout retains its own episode ID. A matched seed is a common-random-number
comparison, not automatically a causal counterfactual. Causal/intervention
claims require randomized common-prefix branches or, if state branching is
not available, a weaker episode-level intention-to-treat design.

## Scientific order of operations

1. **Firewall and contract gate:** ensure the monitor cannot receive fault
   schedules, executed actions, simulator state, reward, success, or complete
   `info`; expose typed intervention and action-chunk metadata. The structural
   in-process checks now pass; process-level isolation and the full artifact
   contract remain required before science is unlocked.
2. **Debug smoke test:** run a few seeds to validate environment, logging, and
   close/reset behavior. Do not score it as evidence.
3. **Identifiability pilot:** compare clean, actuator-fault, and
   observation-fault episodes using the fixed 36-episode manifest in
   `configs/identifiability_pilot_v1_4.json`, with hard corruption/channel
   variants, not only all-zero or all-action-zero shortcuts. Run
   `python scripts/validate_research_plan.py` before collection and report
   passive versus probe evidence.
4. **Power and estimand lock:** use pilot estimates to freeze sample size,
   primary endpoints, cost-vector weights/sensitivity, calibration split, and
   multiplicity handling before confirmatory data collection.
5. **Predictive monitor and calibration:** train only on the declared training
   split and calibrate on disjoint clean data. Keep intervention disabled while
   diagnosis is evaluated.
6. **Intervention-effect study:** randomize intervention assignment at a common
   prefix where possible and compare no intervention, fixed retry, fixed
   re-query, selector, and oracle mechanism. If branching is unavailable,
   label the result episode-level intention-to-treat, not counterfactual.
7. **Confirmatory matrix and transfer:** use the frozen balanced schedule, then
   evaluate held-out timing, duration, corruption, task, and policy factors.
   Transfer claims require the extra task/policy gate.

## Primary fault factors

The diagnostic all-zero image and all-channel zero-action conditions are
implementation checks only. The scientific matrix must include hard observation
corruptions (partial masks, blur, stale frames, exposure/color shifts, and
correlated camera corruption) and actuator variants (arm/base/gripper channel,
hold-last, intermittent, and bounded noise). Object displacement, grasp
release, and action delay remain blocked until their task-specific semantics are
documented in a new specification version.

## Primary endpoints and decision gates

The single primary endpoint is fixed before confirmatory collection:

- intention-to-treat recovery difference versus fixed retry at the declared
  cost-vector budget.

Mechanism macro-F1 (`actuator_fault` versus `observation_fault`) on hard exposed
windows is the key secondary identifiability endpoint and a gate for the
mechanism-attribution claim. Clean episodes are excluded from that denominator.

The project must not claim active recovery unless the primary endpoint improves
with a pre-specified confidence/randomization interval and the clean cost
constraint is met. If it does not, report a risk-monitor/diagnosis result only.

## Required reporting

Report the primary endpoint first, followed by the key secondary:

- intention-to-treat recovery difference versus fixed retry at the declared
  cost budget;
- mechanism macro-F1 and detection delay on the hard-corruption set.

Also report clean false intervention rate, per-step and per-episode cost,
success-cost Pareto curves, AUROC/AUPRC, calibration/coverage, missed-fault
censoring, paired confidence intervals, and all required provenance artifacts.

The existing 30-episode clean run is historical checkpoint provenance only. It
does not unlock a confirmatory recovery claim or serve as monitor training data.
The pilot is also not a superiority result. Every report must state the
independent seed/pair unit, branch/randomization status, censoring rule, and
whether the observation condition is diagnostic, hard synthetic, or
naturalistic. A matched-seed result is descriptive; causal recovery language
requires the state-clone/common-prefix gate in
`configs/intervention_protocol_v1_4.json`.

The review disposition and the operational changes are recorded in
[`RESEARCH_SPECIFICATION_REVIEW_RESPONSE.md`](RESEARCH_SPECIFICATION_REVIEW_RESPONSE.md).
