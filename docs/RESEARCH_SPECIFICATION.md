# Research Specification

## Calibrated Active Fault Diagnosis and Recovery for Frozen Action-Chunked VLA Policies

**Status:** pre-science design contract, version 1.4
**Date:** 2026-08-26
**Repository:** `vla-recovery-bench`
**Primary environment:** `robocasa/PickPlaceCounterToCabinet`, `split=target`
**Primary policy:** audited frozen GR00T N1.5 RoboCasa checkpoint

This document is the experiment contract for the scientific phase of VLA
Recovery Bench. It is deliberately stricter than an implementation plan. A
result is not considered evidence for the paper unless it satisfies the
information boundary, split, logging, and reproducibility requirements below.

The document must be updated before changing the research question, the
allowed monitor inputs, the intervention set, the primary metrics, or the
evaluation split. Changes after results have been generated must be recorded as
a new version and must not silently replace an earlier protocol.

Version 1.4 is an operational amendment to version 1.3 after a second design
audit of the external review in
[`RESEARCH_SPECIFICATION_REVIEW.md`](RESEARCH_SPECIFICATION_REVIEW.md). It
keeps the review's valid scientific corrections and fixes a confounding error in
the v1.3 pilot: mechanism was accidentally correlated with onset, duration, and
fault variants. It also makes recovery the single primary endpoint, so a
favorable result cannot be selected from two co-primary tests. The
implementation gate remains open: the plan is executable on paper, but the hard
corruption adapters, per-episode schedule factory, common-prefix branch
capability, and complete artifact writer must pass their checks before a fault
result can be scored.

The review disposition, including accepted, modified, and deferred
recommendations, is recorded in
[`RESEARCH_SPECIFICATION_REVIEW_RESPONSE.md`](RESEARCH_SPECIFICATION_REVIEW_RESPONSE.md).

### Version 1.4 decisions

- The **single primary endpoint** is intention-to-treat recovery difference
  versus fixed retry at a frozen component-wise cost-vector budget. A scalar
  utility is secondary and its weights are frozen before confirmatory data.
- Episode-balanced binary mechanism macro-F1 for `actuator_fault` versus
  `observation_fault` on hard exposed windows is a **key secondary endpoint and
  identifiability gate**, not a second primary hypothesis. Clean episodes are
  excluded from this F1 and are used only for false-intervention and calibration
  estimates. A three-class (`none` included) confusion matrix is exploratory.
- The identifiability pilot is a fixed 12-scene-seed, 36-episode balanced
  crossed design specified in `configs/identifiability_pilot_v1_4.json`. Each
  scene seed generates one clean, one actuator, and one observation episode;
  the two fault episodes reuse the same onset, duration, and scene-level factor
  row. It is an exploratory gate and cannot establish superiority.
- The historical clean run and the pilot are provenance/identifiability
  evidence only. No fault result is scientific evidence until the runtime
  firewall and audit-channel tests pass.
- One task and one frozen GR00T policy remain a bounded case study. A second
  task and policy are required before any transfer, policy-agnostic, or general
  RoboCasa claim.
- Confirmatory sample size is not a rough `80--120` range. It must be
  selected by the versioned simulation inputs in
  `configs/power_analysis_v1_4.json`, with attrition and sensitivity analysis
  reported before test collection.

### Review disposition and scope of the amendment

The external review is substantively accepted, but its recommendations are not
all promoted to claims or implementation facts:

| Review item | v1.4 disposition | Consequence for the plan |
|---|---|---|
| F1, mixed latent modes | Accepted | Mechanism (`M`), exposure (`E_t`), risk (`Q_t`), and potential outcome (`Y(u)`) remain separate. |
| F2/F5, causal language | Accepted with a strict fallback | Common-prefix randomized branches are required for a causal recovery claim; matched seeds alone are descriptive. |
| F3, sample size | Accepted and tightened | The 12-seed pilot is exploratory; confirmatory `n` is produced by a versioned simulation for one primary endpoint, with independent scene seeds as the resampling unit. |
| F4, intervention budget | Accepted and operationalized | The primary intervention set has component-wise limits; raw time/compute/human/risk costs are always reported. |
| M1, novelty | Accepted with reframing | The falsifiable contribution is mechanism-conditioned evidence plus costed intervention, not feature aggregation. |
| M2/M3, shortcut faults | Accepted | All-zero/all-channel-zero conditions are diagnostic only; hard variants are mandatory for the primary endpoint. |
| M4/M5, reward and irrecoverability | Accepted | Reward is excluded from the primary context; irrecoverability is hindsight-only. |
| M6/M7, runtime and artifacts | Accepted as hard gates | No pilot result is scored until the firewall, artifact, and schedule checks pass. |
| M8, external validity | Accepted | The first paper claim is a GR00T/RoboCasa case study; transfer claims require the additional task/policy gate. |

The v1.3 configuration is retained as historical design provenance but is not
valid for data collection: its assignment rows confound mechanism with onset
and fault-family factors. Only the v1.4 configuration may generate a new pilot
manifest. The checked-in v1.4 configuration remains a design manifest: its
`implementation_gate.current_status` is blocked until the runtime adapters,
per-episode schedule derivation, and artifact checks have been executed.

The v1.4 pilot is an identifiability gate, not a recovery comparison. Its
clean/actuator/observation episodes share scene seeds and factor rows, but they
are separate rollouts; this controls measured initial-scene variation without
claiming that post-injection trajectories are identical. A mechanism result is
interpretable only if performance remains above the pre-registered reference
when the diagnostic all-zero/all-channel-zero shortcuts are excluded.

The review's numerical acceptance estimate is not treated as evidence and is
not copied into the paper. It is useful as a warning about missing evidence,
not as a quantitative prior on acceptance.

## 1. Research question

The central question is:

> For a frozen, action-chunked VLA, can an external monitor infer the mechanism
> of one declared execution fault from non-privileged history and requested
> actions, then select a cost-constrained diagnostic or recovery intervention
> that improves autonomous task recovery at a fixed clean intervention cost?

The target is not merely binary failure detection. The system must be evaluated
as a closed-loop runtime safety component with four separately measured jobs:

1. **Risk estimation:** predict whether the current rollout is leaving a
   successful trajectory. OOD and policy uncertainty are evidence sources for
   this score, not fault-mechanism classes in the primary protocol.
2. **Fault diagnosis:** distinguish the declared episode mechanism (no fault,
   actuator fault, or observation fault) when the permitted evidence supports
   that distinction. The mechanism is an episode-level label; whether it is
   currently exposed is a separate, offline-only label.
3. **Intervention selection:** choose whether to continue, re-query the frozen
   policy, use a diagnostic probe, switch camera evidence, retry/replan, ask for
   help, or terminate.
4. **Recovery evaluation:** measure whether the intervention restores task
   success without hiding excessive false alarms or interventions.

The primary protocol contains one fault mechanism per episode. Simultaneous
faults, natural unlabelled failures, and an online `irrecoverable` class are out
of scope until a new protocol version defines their labels and interventions.
Each pilot scene seed generates exactly one episode per condition (clean,
actuator, or observation). The factor row is shared by the two fault episodes,
so mechanism is crossed with onset, duration, camera, and corruption/channel
factors rather than encoded by them. Simultaneous faults and multi-mechanism
episodes remain out of scope.
Recoverability is evaluated as a protocol-relative outcome, not as an
unqualified property of a hidden state. The monitor is not allowed to modify,
fine-tune, or distill the frozen VLA. It is an external runtime component.

## 2. Scientific claim and novelty boundary

### 2.1 Proposed claim

The proposed method is a **fault-hypothesis-conditioned, action-conditional,
calibrated monitor with cost-constrained active diagnosis and recovery-aware
intervention selection** for a frozen VLA. Until the transfer gates in Section
14 pass, this is explicitly a bounded GR00T/RoboCasa case study.

At time `t`, the monitor receives a history of permitted observations and
requested action chunks. It maintains a posterior over the *episode mechanism*
under a single-fault protocol, predicts the consequences of the current action
chunk under each mechanism hypothesis, and selects an intervention using
predicted recovery utility, diagnostic value, and an explicit cost vector. The
final decision is calibrated on held-out clean data and evaluated on matched
seeds and, when the branch requirements pass, randomized common-prefix
branches. A matched seed alone is never called a counterfactual.

The intended contribution is the following formally defined estimand, rather
than an unstructured collection of uncertainty features:

- typed fault-mechanism inference rather than one undifferentiated failure
  score;
- mechanism-conditioned action prediction and residuals;
- active interventions whose purpose can be recovery, diagnosis, or both;
- calibration against a clean intervention-cost budget;
- intention-to-treat recovery evaluation at matched intervention cost.

The **single primary estimand** is the intention-to-treat difference
`Delta_recovery(B) = E[Y(selector) - Y(fixed_retry)]` at a pre-registered clean
intervention-cost budget `B`, where `Y` is autonomous task success and the
expectation is over held-out randomized common-prefix branches (or, if state
branching cannot be implemented, randomized episode-level assignments with a
weaker claim). Mechanism macro-F1 is the **key secondary identifiability
endpoint**, evaluated only on hard exposed windows at the same clean
false-alarm operating point. Clean episodes are excluded from this F1 and are
used for false-intervention and calibration estimates. A three-class confusion
matrix including `none` is exploratory. If the selector does not improve
recovery at matched cost, the work must be reported as a monitor, not as an
active recovery method.

“Improve recovery” means a positive, pre-registered paired intention-to-treat
difference whose two-sided 95% interval excludes zero. A non-significant result
cannot be converted into a cost or Pareto claim after inspecting the results.

### 2.2 What is not claimed as novel by itself

The following are established ingredients and must be treated as baselines,
components, or ablations rather than standalone contributions:

- observation OOD scoring;
- action-chunk variance or ensemble disagreement;
- action-conditioned latent prediction;
- conformal prediction or Platt scaling;
- VLA hidden-feature probes;
- image-health statistics;
- action/state residuals;
- CUSUM, Page-Hinkley, or a finite-state hysteresis machine;
- retry, replan, camera switching, or human-help actions;
- a weighted sum of several risk scores.

The closest prior directions include clean-only uncertainty and conformal
failure detection ([FAIL-Detect](https://arxiv.org/abs/2503.08558)), OOD plus
action-chunk uncertainty ([FIPER](https://arxiv.org/abs/2510.09459)), VLA
internal-feature monitoring ([SAFE](https://arxiv.org/abs/2506.09937)), and
action-conditioned world-model latent monitoring with functional conformal
prediction ([Foresight](https://arxiv.org/abs/2606.23085)). VLA confidence
calibration ([2507.17383](https://arxiv.org/abs/2507.17383)), hidden-activation
perturbation uncertainty ([2606.20754](https://arxiv.org/abs/2606.20754)), and
pre-execution action verification ([Pre-VLA](https://arxiv.org/abs/2605.22446))
must also be acknowledged where applicable.

The paper must demonstrate an improvement specifically attributable to
fault-conditioned diagnosis and active intervention. A result obtained only by
adding more uncertainty features is not sufficient. With one task and one
frozen policy, the result is a GR00T/RoboCasa case study; it must not be called
policy-agnostic or general RoboCasa.

### 2.3 Falsifiable hypotheses

The following hypotheses are preregistered design targets, not guaranteed
outcomes:

- **H1, predictive early warning:** an action-conditional predictive residual
  detects impending failure earlier than image-health, action-state residual,
  and non-predictive OOD baselines at the same clean false-alarm rate.
- **H2, mechanism attribution:** a fault-conditioned posterior distinguishes
  actuator dropout from observation corruption better than a single failure
  score on hard corruptions and channel-specific actuator variants, without
  relying on the fault label online.
- **H3, calibrated decisions:** conformal or equivalent calibration controls
  the pre-registered clean false-intervention rate on held-out seeds.
- **H4, active recovery:** intervention selection using predicted recovery
  utility and diagnostic value improves autonomous recovery success or reduces
  the declared intervention cost relative to fixed retry under intention-to-
  treat assignment.
- **H5, transfer:** a monitor trained without the final fault schedule retains
  useful calibration and diagnosis performance on held-out fault timing,
  duration, intensity, and task combinations.

If H2 or H4 is not supported, the method must be described as a risk monitor,
not as a fault-diagnosis-and-recovery method. If a mechanism is observationally
indistinguishable under the permitted inputs, the protocol must add a declared
diagnostic probe or explicitly report the mechanism as non-identifiable.

## 3. Fixed system contract

### 3.1 Environment and policy

The first implementation is bound to the already audited contract:

- environment: `robocasa/PickPlaceCounterToCabinet`;
- split: `target`;
- embodiment: `PandaOmron`;
- control frequency: 20 Hz;
- cameras:
  `video.robot0_agentview_left`, `video.robot0_agentview_right`,
  `video.robot0_eye_in_hand`;
- camera source frames: `256x256x3`, `uint8`;
- proprioception:
  `state.end_effector_position_relative` (3),
  `state.end_effector_rotation_relative` (4),
  `state.gripper_qpos` (2), `state.base_position` (3),
  `state.base_rotation` (4);
- native RoboCasa action: structured Gymnasium `Dict`, 12 scalar values in
  `[-1, 1]`;
- policy output: official GR00T action chunk with model dimension 32, native
  RoboCasa action dimension 12 after the documented transform, and chunk length
  16;
- control semantics: base-frame delta `OSC_POSE` arm control, joint-velocity
  base control, and joint-position torso control;
- language input: `annotation.human.task_description`.

The local evidence is recorded in
[`ROBOCASA_OFFICIAL_CHECKPOINT_AUDIT.md`](ROBOCASA_OFFICIAL_CHECKPOINT_AUDIT.md),
[`SMOLVLA_INTEGRATION.md`](SMOLVLA_INTEGRATION.md), and the probe output
`/home/pc/VLA/outputs/robocasa_contract.json`.

The clean baseline currently retained as provenance is:

```text
/home/pc/VLA/outputs/groot_atomic_seen_30p_clean_baseline_run2/
```

It contains 30 no-fault episodes on seeds 0 through 29, with 19 successes.
It is an environment/checkpoint baseline, not a detector or recovery result,
and must not be used to tune the final monitor or final thresholds. It is a
historical provenance artifact and is not large enough to support the
confirmatory sample-size or calibration claims below.

### 3.2 Frozen-policy requirements

The policy adapter must satisfy all of the following during every scientific
run:

- policy is in evaluation mode;
- all policy parameters have `requires_grad=False`;
- inference is under `no_grad` or the equivalent inference-only context;
- parameter hash before and after evaluation is identical;
- checkpoint files and per-file SHA256 values match the manifest;
- requested actions pass shape, finite-value, and Gymnasium range checks;
- action chunk reset occurs after a recovery/re-query intervention;
- no policy gradient, optimizer, replay buffer, or test-time parameter update is
  permitted.

The policy process and simulator process may remain separate so policy latency
is measured independently from simulation time.

## 4. Information boundary

The information boundary is part of the method definition. Every monitor input
must be explicitly listed in the run manifest.

### 4.1 Allowed online inputs

At step `t`, the primary monitor may use:

- the previous and current non-privileged observation mappings;
- a bounded history of observations and requested action chunks;
- the task instruction as supplied to the policy;
- the requested action chunk emitted by the frozen policy;
- no reward signal. Reward is excluded from the primary track because RoboCasa
  reward and success timing are not a stable sensor contract. A reward-enabled
  monitor is a separately named ablation with its timing recorded;
- wall-clock and policy-inference latency, if declared as an input;
- outputs of a monitor-owned encoder or predictor trained only on permitted
  data;
- results of an explicitly declared diagnostic intervention.

The monitor may derive image embeddings, proprioceptive features, temporal
differences, predictive residuals, uncertainty scores, and task-progress
estimates from these inputs.

### 4.2 Forbidden online inputs

The following are privileged labels or leakage channels and must never reach the
monitor, recovery controller, or policy adapter as an online feature:

- `FaultSpec`, fault ID, fault type, scheduled fault time, duration, or
  intensity;
- `FaultApplication` details or the environment's internal dropout counter;
- `info["success"]` or any terminal success label before the decision being
  evaluated;
- simulator ground-truth object poses, contacts, body IDs, joint torques, or
  other MuJoCo state not present in the declared observation contract;
- the actual executed action when it differs from the requested action due to a
  fault;
- an oracle reset or a hidden clean/faulted episode indicator;
- future observations, future rewards, or the final episode outcome;
- any model checkpoint, hidden state, or attention map unless the experiment is
  explicitly labeled as a white-box variant;
- a detector trained on final-test seeds or on their fault labels.

The recorder may store requested and executed actions separately for audit, but
the monitor must only receive the requested action and permitted observations.

### 4.3 Black-box primary track and white-box extension

The primary track is black-box with respect to the VLA: only policy inputs and
requested outputs are exposed. A white-box extension may use frozen GR00T hidden
features or activation perturbations, but it must:

- be a separately named method variant;
- report extra compute and memory;
- preserve parameter hashes;
- include an ablation against the black-box method;
- never be used to redefine the primary claim.

## 5. Formal problem definition and identifiability

The review correctly notes that a fault mechanism, its active exposure, and
future recoverability are different objects. The primary protocol therefore
uses separate variables:

- `M` is the episode-level mechanism label: `none`, `actuator_fault`, or
  `observation_fault`;
- `E_t` is the offline exposure indicator, equal to one only while the declared
  fault is active. It is never passed to the monitor;
- `Y(u)` is the potential task-success outcome under intervention policy `u` in
  the declared branch protocol;
- `Q_t` is an online risk/progress estimate, not an `irrecoverable` class.

The primary protocol contains exactly one mechanism per episode. Mechanism
labels and exposure intervals are generated by the offline fault manifest and
are used only for training in explicitly supervised tracks and for evaluation.
There is no online `irrecoverable` label. A hindsight recoverability analysis
may compare `Y(u)` across declared interventions, but it must not be presented
as an observable state classifier.

The label roles are fixed as follows:

| Symbol | Meaning | Available online? | Primary use |
|---|---|---:|---|
| `M` | episode mechanism: `none`, `actuator_fault`, or `observation_fault` | no | offline diagnosis scoring and explicitly supervised monitor training |
| `E_t` | whether the declared fault is active at step `t` | no | detection-delay and exposure-window scoring |
| `T_0` | injection onset step from the offline manifest | no | delay reference only |
| `Y(u)` | success under intervention policy `u` in a declared branch | no | intention-to-treat recovery analysis |
| `Q_t` | monitor's online risk/progress estimate | yes, as output | alarm and intervention selection |

Mechanism diagnosis is scored on exposed fault windows (`E_t=1`) and uses only
the two fault mechanisms in the primary comparison; clean episodes are used to
estimate false alarms and calibration. The primary unit is the **episode**, not
the number of frames: for each episode, average posterior probabilities over
its exposed window, take the argmax as the episode diagnosis, and compute the
macro-F1 over independent scene-seeds. A posterior before `T_0` is an
anticipatory risk signal, not evidence that the monitor has observed a fault.
This prevents an impossible-to-observe future label from entering the diagnosis
metric.

Let `o_t` be the permitted observation, `a[t:t+H-1]` the requested action chunk
of length `H=16`, and `x_t=(o[t-L:t], a[t-L:t+H-1])` the bounded history. The
primary monitor estimates an episode-mechanism belief only from the allowed
history; the evaluator scores it on the declared exposure windows:

\[
\hat b_t(m)=P(M=m\mid x_t),
\qquad m\in\{\text{none},\text{actuator\_fault},
\text{observation\_fault}\}.
\]

Before a fault is exposed, the posterior is evaluated as a risk/alarm signal,
not as evidence that the monitor has already identified a future fault. After
exposure, mechanism diagnosis is scored against the offline `M` label. OOD and
policy uncertainty remain evidence sources and are not additional mechanism
classes.

For each mechanism hypothesis `m`, a predictor estimates future latent
observations, proprioceptive deltas, and a task-progress/success quantity:

\[
(\hat z_{t+1:t+K}^{(m)},\hat q_{t+K}^{(m)})
=F_\theta(x_t,m).
\]

After an allowed transition, the monitor computes a standardized predictive
residual, for example:

\[
r_t(m)=
\|z_{t+1}-\hat z_{t+1}^{(m)}\|_{\Sigma_m^{-1}}
\quad\text{or a calibrated multi-step equivalent.}
\]

The mechanism is considered observationally identifiable for a condition only
if a held-out monitor using the permitted inputs beats the pre-registered
single-score baseline by the pre-registered minimum effect and its confidence
interval excludes the non-identifiable reference. For the binary
actuator-versus-observation pilot, the default planning reference is balanced
accuracy `0.5` and the default minimum useful effect is `0.15`; these values
must be frozen in the pilot manifest before data collection. The pilot must also
report an offline oracle upper bound and explicitly mark conditions where
low-level corruption makes the answer trivial.

The risk and diagnosis outputs must retain the components needed for audit:

```json
{
  "risk": 0.0,
  "failure_detected": false,
  "failure_type": null,
  "posterior": {
    "none": 0.0,
    "actuator_fault": 0.0,
    "observation_fault": 0.0
  },
  "evidence": {
    "predictive_residual": 0.0,
    "action_uncertainty": 0.0,
    "observation_ood": 0.0,
    "progress_probability": 0.0
  },
  "calibration": {
    "method": "declared_method",
    "p_value": 0.0,
    "threshold": 0.0
  }
}
```

The implementation may use a different mathematical parameterization, but it
must expose equivalent quantities and preserve the distinction between raw
scores, calibrated risk, posterior over mechanisms, and selected intervention.

## 6. Proposed monitor architecture

The first research implementation should be modular so each source of gain can
be ablated.

### 6.1 Observation/history encoder

Encode the three camera streams, the five proprioceptive fields, instruction,
and requested action chunks into a time-indexed representation. The encoder may
be a compact CNN/ViT plus an MLP/GRU/Transformer. It must not silently drop a
declared field; missing or malformed fields cause a failed contract check.

The primary track should use a lightweight, independently trained encoder so
the method does not depend on a particular VLA's hidden representation.

### 6.2 Fault-hypothesis-conditioned predictor

The predictor receives the same history under each candidate mode and predicts
future latent observations, proprioception deltas, and a task-progress/success
quantity. It should support multi-step prediction because a single frame can be
ambiguous.

The predictor is trained on monitor-training data only. It is not a simulator
oracle and must not consume MuJoCo privileged state.

### 6.3 Uncertainty branch

Action uncertainty may be estimated from declared stochastic policy queries,
chunk ensembles, or a separately trained uncertainty head. If a white-box
variant uses hidden activation perturbation, the perturbation distribution,
layer, count, and compute budget must be fixed before final evaluation.

Uncertainty is evidence, not a calibrated failure probability, until the
calibration procedure has been applied.

### 6.4 Belief update and temporal persistence

Update the episode-mechanism posterior from predictive residuals and
uncertainty. A temporal filter is allowed, but its state, prior, transition
matrix, and patience parameters must be fitted on the training/calibration
split only. The primary label is the fixed episode mechanism `M`; the monitor
may maintain a time-varying *belief* about it, but must not pretend that the
mechanism switches freely at every step.

The monitor must distinguish:

- instantaneous evidence;
- persistent suspicion;
- confirmed exposure-time alarm;
- cooldown after an intervention.

This prevents a single noisy frame from being counted as a confirmed fault and
makes detection delay well-defined. Exposure onset and duration remain offline
labels used for scoring, not online state inputs.

### 6.5 Recovery-aware critic

Estimate, for each available intervention `u`, the probability of eventual
task success and the expected intervention cost:

\[
V(u\mid x_t)=
P(\text{success}\mid x_t,u)
-\lambda C(u).
\]

This critic is trained without exposing final-test outcomes at decision time.
It may use final episode labels during monitor training, but the split must be
explicit and disjoint from calibration and final evaluation. The critic must
be trained on randomized intervention assignments or a declared offline-policy
evaluation procedure; fitting it only on trajectories selected by a previous
monitor is not accepted as causal evidence.

### 6.6 Active diagnosis and intervention selection

The intervention policy selects:

- `continue`;
- `requery_policy`;
- `reissue_current_chunk`;
- `switch_camera_subset`;
- `diagnostic_probe`;
- `retry` or `replan`;
- `request_help`;
- `terminate`.

The diagnostic probe must be a short, predeclared, bounded action sequence. It
must be legal under the RoboCasa action space and must not inspect fault labels.

The preferred objective is:

\[
u_t^*=\arg\max_u\left[
V(u\mid x_t)+\eta I(M;o_{t+1}\mid x_t,u)
\right],
\]

subject to a clean false-intervention constraint. Recovery and diagnosis are
therefore coupled, but the policy remains frozen.

## 7. Fault model and experimental factors

### 7.1 Primary fault families

The initial scientific track contains only faults that can be injected without
task-specific MuJoCo body semantics:

1. **Actuator fault family:** begin with an all-channel zero-like dropout only
   as a reproducibility probe, then evaluate channel-specific arm/base/gripper
   variants, hold-last-action, intermittent dropout, and bounded execution
   noise. Each variant is a separate mechanism condition until a pilot shows
   that they share the same observable signature and recovery semantics.
2. **Observation fault family:** use a corruption ladder consisting of (a) an
   all-zero diagnostic condition, (b) hard partial masks, blur, frozen/stale
   frames, exposure/color shifts, and correlated multi-camera corruption, and
   (c) a naturalistic/asset-backed occlusion condition where available. The
   all-zero condition is an implementation check, not evidence for a general
   observation-fault claim.

The implementation must record both the requested action and the executed
action for audit, while enforcing the information boundary above. The executed
action is never exposed to the primary monitor.

Object displacement, grasp release, and action delay are out of scope for the
first method claim. They require independently documented body semantics,
coordinate frames, and additional causal assumptions. They may be added only in
a new protocol version.

### 7.2 Fault factors and corruption ladder

Do not evaluate only one fixed fault schedule. The final matrix must vary onset
phase, duration, severity, camera/channel, and action-chunk alignment. The
observation ladder is mandatory:

1. **Diagnostic condition:** all-zero masking, used only to validate injection
   and logging;
2. **Hard synthetic conditions:** partial spatial masks, blur, frozen/stale
   frames, exposure/color shifts, and correlated multi-camera corruption;
3. **Naturalistic condition:** an asset-backed or rendered occlusion when the
   environment supports it.

The actuator ladder must include at least all-channel, arm-only, base-only,
gripper-only, hold-last-action, intermittent, and bounded-noise variants before
those variants are collapsed into a common mechanism label. The exact values
  must be written to a versioned configuration before the final run. In the v1.4
pilot, `arm_only` and `gripper_only` are the channel factors; hold-last,
intermittent, and bounded-noise variants are explicitly deferred to the
confirmatory matrix and cannot be inferred from the pilot.

### 7.3 Matched seeds and randomized branches

For every final faulted episode, run a clean episode with the same task, split,
seed, checkpoint, and initial environment configuration. Clean and faulted
episodes must be linked by a stable **scene pair ID**; each rollout also has a
distinct episode ID. A pair contains the declared clean and fault condition
rollouts and is a **matched-seed/common-random-number comparison**, not
automatically a counterfactual causal estimate. The v1.4 pilot's three episodes
per scene seed are one clean, one actuator, and one observation condition, with
distinct episode IDs and a shared scene pair ID.

For recovery claims, create a common-prefix decision checkpoint: pause after
the same observation/action history, clone or deterministically replay the
state when the simulator supports it, assign `continue`, `fixed_retry`,
`diagnostic_probe`, or `selector` using the frozen rule below, and then resume
separate branches. The branch randomization seed is `SHA256(pair_id ||
decision_step || protocol_version)`, with equal probability over eligible
arms and no re-randomization after assignment. A branch is eligible only if all
arms can execute the same remaining horizon and intervention budget. If exact
state cloning is unavailable, use episode-level randomized assignment and label
the result as a weaker intention-to-treat comparison. Report where
post-injection simulator randomness diverges and never describe a matched seed
alone as a strict counterfactual.

The current RoboCasa adapter does not yet expose simulator state cloning. Until
that capability is implemented and tested, no recovery result may use causal
common-prefix language; it must be labeled episode-randomized intention-to-
treat or matched-seed descriptive evidence. For observation faults, the
adapter's current after-step transform means the first affected policy input is
one control step after the injection call; the timing contract must log both
timestamps and use the first affected input as the delay origin.

## 8. Data protocol and split

The final test set must not influence monitor training, hyperparameter tuning,
threshold selection, or architecture choice.

Recommended staged split (planning ranges, not a collected dataset):

```text
monitor training:       clean seeds 100..299, plus declared training faults
calibration:             clean seeds 300..399
threshold validation:    clean seeds 400..499
  pilot / mechanism test:  seeds 500..511 (fixed v1.4 pilot; 36 episodes)
confirmatory evaluation: power-selected seeds after pilot (paired/stratified)
transfer evaluation:     newly frozen task/factor/policy holdout
```

These ranges are a planning minimum, not a claim that every episode is
independent. The pilot has 12 independent scene-seed units, each producing
three condition episodes; the episode is the rollout unit, while the seed is
the independent resampling unit. The final seed count must be selected by the
  simulation protocol in `configs/power_analysis_v1_4.json` before confirmatory
collection. Thirty episodes are suitable for historical provenance only, not
for a confirmatory superiority claim.

The current seeds 0..29 clean run is retained as historical checkpoint
provenance. It must not be used for training, calibration, threshold selection,
or a confirmatory superiority claim. It may be reported as a held-out
integration reference only.

For each monitor-training episode, store:

- task, split, seed, pair ID, policy/checkpoint hash;
- observation contract and action contract version;
- requested action chunks and timestamps;
- permitted observations. Rewards are stored only in the offline audit stream;
  they are not part of the primary monitor input or training tensor.
- fault schedule only in the offline label file, not in monitor inputs;
- final success and termination reason;
- environment and software commits.

Fault labels are available to offline training and scoring only when the chosen
track explicitly permits supervised failure data. A clean-only variant must be
implemented and reported separately, not mixed with supervised labels. The
confirmatory analysis must predeclare one primary recovery endpoint
(`Delta_recovery(B)`). Mechanism macro-F1 on the hard-corruption set is the key
secondary identifiability endpoint; all other metrics are secondary or
exploratory.

## 9. Baselines and ablations

### 9.1 Required baselines

At minimum, compare:

1. no monitor, always continue;
2. image-health statistics;
3. action-state residual;
4. residual plus temporal hysteresis;
5. clean-only OOD detector;
6. OOD plus action uncertainty in the style of FIPER;
7. action-conditioned predictor without fault hypotheses, in the style of
   Foresight;
8. VLA-feature monitor for the white-box track, in the style of SAFE;
9. fixed retry after any alarm;
10. oracle fault type with the same intervention budget.

All baselines must use the same episodes, fault schedules, and policy
checkpoint. Detector thresholds must be calibrated on the same calibration
split whenever a calibrated comparison is claimed.

### 9.2 Required ablations

The proposed method must be evaluated with:

- no fault-hypothesis conditioning;
- no action-conditional prediction;
- no uncertainty branch;
- no calibration;
- no recovery-aware critic;
- no information-gain term;
- no active diagnosis, fixed retry instead;
- no multi-step history;
- black-box versus white-box features;
- single-camera versus all declared cameras.

The main claim is unsupported if removing fault conditioning or active
diagnosis has no measurable effect under appropriate confidence intervals.

## 10. Metrics and decision criteria

### 10.1 Detection and diagnosis

Report, with episode counts and confidence intervals:

- AUROC and AUPRC;
- precision, recall, F1 at the predeclared operating point;
- detection delay in environment steps and seconds;
- recall by fault family, onset phase, duration, and severity;
- confusion matrix for mechanism diagnosis;
- false alarms per clean episode and per 1,000 steps;
- alarm persistence and duplicate-alarm rate.

### 10.2 Calibration

Report:

- clean false-intervention rate;
- expected calibration error and reliability diagrams;
- conformal coverage or the exact finite-sample guarantee claimed;
- risk-stratified success curves;
- calibration drift on held-out timing and fault factors.

No calibration guarantee may be claimed without stating its assumptions and
whether the sequential dependence violates them. If a sequential conformal or
functional conformal method is used, the exact implementation and calibration
window must be recorded.

### 10.3 Recovery and cost

Report:

- clean task success;
- faulted task success with no recovery;
- faulted task success with monitor/recovery;
- recovery success conditioned on a true detected fault;
- intervention count and a declared intervention-cost vector;
- request-help and terminate rates;
- episode steps and wall-clock time;
- policy inference latency and monitor latency;
- GPU memory and CPU overhead.

The cost vector must report at least:

- elapsed execution time;
- extra policy/monitor computation and memory;
- human-help burden;
- task-risk or irreversible-action penalty, if applicable.

For v1.4 the components are measured, rather than inferred from an intervention
count: `elapsed_execution_time_ms` is wall-clock time from the first policy
request through episode termination; `extra_compute_ms` is monitor/intervention
time above a no-monitor reference on the same machine; `peak_memory_mb` is the
maximum resident/GPU allocation recorded during the episode; `human_help_count`
is one per `request_help` decision; and `risk_penalty` is zero unless a future
specification version defines a task-specific irreversible-action event. The
primary budget is component-wise: no confirmatory episode may exceed
`time<=1.25x` the paired no-monitor reference, `extra_compute<=250 ms/episode`,
`human_help_count<=0` for the autonomous track, or `risk_penalty>0`. Raw values,
missing measurements, and the reference run ID must be logged. The scalar
weighted cost is secondary and uses only weights frozen in the manifest.

Report each component separately, the number of interventions per episode and
per 1,000 environment steps, clean success degradation, and a success-versus-
cost Pareto frontier. A scalar cost may be used for the primary constrained
analysis only after its weights and a sensitivity range have been frozen before
final evaluation.

The primary operating curve is:

> recovery success at a fixed clean component-wise intervention-cost budget;
> detection delay is a key secondary operating curve.

The clean budget is therefore a bound on the declared cost vector (with a
separately reported event-rate view), not simply a count of non-`continue`
episodes. A provisional event-rate point of 5% may be shown as a secondary
operating point, but it is not the sole safety budget and cannot be selected
after seeing test results.

### 10.4 Statistical analysis

Use paired comparisons on matched seeds whenever possible and randomized
common-prefix branches for intervention effects. Report bootstrap or
randomization confidence intervals, the number of independent seeds, and the
unit of resampling. Do not treat individual frames as independent episodes.

The confirmatory analysis must include a simulation-based power analysis before
data collection. The v1.4 inputs are frozen in
`configs/power_analysis_v1_4.json`: two-sided `alpha=0.05`, target power `0.80`,
minimum detectable absolute effect `0.20`, baseline recovery rate `0.50`,
within-pair correlation `0.30`, 5% invalid-run rate, and 10% attrition
allowance. Simulate 10,000 replicates for every candidate sample size and all
declared sensitivity scenarios; select the smallest candidate that reaches the
target power in every scenario. This file must be copied into the run manifest
and its generated result must be immutable before confirmatory collection. Use hierarchical
paired bootstrap or a pre-specified mixed-effects model with seed/pair as the
independent unit. Detection-delay analyses must define right-censoring for
missed faults and use a declared censored-time estimator or report missed cases
separately.

For multiple fault families and operating points, control or disclose multiple
comparisons. The single primary recovery endpoint is tested first; diagnosis, delay,
calibration, transfer, and Pareto-frontier analyses are secondary or
exploratory unless a multiplicity plan promotes them. A single favorable seed
subset is not evidence of improvement.

## 11. Required run artifacts and implementation firewall

Every run must be self-describing and must refuse to overwrite a non-empty
output directory. These are implementation requirements, not documentation
aspirations. No pilot or scientific result may be scored until the monitor-safe
context and audit channel have passed an automated test. At minimum, write:

```text
run_manifest.json
episodes.jsonl
metrics.json
monitor_config.json
calibration.json
software_versions.json
policy_state_before.json
policy_state_after.json
shard_integrity.json (formal monitor-data shards)
```

For sharded formal monitor data, `shard_integrity.json` must seal the byte size
and SHA256 of every source artifact except itself. A read-only pre-training gate
must require exact protocol-partition seed coverage, three crossed conditions
per seed, no duplicate seed across shards, identical policy/environment/software
provenance, full configured horizon, a clean repository snapshot, and agreement
between the HDF5 channels and all JSONL indices. Debug or partial coverage must
never be silently promoted to formal training data.

The runtime must expose two separate records:

- `MonitorContext`: only the declared non-privileged inputs;
- `AuditRecord`: fault schedule, executed action, success, and other offline
  labels, inaccessible to the monitor process.

Passing a complete simulator `info` mapping to a monitor is prohibited. The
primary context must omit reward and terminal success; a reward-enabled
ablation must use a distinct typed context.

Each episode record must contain:

- `episode_id`, `pair_id`, `seed`, task and split;
- policy/checkpoint and environment commits;
- fault label in the offline audit section;
- requested action shape/range and finite-value checks;
- observation keys/shapes and camera health features used;
- raw monitor scores, calibrated risk, posterior, evidence;
- alarm step, diagnosed type, and detection delay;
- selected intervention and reason;
- success, reward, steps, termination reason;
- monitor and policy latency;
- any saturation or contract violation;
- parameter hash state and error details if the episode fails.

The primary result must be reproducible from the manifest and immutable input
artifacts without relying on shell history.

Before Phase 0, implement and test:

- typed intervention enum and serialization for `continue`, `requery_policy`,
  `reissue_current_chunk`, `switch_camera_subset`, `diagnostic_probe`,
  `retry`, `replan`, `request_help`, and `terminate`;
- action-chunk position and remaining-horizon metadata;
- explicit chunk reset semantics after re-query/retry/replan;
- monitor-context firewall tests proving forbidden fields are absent;
- atomic refusal to write into a non-empty output directory;
- complete artifact manifest generation.

## 12. Implementation phases and gates

### Phase 0: specification, firewall, and identifiability pilot

Before training a detector:

- freeze this specification version;
- implement and test the monitor-safe context and typed intervention protocol;
- collect the frozen 36-episode pilot from
  `configs/identifiability_pilot_v1_4.json`: 12 independent scene-seeds, each
  with one clean episode, one actuator-fault episode, and one observation-fault
  episode. Three seeds are a debugging smoke test only;
- use the exact per-scene-seed factor row in the manifest. Each scene seed is
  reused for both fault mechanisms with the same onset and duration; the pilot
  covers onset steps 80/240, durations 4/8, actuator variants `arm_only`/
  `gripper_only`, observation variants `partial_mask`/`blur`/`stale_frame`/
  `color_shift`, and all three declared cameras. The all-zero and all-channel-
  zero conditions are separate diagnostic checks and are not counted toward the
  36 episodes;
- verify that the monitor sees only permitted fields;
- test whether passive evidence can distinguish mechanisms, and quantify the
  value of a predeclared diagnostic probe.

**Gate:** if the mechanisms are not distinguishable from passive evidence, add a
predeclared diagnostic probe rather than leaking simulator labels. If a probe
cannot distinguish them without unacceptable cost, narrow the claim to
recovery-aware risk estimation. The all-zero image condition cannot by itself
pass this gate.

### Phase 1: monitor-only predictive model

Train the black-box action-conditional predictor and risk head. Do not enable
active recovery yet. Evaluate on calibration and held-out validation splits.

**Gate:** detector must beat the temporal-residual baseline at the same clean
false-alarm budget, or the proposed predictor is not justified.

### Phase 2: mechanism posterior

Add fault-hypothesis conditioning and evaluate diagnosis confusion matrices.
Keep the intervention fixed so diagnosis quality is not confounded with a
different recovery policy.

**Gate:** mechanism attribution must improve at least one predeclared metric
without increasing clean false interventions beyond the budget.

### Phase 3: calibration

Freeze the architecture and fit conformal/other calibration only on the
calibration split. Threshold validation may select the operating point but may
not access final test outcomes.

**Gate:** held-out clean false-intervention rate must satisfy the predeclared
budget within the reported confidence interval, or the method is marked
uncalibrated and cannot make a calibration claim.

### Phase 4: intervention-effect study

Enable the intervention selector and compare it with no intervention, fixed
retry, fixed re-query, and oracle fault-type policies. Use the frozen
common-prefix randomization rule in Section 7.3 only after state cloning and
branch equivalence tests pass. Otherwise use episode-level randomized
assignment and report the weaker intention-to-treat estimand. Equalize and
report the component-wise cost vector; do not select scalar weights after seeing
results.

**Gate:** active diagnosis must improve recovery utility, recovery success, or
intervention cost with a statistically supported paired comparison. Otherwise
report the monitor as detection-only.

### Phase 5: final matrix and generalization

Run the complete paired matrix across held-out timing, duration, severity,
camera, and task factors. Only after this phase may failure-recovery claims be
made in the paper.

## 13. Reproducibility and provenance

Record, at minimum:

- repository commit and dirty-worktree status;
- RoboCasa and RoboSuite commits;
- official GR00T fork commit;
- checkpoint revision, file sizes, per-file SHA256, and set hash;
- Python, PyTorch, CUDA, MuJoCo, Gymnasium, RoboCasa, and RoboSuite versions;
- GPU model, driver, and memory;
- seed list and fault configuration;
- monitor architecture/configuration and random seeds;
- calibration split and threshold-selection procedure;
- command line and output paths;
- parameter hashes before and after policy evaluation.

Use deterministic seeds where supported. If CUDA, MuJoCo, or a data loader is
not bitwise deterministic, report the nondeterministic component and quantify
run-to-run variation with repeated seeds.

No checkpoint, dataset, or output artifact may be silently replaced. Existing
clean baseline and smoke-test outputs are immutable provenance.

## 14. Reviewer-facing comparison claims

The paper may claim the following only if the corresponding experiment passes:

| Claim | Required evidence |
|---|---|
| Earlier warning | Same clean false-alarm budget and paired held-out episodes against temporal residual, FIPER-style, and Foresight-style baselines |
| Better diagnosis | Mechanism confusion matrix and held-out fault-factor generalization, without fault labels online |
| Better calibration | Held-out clean false-intervention control and reliability/coverage analysis |
| Better recovery | Matched intervention budget against fixed retry/re-query and oracle fault-type controls |
| Policy-agnostic monitor | At least two frozen policies or a carefully controlled cross-policy test; otherwise call it GR00T-specific |
| General RoboCasa result | More than one task and more than one fault schedule; one task is a case study only |

If only the first RoboCasa task and one frozen GR00T checkpoint are available,
the title and abstract must say so explicitly. Do not call the method
general-purpose or policy-agnostic without the required transfer experiment.

## 15. Stop conditions and prohibited shortcuts

Stop the scientific run and fix the protocol if any of the following occurs:

- a monitor input is discovered to contain a privileged simulator label;
- a test seed is used for training or threshold tuning;
- the policy parameter hash changes;
- the adapter silently drops or fabricates an observation field;
- a requested or recovered action is outside the declared Gymnasium range;
- a failure is counted without a logged injection or an explicitly declared
  natural-failure protocol;
- a clean intervention budget is exceeded without reporting it;
- a fault cannot be reproduced from the recorded seed and configuration;
- an output directory would be overwritten;
- a result depends on a hidden manual reset or operator decision.

The following shortcuts are prohibited:

- reading `FaultSpec`, `info["success"]`, MuJoCo body state, or executed action
  as monitor input;
- tuning thresholds on final paired episodes;
- using an oracle fault type while claiming automatic diagnosis;
- masking only all pixels and calling the result general observation-fault
  detection;
- reporting only AUROC or average success;
- presenting the existing clean baseline as a recovery result;
- downloading or using an unrelated generic VLA checkpoint and calling it a
  RoboCasa baseline;
- changing metrics or the primary claim after seeing favorable results.

## 16. Immediate next experiment

The next implementation task is **Phase 0 firewall and identifiability pilot**,
not a full 30-seed failure experiment and not detector training.

First run a three-seed debugging smoke test to validate environment and logging.
It is not scored as evidence. Then run the exact 36-episode pilot specified in
  `configs/identifiability_pilot_v1_4.json` (12 independent matched seeds, three
episodes per seed) under:

```text
clean
actuator_dropout
observation_occlusion
```

The assignment covers two onset positions, two durations, two actuator channel
variants, four hard observation corruptions, and all three cameras. The primary
monitor receives no reward. Log only the allowed observation history and
requested action chunks in its input stream; write schedules, executed actions,
rewards, and success labels to the separate offline audit stream.
Produce:

```text
/home/pc/VLA/outputs/identifiability_pilot_v1_4/
  run_manifest.json
  episodes.jsonl
  metrics.json
```

The pilot must answer four questions before a learned monitor is implemented:

1. Can passive evidence distinguish the two fault mechanisms?
2. Which evidence appears before task failure rather than after it?
3. Does a diagnostic probe add information that passive monitoring does not?
4. Are the mechanism and intervention effects still measurable when the
   all-zero/zero-action shortcuts are removed?

Only after the firewall and pilot gates pass, and a simulation-based power
analysis freezes the confirmatory sample size, should the project implement the
predictive model, calibration, and active recovery selector. Fault experiments
must remain disabled while any contract check is failing.
