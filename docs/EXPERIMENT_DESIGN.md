# Experiment design

## Research question

Can an external, calibrated runtime monitor improve the net success of a frozen
VLA policy on long-horizon tasks under controlled, recoverable failures?

The benchmark must distinguish four questions:

1. Was a physical or sensing fault actually injected?
2. Did the monitor detect it, and how late?
3. Which response was selected: continue, retry, replan, request help, or stop?
4. Did the task ultimately recover without hiding excessive interventions?

## Unit of evaluation

The unit is an episode with a fixed task, scene, policy checkpoint, random seed,
and fault schedule. A run must include paired clean and faulted episodes with the
same initial seeds.

## Initial fault families

- actuator dropout: replace actions with no-ops for a fixed duration;
- observation occlusion: mask the policy camera for a fixed duration;
- object displacement: move a named object in an explicit coordinate frame;
- grasp release: open the gripper after a verified grasp;
- action delay: execute an older action chunk.

The first two are embodiment-independent. The others require task-specific state
access and must fail closed when their target cannot be identified.

Fault timing is explicit. A `before_action` fault changes execution at the same
step and can have zero-step detection delay. An `after_step` physical fault is
applied after the recorded transition and first appears in the next observation,
so its minimum detection delay is one step. Detections are matched to unmatched
faults only within a configured finite window; an alarm cannot claim an arbitrary
old fault as a true positive.

## Required metrics

- clean and faulted task success;
- fault detection precision, recall, F1, and delay;
- recovery success conditioned on a detected fault;
- false recovery/intervention count on clean episodes;
- episode length and human-help rate;
- per-fault-family confidence calibration in later milestones.

Average success alone is not sufficient. Report episode counts and uncertainty
intervals when moving from smoke tests to paper experiments.

## Current milestone

The local dummy environment validates orchestration only. The first remote
milestone is a RoboCasa RGB smoke test. The first scientific milestone begins
only after a RoboCasa-compatible frozen checkpoint reproduces its published
clean baseline.
