# RoboCasa official checkpoint audit

Audit date: 2026-08-26. The official metadata/configuration files and both model
shards for the selected candidate were fetched from the pinned Hub revision and
verified against the official per-file SHA256 values. The weights were then
loaded only through the frozen adapter for a no-fault baseline. The repository
revisions inspected were
RoboCasa
`a07e365c958c4216cd6bbd5f30b47f09a65c6f00`, RoboSuite
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`, the official RoboCasa GR00T fork
`9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10`, OpenPI fork
`5a6beda9ff99da30b4e1b59320f6a32971d7c397`,
and Diffusion Policy fork `41212698b6f481ed92a55f0d5f1778ec56bea417`.

## Local target contract

- Environment: `robocasa/PickPlaceCounterToCabinet`
- Split: `target`
- Embodiment: `PandaOmron`
- Cameras: `video.robot0_agentview_left`, `video.robot0_agentview_right`,
  `video.robot0_eye_in_hand`
- Camera frames: `256x256x3 uint8`
- Proprioception: `state.end_effector_position_relative` (3),
  `state.end_effector_rotation_relative` (4), `state.gripper_qpos` (2),
  `state.base_position` (3), `state.base_rotation` (4), total 16 values
- RoboCasa action: structured Dict, 12 values total, all Box components in
  `[-1, 1]`, with binary semantics for `gripper_close` and `control_mode`
- Control frequency: 20 Hz

## Candidate table

| Candidate | Official revision/source | Evidence found | Blocking gaps | Downloaded |
|---|---|---|---|---|
| GR00T N1.5 target-only atomic-seen checkpoint-60000 | `robocasa-benchmark/Isaac-GR00T`, HF path `gr00t_n1-5/foundation_model_learning/target_only/atomic_seen/checkpoint-60000` | `PandaOmronDataConfig` names all 3 cameras, all 5 state fields, all 5 action fields; state/action normalization modes; quaternion to rotation-6D; video crop/resize; language key; `action_dim=32`, `action_horizon=16`; checkpoint config and metadata available | model output is padded 32-D, not native 12-D; exact inference preprocessing/rotation convention must be frozen in adapter; checkpoint metadata does not record complete training package/environment versions; no exact task/seed-0..29 clean reference; complete checkpoint SHA256 is unavailable without downloading both shards. Official shard LFS oids: `d78d7b910a7ed9e20ef5d7963f800ddec291f712d8b643103453caa5fe648154`, `5a1693ec80308187c8dda2adb15843903d57ca159085408907cc3de8ade546ee` | No |
| GR00T N1.5 target-fraction atomic-seen 30% checkpoint | `robocasa-benchmark/Isaac-GR00T`, HF path `gr00t_n1-5/target_fraction/atomic_seen_30p` | Same PandaOmron modality/action evidence; official README and `eval_results.json` report 30 held-out test rollouts per task, including `PickPlaceCounterToCabinet` / `PnPCounterToCabinet` at 60.0%; official source code resolves the 32-to-12 split, frame, control mode, normalization, and 16-step chunk | official reference does not publish the fixed seeds 0..29; checkpoint metadata has no runtime lockfile | Complete; both shards downloaded and SHA256 verified locally |
| GR00T N1.5 multitask checkpoint-120000 | same | Same PandaOmron interface evidence; HF config has 32-D actions and 16-step horizon | pretrain training distribution, not target-only; no exact target-task clean reference; same environment-version and adapter gaps | No |
| π0 RoboCasa pretrain checkpoint-75000 | `robocasa-benchmark/openpi`, HF `robocasa/robocasa365_checkpoints` | official `RobocasaInputs/Outputs`; three camera input names in eval code; 16-D state concatenation; 12-D output slice; 224 resize-with-pad; prompt passthrough; `convert_action`; checkpoint norm stats available | model config is 32-D/50-step by default and pads state/actions; official norm stats include padded dimensions; no explicit PandaOmron control-mode/frame manifest in checkpoint; clean reference is aggregate pretrain, not target exact task; training environment versions absent | No |
| π0.5 RoboCasa pretrain checkpoint-75000 | same | official three-camera path and RoboCasa transform code; norm stats available; π0.5 requires all 3 cameras | official `pi05` config is 32-D and 50-step; action semantics are not documented as the RoboCasa Dict fields in the checkpoint; training environment versions and exact task clean reference absent | No |
| Diffusion Policy RoboCasa checkpoint epoch=0500 | `robocasa-benchmark/diffusion_policy`, HF official checkpoint | official RoboCasa evaluation runner, 8-step action execution, and task/split CLI | checkpoint is trained/evaluated on `pretrain` in the official benchmark; exact action frame/normalization and all three-camera/state mapping require loading the checkpoint config; 1.7 GB file was not downloaded; training environment versions and exact target-task clean reference absent | No |

## GR00T evidence details

The official GR00T `PandaOmronDataConfig` defines exactly the local camera keys,
state keys, action keys, and language key. It normalizes continuous fields with
`min_max`, gripper/control with `binary`, and converts the two quaternion state
fields to `rotation_6d`. Its transform is `VideoToTensor`, a 0.95 center crop in
evaluation mode, bilinear resize to `224x224`, and no color jitter in evaluation
mode. The modality config uses one observation frame and 16 action indices.

The local RoboCasa source provides the remaining action proof. Its official
`PandaOmron_modality.json` defines the five action fields and their native
dimensions/slices. The runtime flat action order is specified independently by
`robocasa.utils.env_utils.convert_action`: end-effector position (3),
end-effector axis-angle rotation (3), gripper (1), base motion (4), and control
mode (1). The GR00T `ConcatTransform` uses that same action-key order and splits
the model output back into these fields, discarding only the documented padding
after the first 12 dimensions. RoboCasa `PandaOmronKeyConverter.unmap_action`
then maps these fields to the arm, gripper, base, torso, and base-mode
controller inputs.

The controller frame is also explicit: official
`default_pandaomron.json` sets the arm controller to `OSC_POSE` with
`input_type="delta"` and `input_ref_frame="base"`; the base uses joint
velocity and the torso uses joint position. Thus the adapter must emit the
normalized action fields above, in this order, without guessing a world-frame
conversion.

The checkpoint config is explicit about the model padding:
`action_dim=32` and `action_horizon=16`. The model output is therefore not a
native RoboCasa action vector; the official transform pipeline's first-12
dimension split is part of the required adapter contract. The official
checkpoint API reports two safetensor shard LFS SHA-256 identifiers, but those
are per-file identifiers and are not a complete checkpoint-directory SHA256.

### Source-anchored evidence

| Contract field | Official evidence | Status |
|---|---|---|
| task and target split | RoboCasa registry has `PickPlaceCounterToCabinet.target`; GR00T CLI accepts `--split target`; HF README reports the same task under the legacy `PnPCounterToCabinet` label | Resolved; legacy label is recorded as an alias |
| embodiment and frequency | RoboCasa `PandaOmron_embodiment.json`: PandaOmron, 20 Hz; GR00T `PandaOmronDataConfig` and checkpoint metadata use the same embodiment family | Resolved |
| cameras and image preprocessing | Three `robot0_*` cameras at 256x256x3/20 Hz; GR00T evaluation transform is center-crop 0.95, bilinear 224x224, no color jitter | Resolved |
| state keys and normalization | `PandaOmronDataConfig`, `PandaOmron_modality.json`, checkpoint `experiment_cfg/metadata.json`; quaternion state fields become rotation-6D | Resolved |
| action keys, dimensions, range | RoboCasa `convert_action`, `PandaOmron_modality.json`, GR00T `ConcatTransform`; continuous fields min-max to [-1,1], gripper/control binary | Resolved |
| frame and control mode | `default_pandaomron.json` plus `PandaOmronKeyConverter.unmap_action`; arm is base-frame delta OSC pose, control mode selects base mode by threshold | Resolved |
| chunk length | GR00T config and data config: 16 action steps; official evaluator defaults to executing 16 action steps | Resolved |
| prompt format | `annotation.human.task_description`; GR00T inference formalizes language only when configured, and the RoboCasa data config uses the task description field | Resolved |
| training environment | Official GR00T commit `9d7d7a9...`, `pyproject.toml` (GR00T 1.1.0, Transformers 4.51.3), Dockerfile (PyTorch 2.6 CUDA 12.4 base, final pinned PyTorch 2.5.1/torchvision 0.20.1/NumPy 1.26.4), and checkpoint config (Transformers 4.51.3, bfloat16) | Source-reproducible specification; checkpoint metadata does not embed a lockfile |
| clean evaluation reference | Official target-fraction README and `eval_results.json`: 30 held-out test rollouts, `PickPlaceCounterToCabinet` 60.0% | Resolved as an official aggregate reference; seeds are not published |
| complete checkpoint SHA256 | Both official model shards; deterministic hash over sorted `sha256  filename` records is recorded in the manifest | Resolved: per-file hashes and checkpoint-set hash are verified locally |

The official `target_fraction/atomic_seen_30p` README and `eval_results.json` do
provide a 30-rollout reference for the named pick-and-place task, but it is an
aggregate test protocol and does not identify the requested fixed seeds 0..29.
Both local files match the official Hub LFS hashes. The manifest additionally
records a deterministic SHA256 over the sorted `sha256  filename` records.

## Decision

The RoboCasa/GR00T interface, artifact provenance, and frozen clean-baseline
gates passed. The no-fault run on target seeds 0..29 completed 30 episodes with
19 successes (63.33%), versus the official 18/30 aggregate reference whose
seeds were not published. Mean policy inference latency was 115.23 ms over 806
inferences. The model stayed in evaluation mode with all parameters frozen; its
parameter SHA256 was
`facb9d875a6e590e429bc2724b31d6af9f6346b36db825898acbe4ef3a364a08`
both before and after evaluation.

The adapter explicitly saturates generated values to the probed Gym action
bounds before stepping RoboCasa, matching the downstream controller limits
instead of relying on implicit clipping. It records every raw range and
saturation event in the episode JSONL. Across the baseline, 637 scalar values
were saturated; the raw global range was approximately [-1.03067, 1.04038]. No
fault was enabled. The project is now eligible to start separately configured
fault experiments, while preserving the clean output as immutable provenance.
If the missing training-environment and per-task clean-reference evidence is
required as a hard gate, the alternative is to train a RoboCasa-specific policy
and record its full provenance.

Official sources:

- [RoboCasa official repository](https://github.com/robocasa/robocasa)
- [RoboCasa policy benchmark documentation](https://github.com/robocasa/robocasa/blob/main/docs/benchmarking/policy_learning_algorithms.md)
- [Official RoboCasa GR00T fork](https://github.com/robocasa-benchmark/Isaac-GR00T)
- [Official RoboCasa OpenPI fork](https://github.com/robocasa-benchmark/openpi)
- [Official RoboCasa Diffusion Policy fork](https://github.com/robocasa-benchmark/diffusion_policy)
- [RoboCasa official checkpoint repository](https://huggingface.co/robocasa/robocasa365_checkpoints)

Pinned evidence files:

- [GR00T `PandaOmronDataConfig`](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/gr00t/experiment/data_config.py)
- [GR00T action/video transforms](https://github.com/robocasa-benchmark/Isaac-GR00T/tree/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/gr00t/data/transform)
- [GR00T policy inference and unapply path](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/gr00t/model/policy.py)
- [GR00T RoboCasa evaluator](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10/scripts/run_eval.py)
- [RoboCasa task registry](https://github.com/robocasa/robocasa/blob/a07e365c958c4216cd6bbd5f30b47f09a65c6f00/robocasa/utils/dataset_registry.py)
- [RoboCasa action conversion](https://github.com/robocasa/robocasa/blob/a07e365c958c4216cd6bbd5f30b47f09a65c6f00/robocasa/utils/env_utils.py)
- [RoboCasa PandaOmron Gym key converter](https://github.com/robocasa/robocasa/blob/a07e365c958c4216cd6bbd5f30b47f09a65c6f00/robocasa/wrappers/gym_wrapper.py)
- [RoboSuite PandaOmron controller](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/controllers/config/robots/default_pandaomron.json)
- [GR00T target-fraction 30% README](https://huggingface.co/robocasa/robocasa365_checkpoints/blob/main/gr00t_n1-5/target_fraction/atomic_seen_30p/README.md)
- [GR00T target-fraction evaluation results](https://huggingface.co/robocasa/robocasa365_checkpoints/blob/main/gr00t_n1-5/target_fraction/atomic_seen_30p/eval_results.json)
- [GR00T target-fraction checkpoint files](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/target_fraction/atomic_seen_30p)
