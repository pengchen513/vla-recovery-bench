# SmolVLA integration gate

Do not connect `lerobot/smolvla_base` directly to RoboCasa and interpret the
result as a policy benchmark. A valid frozen checkpoint must define all of the
following for the chosen RoboCasa embodiment:

- camera names, ordering, resolution, and image preprocessing;
- proprioceptive state fields and normalization statistics;
- action dimensions, coordinate frame, control mode, and chunk length;
- language prompt format;
- training task distribution and published clean evaluation result.

The integration is accepted only when:

1. the checkpoint loads in a separate policy environment;
2. one observation can be converted without fabricated or silently dropped keys;
3. the returned action passes shape, finiteness, and range checks;
4. the policy reproduces a credible clean-task baseline before faults are enabled;
5. model parameters remain unchanged during evaluation.

The policy process and RoboCasa process should communicate through a small typed
request/response boundary. This avoids Python dependency conflicts and makes it
possible to measure policy latency separately from simulation time.

