# AutoDL deployment

## Expected instance

- NVIDIA GPU with 24 GB VRAM or more;
- 32 GB RAM minimum, 64 GB preferred;
- at least 100 GB free under `/root/autodl-tmp`;
- official Ubuntu/PyTorch image;
- no Docker and no driver installation inside the instance.

## Clone and install

```bash
cd /root/autodl-tmp
git clone <repository-url> VLA/recovery-bench
cd VLA/recovery-bench
bash scripts/setup_autodl.sh
```

The script creates the RoboCasa Conda environment and caches on the data disk.
It downloads only the required kitchen assets, not the full demonstration set.

## Acceptance test

```bash
source /root/autodl-tmp/VLA/env.sh
conda activate /root/autodl-tmp/VLA/envs/robocasa
cd /root/autodl-tmp/VLA/recovery-bench

python scripts/smoke_test_robocasa.py \
  --output /root/autodl-tmp/VLA/outputs/smoke_test_run1
python scripts/smoke_test_robocasa.py \
  --output /root/autodl-tmp/VLA/outputs/smoke_test_run2
```

Both runs must report `status: passed`, and both output images must have non-zero
standard deviation.

## Persistence

Code should be synchronized through Git. Large assets, checkpoints, and rollout
outputs stay on `/root/autodl-tmp`. Important source changes and result summaries
should also be backed up because an AutoDL local data disk is not redundant.
