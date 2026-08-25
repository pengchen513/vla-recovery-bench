#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/root/autodl-tmp"
MIN_VRAM_MIB=20000
MIN_RAM_MIB=30000
MIN_DISK_KIB=$((80 * 1024 * 1024))

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FAIL: nvidia-smi is unavailable" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "FAIL: expected AutoDL data disk at $DATA_ROOT" >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
RAM_MIB="$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)"
DISK_KIB="$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')"

echo "GPU: $GPU_NAME"
echo "VRAM: ${VRAM_MIB} MiB"
echo "RAM: ${RAM_MIB} MiB"
echo "Free data disk: $((DISK_KIB / 1024 / 1024)) GiB"
nvidia-smi --query-gpu=driver_version --format=csv,noheader

if (( VRAM_MIB < MIN_VRAM_MIB )); then
  echo "FAIL: at least ${MIN_VRAM_MIB} MiB VRAM is required before large downloads" >&2
  exit 1
fi
if (( RAM_MIB < MIN_RAM_MIB )); then
  echo "FAIL: at least ${MIN_RAM_MIB} MiB RAM is required" >&2
  exit 1
fi
if (( DISK_KIB < MIN_DISK_KIB )); then
  echo "FAIL: at least 80 GiB must be free on $DATA_ROOT" >&2
  exit 1
fi

echo "AutoDL preflight passed"

