#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/root/autodl-tmp/VLA}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$REPO_ROOT/scripts/check_autodl.sh"

mkdir -p "$VLA_ROOT"/{src,envs,cache,outputs}

cat > "$VLA_ROOT/env.sh" <<'EOF'
export VLA_ROOT=/root/autodl-tmp/VLA
export HF_HOME="$VLA_ROOT/cache/huggingface"
export TORCH_HOME="$VLA_ROOT/cache/torch"
export XDG_CACHE_HOME="$VLA_ROOT/cache"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
fi
EOF

source /root/miniconda3/etc/profile.d/conda.sh
source "$VLA_ROOT/env.sh"
if [[ ! -d "$VLA_ROOT/envs/robocasa" ]]; then
  conda create -p "$VLA_ROOT/envs/robocasa" -c conda-forge python=3.11 -y
fi
conda activate "$VLA_ROOT/envs/robocasa"
python -m pip install --upgrade pip

if [[ ! -d "$VLA_ROOT/src/robosuite/.git" ]]; then
  git clone https://github.com/ARISE-Initiative/robosuite.git "$VLA_ROOT/src/robosuite"
fi
python -m pip install -e "$VLA_ROOT/src/robosuite"

if [[ ! -d "$VLA_ROOT/src/robocasa/.git" ]]; then
  git clone https://github.com/robocasa/robocasa.git "$VLA_ROOT/src/robocasa"
fi
python -m pip install -e "$VLA_ROOT/src/robocasa"
python -m pip install -e "$REPO_ROOT[robocasa]"

python "$VLA_ROOT/src/robosuite/robosuite/scripts/setup_macros.py"
python -m robocasa.scripts.download_kitchen_assets

python "$REPO_ROOT/scripts/write_setup_report.py" \
  --output "$REPO_ROOT/SETUP_REPORT.md" \
  --robosuite-repo "$VLA_ROOT/src/robosuite" \
  --robocasa-repo "$VLA_ROOT/src/robocasa"

echo "robosuite commit: $(git -C "$VLA_ROOT/src/robosuite" rev-parse HEAD)"
echo "robocasa commit:  $(git -C "$VLA_ROOT/src/robocasa" rev-parse HEAD)"
echo "Run the smoke test twice before installing a VLA policy."
