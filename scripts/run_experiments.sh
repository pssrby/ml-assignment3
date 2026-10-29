#!/usr/bin/env bash
# Train the five ensemble members used in the report.
# Assumes the current conda environment is the project environment and that
# an NVIDIA GPU is visible.  Set SKIP_* variables or edit the list to run a
# subset.  Results go to outputs/runs/<name>.
set -uo pipefail
cd "$(dirname "$0")/.."

COMMON=(
  --data-dir data/CIFAR10
  --epochs 200
  --warmup-epochs 5
  --batch-size 128
  --lr 0.1
  --weight-decay 5e-4
  --label-smoothing 0.1
  --cutout 8
  --ema-decay 0.9998
  --amp bf16
  --workers 8
  --save-every 20
)

run() {
  local gpu=$1 name=$2 seed=$3
  shift 3
  echo "[launch] gpu=$gpu run=$name seed=$seed extra=$*"
  CUDA_VISIBLE_DEVICES="$gpu" python -u src/train.py \
    --output-dir "outputs/runs/$name" --seed "$seed" "${COMMON[@]}" "$@" \
    > "outputs/logs/$name.log" 2>&1
}

mkdir -p outputs/runs outputs/logs
pids=()

run 0 A_wrn28_10_cutmix 101 --mix cutmix &
pids+=($!)
run 1 B_wrn28_10_se_cutmix 202 --mix cutmix --se &
pids+=($!)
run 2 C_wrn28_10_se_mixup 303 --mix mixup --se &
pids+=($!)
run 3 D_wrn28_10_se_ra 404 --mix none --se --randaugment &
pids+=($!)
run 4 E_wrn28_10_ra_cutmix 505 --mix cutmix --randaugment &
pids+=($!)

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ $status -eq 0 ]]; then
  python src/ensemble.py --run-dirs \
    outputs/runs/A_wrn28_10_cutmix \
    outputs/runs/B_wrn28_10_se_cutmix \
    outputs/runs/C_wrn28_10_se_mixup \
    outputs/runs/D_wrn28_10_se_ra \
    outputs/runs/E_wrn28_10_ra_cutmix \
    --output-dir outputs/ensemble
  echo "all training and ensemble evaluation finished"
else
  echo "at least one training run failed; inspect outputs/logs" >&2
fi
exit "$status"
