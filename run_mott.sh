#!/usr/bin/env bash
# MoTT vs GridPolicy (+ the same physics prior) on two GPUs, then certified evaluation.
#   0) smoke_mott.py  : tiny end-to-end check (pools, Woodbury vs direct, models, train, eval)
#   1) build_pool.py  : certified problem pools (CPU, scipy; skips files that already exist)
#   2) train.py       : one model per GPU, wall-clock limited by HOURS
#   3) eval_mott.py   : every method inside cutting_plane, success = exact dense check
# Usage: HOURS=6 bash run_mott.sh
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p runs
HOURS=${HOURS:-6}

# 0) smoke test: tiny end-to-end run on the GPU (about a minute); stops here if anything is broken
python smoke_mott.py --device cuda

python build_pool.py --out runs/pool 2>&1 | tee -a runs/pool.log

COMMON="--pool runs/pool --val_pools test_pcb10_0 --d 128 --layers 4 --heads 8 --B 16 --M 8 --lr 3e-4 --hours $HOURS --iters 100000 --val_every 100"
CUDA_VISIBLE_DEVICES=0 python train.py $COMMON --model mott --out runs/mott > runs/mott.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON --model grid --feat phys --prior --out runs/grid_prior > runs/grid_prior.log 2>&1 &
wait

python eval_mott.py --runs runs/mott,runs/grid_prior --ckpt best.pt --out reports/mott_result.md | tee runs/eval.log
