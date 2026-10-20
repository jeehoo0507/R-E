#!/usr/bin/env bash
# Main model + baselines/ablations for the revised hypotheses, then evaluation.
# Run sequentially on one GPU. Each run writes runs/<name>/{config.json,log.csv,best.pt,last.pt}.
set -e
ITERS=${ITERS:-5000}

python train.py --out runs/gnn_phys     --iters $ITERS                    # main: physics features + add/remove/stop
python train.py --out runs/gnn_coord    --iters $ITERS --feat coord       # H1': coordinate input (Park / DevFormer style)
python train.py --out runs/gnn_k3       --iters $ITERS --layers 3         # H2': message-passing depth
python train.py --out runs/gnn_k12      --iters $ITERS --layers 12
python train.py --out runs/gnn_glob     --iters $ITERS --glob             # + global self-attention
python train.py --out runs/gnn_noprune  --iters $ITERS --no_prune         # H3': add + stop only
python train.py --out runs/gnn_noimit   --iters $ITERS --imit 0           # pure RL, no best-known imitation
python train.py --out runs/gnn_lsteach  --iters $ITERS --teacher ls       # stronger teacher (local search)

python evaluate.py --sizes 10,15,20,30,50 --n_inst 16 --out results \
  --ckpt gnn=runs/gnn_phys/best.pt coord=runs/gnn_coord/best.pt k3=runs/gnn_k3/best.pt \
         k12=runs/gnn_k12/best.pt glob=runs/gnn_glob/best.pt noprune=runs/gnn_noprune/best.pt \
         noimit=runs/gnn_noimit/best.pt lsteach=runs/gnn_lsteach/best.pt
