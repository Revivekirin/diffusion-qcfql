#!/bin/bash
#SBATCH --job-name=qc
#SBATCH --nodelist=pat-t3
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda activate qc_dp

MUJOCO_GL=egl python run_qcfql.py \
 --run_group=reproduce  \
 --agent.alpha=100  \
 --env_name=transport-mh-low  \
 --sparse=False \
 --horizon_length=5 \
#  --eval_interval=1