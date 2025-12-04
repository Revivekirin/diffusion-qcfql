#!/bin/bash
#SBATCH --job-name=qc
#SBATCH --nodelist=pat-t6
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda activate qc

MUJOCO_GL=egl python run_sac.py \
 --run_group=reproduce  \
 --agent.alpha=100  \
 --env_name=square-mg-low  \
 --sparse=False \
 