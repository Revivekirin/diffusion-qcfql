#!/bin/bash
#SBATCH --job-name=qc
#SBATCH --nodelist=pat-t3
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda activate qc

# # teacher : transport/ph/bc_rnn
# MUJOCO_GL=egl python run_fql_bc_rnn.py \
#  --run_group=reproduce  \
#  --agent.alpha=100  \
#  --env_name=transport-ph-low  \
#  --sparse=False \
#  --horizon_length=1 \
#  --teacher_ckpt=/home/robros/git/robomimic/trained_model/transport_ph_low_dim_epoch_1000_succ_78.pth \
# #  --eval_interval=1

# # X teacher : qcfql
# MUJOCO_GL=egl python run_qcfql.py \
#  --run_group=reproduce  \
#  --agent.alpha=100  \
#  --env_name=transport-mh-low  \
#  --sparse=False \
#  --horizon_length=5 \
# #  --eval_interval=1


MUJOCO_GL=egl python run_qcfql_ptr_ver2.py \
 --run_group=reproduce  \
 --agent.alpha=100  \
 --env_name=transport-mg-low  \
 --sparse=False \
 --horizon_length=5 \
 --sparse=False \
 --priority_mode=chunk \ 
 