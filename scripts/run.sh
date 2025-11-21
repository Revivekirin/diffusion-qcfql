MUJOCO_GL=egl python run_qcfql.py \
 --run_group=reproduce  \
 --agent.alpha=100  \
 --env_name=lift-mh-low  \
 --sparse=False \
 --horizon_length=5 \
#  --eval_interval=1