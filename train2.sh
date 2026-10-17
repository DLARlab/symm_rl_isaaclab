#!/usr/bin/env bash

export LD_LIBRARY_PATH=/home/dlar58/anaconda3/envs/symm_rl_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export WANDB_API_KEY='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'
export WANDB_USERNAME="xchen168-syracuse-university"
# OMNI_KIT_ACCEPT_EULA=YES ./scripts/symm_locomotion/train.sh --robot x1 --iterations 10000 --num-envs 2048 --logger wandb --log_project_name Go2-symm-rl --tr-value-coef 0 --tr-policy-coef 0.5 env.policy_observation_history.history_length=20
# OMNI_KIT_ACCEPT_EULA=YES ./scripts/symm_locomotion/train.sh --robot x1 --iterations 10000 --num-envs 2048 --logger wandb --log_project_name Go2-symm-rl --no-trs env.policy_observation_history.history_length=20
# OMNI_KIT_ACCEPT_EULA=YES ./scripts/symm_locomotion/train.sh --robot go2 --iterations 10000 --num-envs 2048 --logger wandb --log_project_name Go2-symm-rl --tr-value-coef 0 env.policy_observation_history.history_length=20
# OMNI_KIT_ACCEPT_EULA=YES ./scripts/symm_locomotion/train.sh --robot go2 --iterations 15000 --num-envs 2048 --logger wandb --log_project_name Go2-symm-rl --no-trs env.policy_observation_history.history_length=20


# OMNI_KIT_ACCEPT_EULA=YES ./scripts/symm_locomotion/train.sh --gait bound --robot x1_go2bend --iterations 1500 --num-envs 2048 --logger wandb --log_project_name Go2-symm-rl --tr-value-coef 0 --tr-policy-coef 0.1 env.policy_observation_history.history_length=20
# OMNI_KIT_ACCEPT_EULA=YES ./scripts/symm_locomotion/train.sh --gait bound --robot x1_go2bend --iterations 1500 --num-envs 2048 --logger wandb --log_project_name Go2-symm-rl --tr-value-coef 0 --tr-policy-coef 0.5 env.policy_observation_history.history_length=20
OMNI_KIT_ACCEPT_EULA=YES ./scripts/symm_locomotion/train.sh --gait bound --robot x1_go2bend --iterations 1500 --num-envs 2048 --logger wandb --log_project_name Go2-symm-rl --no-trs env.policy_observation_history.history_length=20