#!/bin/bash

<<test
python launch_rollouts.py \
    env_name=FrankaBinPick_v2d-v0 \
    mode=evaluation \
    seed=0 \
    output_dir=/checkpoint/plancaster/outputs/robohive_base/demonstrations/franka-FrankaBinPick_v2d \
    num_rollouts=1\
    policy_path=robohive.envs.arms.bin_pick_v0.BinPickPolicy 
test

python launch_rollouts.py -m \
    env_name=FrankaBinPick_v2d-v0 \
    mode=exploration \
    seed=0,100,200,300,400,500,600,700,800,900 \
    output_dir=/checkpoint/plancaster/outputs/robohive_base/demonstrations/franka-FrankaBinPick_v2d \
    num_rollouts=100 \
    policy_path=robohive.envs.arms.bin_pick_v0.BinPickPolicy \
    hydra/launcher=slurm \
    hydra/output=slurm 
