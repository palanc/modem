#!/bin/bash

python launch_rollouts.py \
    env_name=FrankaBinPick-v0 \
    mode=evaluation \
    seed=0 \
    output_dir=/checkpoint/plancaster/outputs/robohive_base/demonstrations/franka-FrankaBinPick \
    num_rollouts=1000 \
    policy_path=robohive.envs.arms.bin_pick_v0.BinPickPolicy
