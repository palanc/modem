#!/bin/bash

python launch_rollouts.py \
    env_name=FrankaBinPush-v0 \
    mode=evaluation \
    seed=0 \
    output_dir=/checkpoint/plancaster/outputs/robohive_base/demonstrations/franka-FrankaBinPush \
    num_rollouts=1000 \
    policy_path=/private/home/plancaster/robohive_base/hand_dapg/dapg/examples/bin_push_exp/iterations/best_policy.pickle
