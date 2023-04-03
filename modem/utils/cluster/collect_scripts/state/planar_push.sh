#!/bin/bash

python launch_rollouts.py \
    env_name=FrankaPlanarPush-v0 \
    mode=exploration \
    seed=0 \
    output_dir=/checkpoint/plancaster/outputs/robohive_base/demonstrations/franka-FrankaPlanarPush \
    num_rollouts=1000 \
    policy_path=/private/home/plancaster/robohive_base/hand_dapg/dapg/examples/planar_push_exp/iterations/best_policy.pickle
