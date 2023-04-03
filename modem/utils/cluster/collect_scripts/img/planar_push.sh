#!/bin/bash

python launch_rollouts.py -m\
    env_name=FrankaPlanarPush_v2d-v0 \
    mode=evaluation \
    seed=0,100,200,300,400,500,600,700,800,900 \
    output_dir=/checkpoint/plancaster/outputs/robohive_base/demonstrations/franka-FrankaPlanarPush_v2d \
    num_rollouts=100 \
    policy_path=/private/home/plancaster/robohive_base/hand_dapg/dapg/examples/planar_push_exp/iterations/best_policy.pickle \
    hydra/launcher=slurm \
    hydra/output=slurm 
