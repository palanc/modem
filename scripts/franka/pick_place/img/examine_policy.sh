#!/bin/bash

python examine_policy.py \
    task=franka-FrankaPickPlaceRandom_v2d  \
    suite=franka \
    exp_name=bc_eval \
    discount=0.95 \
    train_steps=1000 \
    seed=1 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    eval_episodes=3 \
    plan_policy=false \
    bc_rollout=false \
    bc_q_pol=false \
    bc_model_fp=/checkpoint/plancaster/outputs/modem/logs/franka-FrankaPickPlaceRandom_v2d/policy_plan_exp/1/models/0.pt    

