#!/bin/bash

python train.py -m \
    task=franka-FrankaBinPick_v2d  \
    exp_name=bin_pick_img \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=1000 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    episode_length=100 \
    left_crops=[0,0] \
    top_crops=[0,0] \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    hydra/launcher=slurm


<<test

python train.py \
    task=franka-FrankaBinPick_v2d  \
    exp_name=test \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=8 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    seed_steps=100 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    episode_length=100 \
    left_crops=[0,0] \
    top_crops=[0,0] \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \

test


