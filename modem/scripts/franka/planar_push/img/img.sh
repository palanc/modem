#!/bin/bash

<<test
python train.py \
    task=franka-FrankaPlanarPush_v2d  \
    exp_name=test \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=8 \
    camera_views=[right_cam] \
    action_repeat=1 \
    seed_steps=100 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    episode_length=50 \
    left_crops=[0] \
    top_crops=[0] \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base'
test

python train.py -m \
    task=franka-FrankaPlanarPush_v2d  \
    exp_name=planar_push_img \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=1000 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    camera_views=[right_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    episode_length=50 \
    left_crops=[0] \
    top_crops=[0] \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    hydra/launcher=slurm

