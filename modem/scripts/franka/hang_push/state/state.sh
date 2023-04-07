#!/bin/bash

<<test
python train.py \
    task=franka-FrankaHangPush  \
    exp_name=test \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=3 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    episode_length=50 \
    camera_views=[top_cam] \
    action_repeat=1 \
    seed_steps=300 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    eval_freq=900 \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' 

test

python train.py -m \
    task=franka-FrankaHangPush  \
    exp_name=hang_push_state \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=1000 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    episode_length=50 \
    camera_views=[top_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    hydra/launcher=slurm


