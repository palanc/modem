#!/bin/bash
<<test
python train.py \
    task=franka-FrankaPlanarPush_v2d  \
    suite=franka \
    exp_name=test \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    camera_views=[right_cam] \
    action_repeat=1 \
    seed_steps=100 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true 
test

python train.py -m \
    task=franka-FrankaPlanarPush_v2d  \
    suite=franka \
    exp_name=planar_push_img_smooth \
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
    save_freq=2000 \
    hydra/launcher=slurm
    