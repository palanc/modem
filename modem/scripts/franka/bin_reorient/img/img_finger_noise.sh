#!/bin/bash


<<test
python train.py \
    task=franka-FrankaBinReorient_v2d  \
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
    seed_steps=300 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    episode_length=150 \
    left_crops=[0,0] \
    top_crops=[0,0] \
    finger_noise=0.2 \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' 

test

python train.py -m \
    task=franka-FrankaBinReorient_v2d  \
    exp_name=bin_reorient_img-finger-noise \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    episode_length=150 \
    left_crops=[0,0] \
    top_crops=[0,0] \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    seed_steps=7500\
    eval_freq=2500\
    finger_noise=0.2\
    hydra/launcher=slurm
