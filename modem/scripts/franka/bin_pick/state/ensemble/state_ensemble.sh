#!/bin/bash

<<test
python train.py \
    task=franka-FrankaBinPick  \
    exp_name=test-ensemble-freeze \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=3 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    seed_steps=300 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    eval_freq=300 \
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=0.0 \
    val_std_w=-1.0 \
    mixture_coef=0.5\
    episode_length=100 \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base'
test

python train.py \
    task=franka-FrankaBinPick  \
    exp_name=bin_pick_state-ensemble-freeze \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=100 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=0.0 \
    val_std_w=-1.0 \
    mixture_coef=0.5\
    save_freq=2500\
    eval_freq=2500\
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' 

