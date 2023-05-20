#!/bin/bash

<<test
python train.py \
    task=franka-FrankaBinPickReal_v2d  \
    exp_name=bin_pick_real-ensemble-test \
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=2 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    save_freq=1000 \
    camera_views=[left_cam,right_cam] \
    left_crops=[115,155] \
    top_crops=[0,0] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=1.0 \
    val_std_w=-10.00 \
    mix_schedule='"linear(0.0,1.0,100,200)"' \
    mixture_coef=1.0\
    min_std=0.1\
    seed_steps=200 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    eval_freq=200\
    eval_episodes=2 
test


python train.py \
    task=franka-FrankaBinPickReal_v2d  \
    exp_name=bin_pick_real-ensemble \
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    save_freq=1000 \
    camera_views=[left_cam,right_cam] \
    left_crops=[115,155] \
    top_crops=[0,0] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=1.0 \
    val_std_w=-10.00 \
    mix_schedule='"linear(0.0,1.0,5000,105000)"' \
    mixture_coef=1.0\
    min_std=0.1\
    seed_steps=5000 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    eval_freq=5000 

#bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinPickReal_v2d/bin_pick_real_bc_100demos/bc_seed'\