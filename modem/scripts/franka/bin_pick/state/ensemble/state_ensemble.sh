#!/bin/bash

<<com
python train.py \
    task=franka-FrankaBinPick  \
    exp_name=test-ensemble-all \
    discount=0.95 \
    train_steps=200000 \
    seed=2 \
    demos=3 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    seed_steps=100 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    eval_freq=100 \
    ensemble_size=6 \
    val_min_w=1.0 \
    val_mean_w=0.0 \
    val_std_w=0.0 \
    final_unfreeze=300\
    mix_schedule='"linear(0.0,1.0,300,600)"' \
    mixture_coef=0.5\
    episode_length=100 \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base'
com

python train.py   \
    task=franka-FrankaBinPick  \
    exp_name=bin_pick_state-ensemble-prob-sched \
    iterations=1\
    discount=0.95 \
    train_steps=200000 \
    seed=6 \
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
    val_mean_w=1.0 \
    val_std_w=0.0 \
    final_unfreeze=50000\
    mix_schedule='"linear(0.0,1.0,50000,150000)"' \
    mixture_coef=0.5\
    save_freq=2500\
    eval_freq=2500\
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' 
    
    
    #hydra/launcher=slurm
