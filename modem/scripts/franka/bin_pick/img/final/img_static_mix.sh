#!/bin/bash


python train.py  -m \
    task=franka-FrankaBinPick_v2d  \
    exp_name=bin_pick_img-final-static-mix \
    iterations=1\
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=10 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[0,0] \
    top_crops=[0,0] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=1.0 \
    val_std_w=-10.00 \
    mix_schedule='"linear(0.0,1.0,2500,5000)"' \
    mixture_coef=0.5\
    save_freq=2500\
    eval_freq=2500\
    min_std=0.1\
    uncertainty_weighting=false\
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    hydra/launcher=slurm

