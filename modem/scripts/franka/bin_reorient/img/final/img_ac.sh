#!/bin/bash

python train.py  -m \
    task=franka-FrankaBinReorient_v2d  \
    exp_name=bin_reorient_img-final-ac\
    iterations=1\
    discount=0.95 \
    train_steps=300000 \
    seed=1,2,3,4,5 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=150 \
    camera_views=[left_cam,right_cam] \
    left_crops=[0,0] \
    top_crops=[0,0] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    ensemble_size=2 \
    val_min_w=1.0 \
    val_mean_w=0.0 \
    val_std_w=-0.0 \
    mix_schedule='"linear(0.0,1.0,7500,107500)"' \
    mixture_coef=1.0\
    save_freq=2500\
    eval_freq=2500\
    seed_steps=7500 \
    min_std=0.1\
    uncertainty_weighting=false\
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    hydra/launcher=slurm