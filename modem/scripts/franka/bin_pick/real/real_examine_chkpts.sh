#!/bin/bash

python exmine_chkpts.py \
    task=franka-FrankaBinPickReal_v2d  \
    exp_name=bin_pick_real \
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
    seed_steps=5000 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinPickReal_v2d/bin_pick_real_bc_100demos/bc_seed'\
    eval_freq=5000 
