#!/bin/bash

python train.py \
    task=franka-FrankaBinPickRealRP03_v2d  \
    suite=franka \
    exp_name=bin_pick_rp03_bc_eval_100demo \
    seed=1 \
    demos=0 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[78,95] \
    top_crops=[0,0] \
    bc_only=true \
    real_robot=true \
    action_repeat=1 \
    min_std=0.05\
    logging_dir='/mnt/raid5/plancaster/data/robohive_base' \
    demo_dir='/mnt/raid5/plancaster/data/robohive_base' \
    bc_model_fp='/mnt/raid5/plancaster/data/robohive_base/franka-FrankaBinPickRealRP03_v2d/bin_pick_real_bc_100demos_rp03/bc_seed' 
    
