#!/bin/bash

python train.py \
    task=franka-FrankaBinPickRealRP05_v2d  \
    exp_name=bin_pick_rp05_bc_eval_100demo \
    seed=1 \
    demos=0 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[70,90] \
    top_crops=[16,16] \
    bc_only=true \
    real_robot=true \
    action_repeat=1 \
    min_std=0.05\
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinPickRealRP05_v2d/bin_pick_100demos_rp05_models/bc_seed' 
    
