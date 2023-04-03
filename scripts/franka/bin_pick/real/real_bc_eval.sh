#!/bin/bash

python train.py \
    task=franka-FrankaBinPickReal_v2d  \
    suite=franka \
    exp_name=real_bc_smooth_eval-1000demosb \
    seed=120000 \
    demos=0 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[115,155] \
    top_crops=[0,0] \
    bc_only=true \
    real_robot=true \
    action_repeat=1 \
    min_std=0.05\
    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
    bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth-3seed/bc_' \
    h5_demos=true \
    
    
    
    #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/half_bc/real_bc_seed'
