#!/bin/bash

python train.py \
    task=franka-FrankaPlanarPushReal_v2d  \
    suite=franka \
    exp_name=real_bc_smooth_eval-1000demosb \
    seed=120000 \
    demos=0 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[right_cam] \
    left_crops=[50] \
    top_crops=[0] \
    bc_only=true \
    real_robot=true \
    action_repeat=1 \
    min_std=0.05\
    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
    bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth-3seed/bc_' \
    h5_demos=true \
    left_success_mask=50 \
    right_success_mask=140 \
    top_success_mask=105 \
    bottom_success_mask=145 \
    success_thresh=0.2 \ 
    
    #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/half_bc/real_bc_seed'
