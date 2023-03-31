#!/bin/bash

python train.py \
    task=franka-FrankaBinPushReal_v2d  \
    suite=franka \
    exp_name=real_bin_push_eval \
    seed=0 \
    demos=0 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[top_cam] \
    left_crops=[116] \
    top_crops=[16] \
    bc_only=true \
    real_robot=true \
    action_repeat=1 \
    min_std=0.05\
    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/franka-FrankaBinPushReal_v2d' \
    bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/bin_push/real_bin_push_seed' \
    h5_demos=true
    
    #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/half_bc/real_bc_seed'
