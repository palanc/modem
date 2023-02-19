#!/bin/bash

python train.py \
    task=franka-FrankaPickPlaceRandomReal_v2d  \
    suite=franka \
    exp_name=real_bc_eval \
    seed=8 \
    demos=0 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    bc_only=true \
    real_robot=true \
    action_repeat=1 \
    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
    bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/real_bc/real_bc_seed'
