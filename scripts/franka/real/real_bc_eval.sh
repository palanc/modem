#!/bin/bash

python train.py \
    task=franka-FrankaPickPlaceRandomReal_v2d  \
    suite=franka \
    exp_name=real_bc_eval \
    seed=1 \
    demos=0 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100
    camera_views=[left_cam,right_cam] \
    bc_only=true \
    real_robot=true \
    bc_model_fp=/mnt/nfs_code/robopen_users/plancaster/remodem/single_block/real_bc