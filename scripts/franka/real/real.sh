#!/bin/bash

python train.py \
    task=franka-FrankaPickPlaceRandomReal_v2d  \
    suite=franka \
    exp_name=real \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=10 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100
    camera_views=[left_cam,right_cam] \
    real_robot=true \
    bc_model_fp=/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/real_bc \
    logging_dir=/mnt/nfs_code/robopen_users/plancaster/remodem \
    demo_dir=/mnt/nfs_code/robopen_users/plancaster/remodem