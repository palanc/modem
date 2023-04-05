#!/bin/bash

<<test
python train.py \
    task=franka-FrankaBinPushReal_v2d  \
    suite=franka \
    exp_name=test \
    seed=1 \
    demos=30 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[top_cam] \
    left_crops=[116] \
    top_crops=[16] \
    bc_only=true \
    h5_demos=true \
    success_mask_left=108 \
    success_mask_right=122 \
    success_mask_top=109 \
    success_mask_bottom=143 \
    success_thresh=0.5 \
    success_uv=[-52,68] \ 
test

# Train bc policies for real 
python train.py -m \
    task=franka-FrankaBinPushReal_v2d  \
    suite=franka \
    exp_name=bin_push_real_bc \
    seed=1,2,3,4,5,6,7,8,9,10 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[top_cam] \
    left_crops=[116] \
    top_crops=[16] \
    bc_only=true \
    h5_demos=true \
    success_mask_left=108 \
    success_mask_right=122 \
    success_mask_top=109 \
    success_mask_bottom=143 \
    success_thresh=0.5 \
    success_uv=[-52,68] \
    hydra/launcher=slurm &
