#!/bin/bash

<<test
python train.py \
    task=franka-FrankaPlanarPushReal_v2d  \
    exp_name=test_planar_push_bc \
    seed=1 \
    demos=30 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[right_cam] \
    left_crops=[50] \
    top_crops=[0] \
    bc_only=true \
    success_mask_left=50 \
    success_mask_right=140 \
    success_mask_top=105 \
    success_mask_bottom=145 \
    success_thresh=0.2 \
    success_uv=[0,0.55]
test

# Train bc policies for real 
python train.py -m \
    task=franka-FrankaPlanarPushReal_v2d  \
    exp_name=planar_push_real_bc_100demos \
    seed=1,2,3,4,5,6,7,8,9,10 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[right_cam] \
    left_crops=[50] \
    top_crops=[0] \
    bc_only=true \
    success_mask_left=50 \
    success_mask_right=140 \
    success_mask_top=105 \
    success_mask_bottom=145 \
    success_thresh=0.2 \
    success_uv=[0,0.55] \
    hydra/launcher=slurm &

