#!/bin/bash

python train.py \
    task=franka-FrankaPlanarPushReal_v2d  \
    suite=franka \
    exp_name=real_bc_planar_push_eval \
    seed=1 \
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
    logging_dir='/mnt/raid5/plancaster/data/robohive_base' \
    demo_dir='/mnt/raid5/plancaster/data/robohive_base' \
    bc_model_fp='/mnt/raid5/plancaster/data/robohive_base/franka-FrankaPlanarPushReal_v2d/planar_push_real_bc_100demos/bc_seed' \
    success_mask_left=50 \
    success_mask_right=140 \
    success_mask_top=105 \
    success_mask_bottom=145 \
    success_thresh=0.2 \
    success_uv=[0,0.55] 
