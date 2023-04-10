#!/bin/bash

python train.py \
    task=franka-FrankaBinPushReal_v2d  \
    exp_name=real_bin_push_bc_eval_100demo \
    seed=1 \
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
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinPushReal_v2d/bin_push_real_bc_100demos/bc_seed' \
    success_mask_left=108 \
    success_mask_right=122 \
    success_mask_top=109 \
    success_mask_bottom=143 \
    success_thresh=0.5 \
    success_uv=[-52,68] 
