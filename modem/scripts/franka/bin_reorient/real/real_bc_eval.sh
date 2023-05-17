#!/bin/bash
# 7,9,2,1,6,10,8,4,5,3
# 5,3
python train.py \
    task=franka-FrankaBinReorientReal_v2d  \
    exp_name=bin_reorient_bc_eval_100demo \
    seed=4 \
    demos=0 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=150 \
    camera_views=[left_cam,right_cam] \
    left_crops=[72,55] \
    top_crops=[0,0] \
    bc_only=true \
    real_robot=true \
    action_repeat=1 \
    min_std=0.05\
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d/bin_reorient_real_bc_100demos/bc_seed' 