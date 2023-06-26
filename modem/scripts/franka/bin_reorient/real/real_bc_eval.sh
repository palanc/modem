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
    left_crops=[72,102] \
    top_crops=[0,0] \
    bc_only=true \
    real_robot=true \
    action_repeat=1 \
    val_min_w=0.0 \
    val_mean_w=1.0 \
    val_std_w=-10.00 \
    mix_schedule='"linear(0.0,1.0,5000,105000)"' \
    mixture_coef=1.0\
    min_std=0.05\
    depth_success_thresh=45\
    reorient_pickup_height=0.95\
    reorient_drop_goal_x=0.63\
    reorient_drop_goal_y=0.0\
    reorient_knock_height=1.12\
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d/bin_reorient_ensemble_bc_100demos/bc_seed' 