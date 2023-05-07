#!/bin/bash

<<test
python train.py \
    task=franka-FrankaPlanarPushReal_v2d  \
    exp_name=real_planar_push_test \
    discount=0.95 \
    train_steps=100000 \
    seed=6 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    camera_views=[right_cam] \
    left_crops=[50] \
    top_crops=[0] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaPlanarPushReal_v2d/planar_push_real_bc_100demos/bc_seed' \
    seed_steps=300 \
    eval_episodes=3 \
    success_mask_left=50 \
    success_mask_right=140 \
    success_mask_top=105 \
    success_mask_bottom=145 \
    success_thresh=0.2 \
    success_uv=[0,0.55] 

test

python train.py \
    task=franka-FrankaPlanarPushReal_v2d  \
    exp_name=real_planar_push_100demo \
    discount=0.95 \
    train_steps=100000 \
    seed=3 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    save_freq=1000 \
    camera_views=[right_cam] \
    left_crops=[50] \
    top_crops=[0] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    seed_steps=5000 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaPlanarPushReal_v2d/planar_push_real_bc_100demos/bc_seed6.pt' \
    eval_freq=5000\
    success_mask_left=50 \
    success_mask_right=140 \
    success_mask_top=105 \
    success_mask_bottom=145 \
    success_thresh=0.2 \
    success_uv=[0,0.55]


