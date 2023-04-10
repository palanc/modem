#!/bin/bash

<<test
python train.py \
    task=franka-FrankaBinPushReal_v2d  \
    exp_name=real_bin_push_test \
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    camera_views=[top_cam] \
    left_crops=[116] \
    top_crops=[16] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='//mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinPushReal_v2d/bin_push_real_bc_100demos/bc_seed' \
    seed_steps=300 \
    eval_episodes=3 \
    success_mask_left=108 \
    success_mask_right=122 \
    success_mask_top=109 \
    success_mask_bottom=143 \
    success_thresh=0.5 \
    success_uv=[-52,68] 

test

python train.py \
    task=franka-FrankaBinPushReal_v2d  \
    exp_name=real_bin_push_100_demos \
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    save_freq=1000 \
    camera_views=[top_cam] \
    left_crops=[116] \
    top_crops=[16] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    seed_steps=5000 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinPushReal_v2d/bin_push_real_bc_100demos/bc_seed' \
    eval_freq=5000\
    success_mask_left=108 \
    success_mask_right=122 \
    success_mask_top=109 \
    success_mask_bottom=143 \
    success_thresh=0.5 \
    success_uv=[-52,68] 

com