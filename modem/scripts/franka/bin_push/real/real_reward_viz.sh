#!/bin/bash

python reward_viz.py \
    task=franka-FrankaBinPushReal_v2d  \
    exp_name=bin_push_reward_viz \
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
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
    eval_freq=5000\
    h5_demos=true \
    success_mask_left=108 \
    success_mask_right=122 \
    success_mask_top=109 \
    success_mask_bottom=143 \
    success_thresh=0.5 \
    success_uv=[-52,68] \
    hydra/launcher=local