#!/bin/bash

<<test
python train.py \
    task=franka-FrankaPlanarPushReal_v2d  \
    exp_name=real_planar_push_100demo-ensemble-test \
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=2 \
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
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=1.0 \
    val_std_w=-10.00 \
    mix_schedule='"linear(0.0,1.0,100,200)"' \
    mixture_coef=1.0\
    eval_freq=200\
    eval_episodes=2\
    min_std=0.1\
    uncertainty_weighting=false\
    seed_steps=200 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    success_mask_left=50 \
    success_mask_right=140 \
    success_mask_top=105 \
    success_mask_bottom=145 \
    success_thresh=0.2 \
    success_uv=[-0.1,0.55]
test

python train.py \
    task=franka-FrankaPlanarPushReal_v2d  \
    exp_name=real_planar_push_100demo-ensemble \
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=10 \
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
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=1.0 \
    val_std_w=-10.00 \
    mix_schedule='"linear(0.0,1.0,5000,105000)"' \
    mixture_coef=1.0\
    min_std=0.1\
    uncertainty_weighting=false\
    seed_steps=5000 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    eval_freq=5000\
    success_mask_left=50 \
    success_mask_right=140 \
    success_mask_top=105 \
    success_mask_bottom=145 \
    success_thresh=0.2 \
    success_uv=[-0.1,0.55]

    #bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaPlanarPushReal_v2d/planar_push_real_bc_100demos/bc_seed6.pt' \
