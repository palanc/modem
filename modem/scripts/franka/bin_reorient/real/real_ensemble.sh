#!/bin/bash

<<test
python train.py \
    task=franka-FrankaBinReorientReal_v2d  \
    exp_name=real_bin_reorient_test2 \
    iterations=1\
    discount=0.95 \
    train_steps=100000 \
    seed=20 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=32 \
    episode_length=150 \
    camera_views=[left_cam,right_cam] \
    left_crops=[72,102] \
    top_crops=[0,0] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    seed_steps=300 \
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=1.0 \
    val_std_w=-10.0 \
    depth_success_thresh=45\
    reorient_pickup_height=0.95\
    mix_schedule='"linear(0.0,1.0,0,300)"' \
    horizon_schedule='"linear(1, ${horizon}, 150)"'\
    mixture_coef=1.0\
    eval_episodes=2  \
    save_freq=300 \
    eval_freq=450 

#bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d/bin_reorient_real_bc_100demos/bc_seed' \

test

python train.py \
    task=franka-FrankaBinReorientReal_v2d  \
    exp_name=bin_reorient_real-ensemble-aggsched \
    iterations=1\
    discount=0.95 \
    train_steps=150000 \
    seed=4 \
    demos=10 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=150 \
    save_freq=1500 \
    camera_views=[left_cam,right_cam] \
    left_crops=[72,102] \
    top_crops=[0,0] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    seed_steps=7500 \
    ensemble_size=6 \
    val_min_w=0.0 \
    val_mean_w=1.0 \
    val_std_w=-10.0 \
    mix_schedule='"linear(0.0,1.0,7500,107500)"' \
    limit_mix_sched=false\
    mixture_coef=1.0\
    min_std=0.05\
    uncertainty_weighting=false\
    depth_success_thresh=45\
    reorient_pickup_height=0.95\
    reorient_drop_goal_x=0.63\
    reorient_drop_goal_y=0.0\
    reorient_knock_height=1.12\
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    eval_freq=7500 

    #bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d/bin_reorient_ensemble_bc_100demos/bc_seed4.pt'\
