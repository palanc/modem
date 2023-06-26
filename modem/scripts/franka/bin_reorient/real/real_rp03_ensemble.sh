#!/bin/bash

<<test
python train.py \
    task=franka-FrankaBinReorientRealRP03_v2d  \
    exp_name=real_bin_reorient_rp03_test \
    iterations=1\
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=32 \
    episode_length=150 \
    camera_views=[left_cam,right_cam] \
    left_crops=[72,147] \
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
    depth_success_thresh=40\
    reorient_pickup_height=0.925\
    reorient_drop_goal_x=0.61\
    reorient_drop_goal_y=0.0\
    reorient_knock_height=1.095\
    mix_schedule='"linear(0.0,1.0,0,300)"' \
    horizon_schedule='"linear(1, ${horizon}, 150)"'\
    mixture_coef=1.0\
    eval_episodes=2  \
    save_freq=300 \
    eval_freq=450 
test
#bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d/bin_reorient_real_bc_100demos/bc_seed' \

python train.py \
    task=franka-FrankaBinReorientRealRP03_v2d  \
    exp_name=bin_reorient_real_rp03-ensemble-noschedlim\
    iterations=1\
    discount=0.95 \
    train_steps=150000 \
    seed=3 \
    demos=10 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=150 \
    save_freq=1500 \
    camera_views=[left_cam,right_cam] \
    left_crops=[72,147] \
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
    mix_schedule='"linear(0.0,1.0,7500,207500)"' \
    mixture_coef=1.0\
    min_std=0.05\
    uncertainty_weighting=false\
    depth_success_thresh=40\
    reorient_pickup_height=0.925\
    reorient_drop_goal_x=0.61\
    reorient_drop_goal_y=0.0\
    reorient_knock_height=1.095\
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    eval_freq=7500 

    #bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d/bin_reorient_ensemble_bc_100demos/bc_seed4.pt'\
