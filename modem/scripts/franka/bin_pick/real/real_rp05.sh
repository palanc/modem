#!/bin/bash

<<test
python train.py \
    task=franka-FrankaBinPickRealRP05_v2d  \
    exp_name=real_bin_pick_rp05_test \
    discount=0.95 \
    train_steps=100000 \
    seed=8 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[70,90] \
    top_crops=[16,16] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/nfs_code/robopen_users/jaydv/bin_pick/modem/modem/utils' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinPickRealRP05_v2d/bin_pick_100demos_rp05_models/bc_seed'\
    seed_steps=300 \
    eval_episodes=3 
test

python train.py \
    task=franka-FrankaBinPickRealRP05_v2d  \
    exp_name=bin_pick_rp05_real \
    discount=0.95 \
    train_steps=100000 \
    seed=8 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    save_freq=1000 \
    camera_views=[left_cam,right_cam] \
    left_crops=[70,90] \
    top_crops=[16,16] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    seed_steps=5000 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/nfs_code/robopen_users/jaydv/bin_pick/modem/modem/utils' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinPickRealRP05_v2d/bin_pick_100demos_rp05_models/bc_seed'\
    eval_freq=5000  

