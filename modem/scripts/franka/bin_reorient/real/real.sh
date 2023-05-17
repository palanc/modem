#!/bin/bash
<<test
python train.py \
    task=franka-FrankaBinReorientReal_v2d  \
    exp_name=real_bin_reorient_test \
    discount=0.95 \
    train_steps=100000 \
    seed=4 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=150 \
    camera_views=[left_cam,right_cam] \
    left_crops=[72,55] \
    top_crops=[0,0] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d/bin_reorient_real_bc_100demos/bc_seed' \
    seed_steps=300 \
    eval_episodes=2  \
    save_freq=300
test

python train.py \
    task=franka-FrankaBinReorientReal_v2d  \
    exp_name=bin_reorient_real \
    discount=0.95 \
    train_steps=150000 \
    seed=4 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=150 \
    save_freq=1500 \
    camera_views=[left_cam,right_cam] \
    left_crops=[72,55] \
    top_crops=[0,0] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    seed_steps=7500 \
    logging_dir='/mnt/raid5/data/plancaster/robohive_base' \
    demo_dir='/mnt/raid5/data/plancaster/robohive_base' \
    bc_model_fp='/mnt/raid5/data/plancaster/robohive_base/models/franka-FrankaBinReorientReal_v2d/bin_reorient_real_bc_100demos/bc_seed4.pt'\
    eval_freq=7500 

