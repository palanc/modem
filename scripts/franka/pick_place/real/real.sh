#!/bin/bash
<<test
python train.py \
    suite=franka \
    task=franka-FrankaPickPlaceRandomReal_v2d  \
    exp_name=real_test \
    discount=0.95 \
    train_steps=100000 \
    seed=8 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
    bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/real_bc/half_bc_seed' \
    seed_steps=300 \
    eval_episodes=3
test

python train.py \
    suite=franka \
    task=franka-FrankaPickPlaceRandomReal_v2d  \
    exp_name=real_policy_smooth-centered \
    discount=0.95 \
    train_steps=100000 \
    seed=120000 \
    demos=1000 \
    img_size=224 \
    lr=3e-4 \
    batch_size=64 \
    episode_length=100 \
    save_freq=1000 \
    camera_views=[left_cam,right_cam] \
    real_robot=true \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    seed_steps=5000 \
    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
    bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth-3seed/bc_' \
    eval_freq=5000

    #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/real_bc/half_bc_seed' \