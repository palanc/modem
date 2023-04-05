#!/bin/bash

#<<test
#python train.py \
#    suite=franka \
#    task=franka-FrankaBinPushReal_v2d  \
#    exp_name=real_test \
#    discount=0.95 \
#    train_steps=100000 \
#    seed=8 \
#    demos=3 \
#    img_size=224 \
#    lr=3e-4 \
#    batch_size=64 \
#    episode_length=100 \
#    camera_views=[top_cam] \
#    left_crops=[116] \
#    top_crops=[16] \
#    real_robot=true \
#    action_repeat=1 \
#    plan_policy=true \
#    bc_rollout=true \
#    bc_q_pol=true \
#    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
#    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
#    seed_steps=300 \
#    eval_episodes=3 \
#    h5_demos=true \
#    success_mask_left=108 \
#    success_mask_right=122 \
#    success_mask_top=109 \
#    success_mask_bottom=143 \
#    success_thresh=0.5 \
#    success_uv=[-52,68] \

#   #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/real_bc/half_bc_seed' \
#test


python train.py \
    suite=franka \
    task=franka-FrankaBinPushReal_v2d  \
    exp_name=real_bin_push_100_demos2 \
    discount=0.95 \
    train_steps=100000 \
    seed=2 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=128 \
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
    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
    eval_freq=5000\
    h5_demos=true \
    success_mask_left=108 \
    success_mask_right=122 \
    success_mask_top=109 \
    success_mask_bottom=143 \
    success_thresh=0.5 \
    success_uv=[-52,68] \

    #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/real_bc/half_bc_seed' \

