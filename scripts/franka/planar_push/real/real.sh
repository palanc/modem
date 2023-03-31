#!/bin/bash

#<<test
#python train.py \
#    suite=franka \
#    task=franka-FrankaPlanarPushReal_v2d  \
#    exp_name=real_test \
#    discount=0.95 \
#    train_steps=100000 \
#    seed=8 \
#    demos=3 \
#    img_size=224 \
#    lr=3e-4 \
#    batch_size=64 \
#    episode_length=100 \
#    camera_views=[right_cam] \
#    left_crops=[50] \
#    top_crops=[0] \
#    real_robot=true \
#    action_repeat=1 \
#    plan_policy=true \
#    bc_rollout=true \
#    bc_q_pol=true \
#    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
#    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
#    seed_steps=300 \
#    eval_episodes=3 \
#    h5_demos=true

#   #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/real_bc/half_bc_seed' \
#test

echo "Entered script"

python train.py \
    suite=franka \
    task=franka-FrankaPlanarPushReal_v2d  \
    exp_name=real_planar_push_100demo \
    discount=0.95 \
    train_steps=100000 \
    seed=1 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
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
    logging_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/modem' \
    demo_dir='/mnt/nfs_code/robopen_users/plancaster/remodem/' \
    eval_freq=5000\
    h5_demos=true

echo "Exiting script"

    #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/real_bc/half_bc_seed' \

