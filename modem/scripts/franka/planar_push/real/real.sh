#!/bin/bash

#<<test
#python train.py \
#    suite=franka \
#    task=franka-FrankaPlanarPushReal_v2d  \
#    exp_name=real_planar_push_test \
#    discount=0.95 \
#    train_steps=100000 \
#    seed=1 \
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
#    logging_dir='/mnt/raid5/plancaster/data/robohive_base' \
#    demo_dir='/mnt/raid5/plancaster/data/robohive_base' \
#    bc_model_fp='/mnt/raid5/plancaster/data/robohive_base/franka-FrankaBinPushReal_v2d/bin_push_real_bc_100demos/bc_seed' \
#    seed_steps=300 \
#    eval_episodes=3 \
#    success_left_mask=50 \
#    success_right_mask=140 \
#    success_top_mask=105 \
#    success_bottom_mask=145 \
#    success_thresh=0.2 \
#    success_uv=[0,0.55] 

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
    seed_steps=5000 \
    logging_dir='/mnt/raid5/plancaster/data/robohive_base' \
    demo_dir='/mnt/raid5/plancaster/data/robohive_base' \
    bc_model_fp='/mnt/raid5/plancaster/data/robohive_base/franka-FrankaPlanarPushReal_v2d/planar_push_real_bc_100demos/bc_seed' \
    eval_freq=5000\
    success_mask_left=50 \
    success_mask_right=140 \
    success_mask_top=105 \
    success_mask_bottom=145 \
    success_thresh=0.2 \
    success_uv=[0,0.55] \  

echo "Exiting script"

    #bc_model_fp='/mnt/nfs_code/robopen_users/plancaster/remodem/models/single_block/smooth/real_bc/half_bc_seed' \

