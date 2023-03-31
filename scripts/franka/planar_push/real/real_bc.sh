#!/bin/bash
<<test
python train.py \
    task=franka-FrankaPlanarPushReal_v2d  \
    suite=franka \
    exp_name=test_planar_push_bc \
    seed=1 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[right_cam] \
    left_crops=[50] \
    top_crops=[0] \
    bc_only=true \
    h5_demos=true
test

# Train bc policies for real 
python train.py -m \
    task=franka-FrankaPlanarPushReal_v2d  \
    suite=franka \
    exp_name=planar_push_real_bc \
    seed=1,2,3,4,5,6,7,8,9,10 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[right_cam] \
    left_crops=[50] \
    top_crops=[0] \
    bc_only=true \
    h5_demos=true \
    hydra/launcher=slurm &

