#!/bin/bash

<<test

python train.py \
    task=franka-FrankaBinPickRealRP05_v2d  \
    exp_name=test_rp05 \
    seed=1 \
    demos=30 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[70,90] \
    top_crops=[16,16] \
    bc_only=true 

test

# Train bc policies for real 
python train.py -m \
    task=franka-FrankaBinPickRealRP05_v2d  \
    exp_name=bin_pick_real_bc_1000demos_rp05 \
    seed=1,2,3,4,5,6,7,8,9,10 \
    demos=1000 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[70,90] \
    top_crops=[16,16] \
    bc_only=true \
    hydra/launcher=slurm
