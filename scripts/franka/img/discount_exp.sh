#!/bin/bash

<<test
python train.py \
    task=franka-FrankaPickPlaceRandom_v2d  \
    suite=franka \
    exp_name=test \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=3 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    seed_steps=100 \
    eval_episodes=3 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1    
test

# Vary the discount factor
for i in 0.75 0.85 0.9 0.95 0.99
do
    python train.py -m \
        task=franka-FrankaPickPlaceRandom_v2d  \
        suite=franka \
        exp_name=img_discount_$i \
        discount=$i \
        train_steps=200000 \
        seed=1,2,3,4,5 \
        demos=1000 \
        img_size=224 \
        lr=3e-4 \
        batch_size=256 \
        camera_views=[left_cam,right_cam] \
        action_repeat=1 \
        hydra/launcher=slurm & 
done