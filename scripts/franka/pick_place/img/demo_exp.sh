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
    eval_episodes=3
test
# img_demo_exp: Vary the number of demos

# Vary the number of demos
for i in 0 1 5 10 100 1000
do
    python train.py -m \
        task=franka-FrankaPickPlaceRandom_v2d  \
        suite=franka \
        exp_name=img_demo_exp_demos$i \
        discount=0.95 \
        train_steps=200000 \
        seed=1,2,3,4,5 \
        demos=$i \
        img_size=224 \
        lr=3e-4 \
        batch_size=256 \
        camera_views=[left_cam,right_cam] \
        action_repeat=1 \
        hydra/launcher=slurm & 
done