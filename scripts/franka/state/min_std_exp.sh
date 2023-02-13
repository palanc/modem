#!/bin/bash

# Try allowing for less noise
python train.py -m \
    task=franka-FrankaPickPlaceRandom \
    suite=franka \
    exp_name="min_std_exp" \
    seed=1,2,3,4,5 \
    demos=1000 \
    discount=0.95 \
    train_steps=200000\
    min_std=0.01\
    hydra/launcher=slurm