#!/bin/bash

# Try larger horizon
python train.py -m \
    task=franka-FrankaPickPlaceRandom \
    suite=franka \
    exp_name="horizon_exp" \
    seed=1,2,3,4,5 \
    demos=1000 \
    discount=0.95 \
    train_steps=200000\
    horizon=10\
    horizon_schedule='"linear(1,10,50000)"'\
    hydra/launcher=slurm