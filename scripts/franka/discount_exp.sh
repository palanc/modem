#!/bin/bash

# Try lower discount rate
python train.py -m \
    task=franka-FrankaPickPlaceRandom \
    suite=franka \
    exp_name="discnt_exp" \
    seed=1,2,3,4,5 \
    demos=1000 \
    discount=0.95 \
    hydra/launcher=slurm