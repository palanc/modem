#!/bin/bash

# demo_exp: Vary the number of demos
# demo_exp2_demos$i: Decrease discount factor, include zero demos, increase train steps

# Vary the number of demos
for i in 0 1 5 10 100 1000
do
    python train.py -m \
        task=franka-FrankaPickPlaceRandom  \
        suite=franka \
        exp_name=demo_exp2_demos$i \
        discount=0.95 \
        train_steps=200000 \
        seed=1,2,3,4,5 \
        demos=$i \
        hydra/launcher=slurm & 
done