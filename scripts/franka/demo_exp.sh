#!/bin/bash

# Vary the number of demos
for i in 1 5 10 100 1000
do
    python train.py -m \
        task=franka-FrankaPickPlaceRandom \
        suite=franka \
        exp_name="demo_exp$i" \
        seed=1,2,3,4,5 \
        demos=$i \
        hydra/launcher=slurm &
done