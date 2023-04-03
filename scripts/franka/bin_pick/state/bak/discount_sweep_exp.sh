#!/bin/bash

<<test
python train.py \
    task=franka-FrankaPickPlaceRandom  \
    suite=franka \
    exp_name=test \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=2 \
    eval_episodes=3 \
    seed_steps=100
test

# demo_exp: Vary the number of demos and discount factor

# Vary the number of demos and discount factor
for i in 10 100
do
    for j in 0.5 0.75 0.85 0.9 0.95
    do
        python train.py -m \
            task=franka-FrankaPickPlaceRandom  \
            suite=franka \
            exp_name=discount_sweep_demos${i}_discount${j} \
            discount=$j \
            train_steps=200000 \
            seed=1,2,3,4,5 \
            demos=$i \
            hydra/launcher=slurm & 
    done
done