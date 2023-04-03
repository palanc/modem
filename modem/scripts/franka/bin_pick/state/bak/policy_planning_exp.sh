#!/bin/bash

python train.py -m \
    task=franka-FrankaPickPlaceRandom  \
    suite=franka \
    exp_name=test \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=3 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    seed_steps=1000 \
    eval_episodes=30 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    eval_freq=2000 \
    episode_length=100 \
    gt_rollout=true \
    mixture_coef=0.0 \
    num_samples=512 \
    num_elites=64 \
    num_cpu=64 \
    hydra/launcher=slurm
    
#bc_model_fp=/checkpoint/plancaster/outputs/modem/models/state/seed    

<<com
python train.py -m \
    task=franka-FrankaPickPlaceRandom  \
    suite=franka \
    exp_name=policy_plan_exp_state_smooth \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=1000 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    episode_length=100 \
    hydra/launcher=slurm
com
#bc_model_fp=/checkpoint/plancaster/outputs/modem/models/state/seed \
