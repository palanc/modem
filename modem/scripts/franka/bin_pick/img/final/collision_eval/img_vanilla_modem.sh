#!/bin/bash

python eval_agent.py  -m \
    task=franka-FrankaBinPick_v2d  \
    exp_name=bin_pick_img-final-vanilla-modem-safety-eval \
    iterations=1\
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=10 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    camera_views=[left_cam,right_cam] \
    left_crops=[0,0] \
    top_crops=[0,0] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    ensemble_size=2 \
    val_min_w=1.0 \
    val_mean_w=0.0 \
    val_std_w=0.00 \
    mix_schedule='"linear(0.0,1.0,0,2500)"' \
    mixture_coef=1.0\
    save_freq=2500\
    eval_freq=2500\
    uncertainty_weighting=false\
    vanilla_modem=true\
    min_std=0.1\
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    eval_dir='/checkpoint/plancaster/outputs/robohive_base/logs/franka-FrankaBinPick_v2d/bin_pick_img-final-vanilla-modem' \
    hydra/launcher=slurm