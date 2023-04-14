#!/bin/bash

<<test
python train.py \
    task=franka-FrankaBinPush  \
    exp_name=test_value_ensemble_bin_push \
    discount=0.95 \
    train_steps=200000 \
    seed=1 \
    demos=3 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    episode_length=50 \
    camera_views=[top_cam] \
    action_repeat=1 \
    seed_steps=300 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    eval_freq=900 \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    value_ensemble_size=5 \
    value_std_wgt=1.0
test

python train.py -m \
    task=franka-FrankaBinPush  \
    exp_name=bin_push_state_val_ens_neg_100 \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=1000 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    episode_length=50 \
    camera_views=[top_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    logging_dir='/checkpoint/plancaster/outputs/robohive_base' \
    demo_dir='/checkpoint/plancaster/outputs/robohive_base' \
    value_ensemble_size=5 \
    value_std_wgt=-100.0 \
    eval_freq=2500 \
    save_freq=2500 \
    hydra/launcher=slurm

