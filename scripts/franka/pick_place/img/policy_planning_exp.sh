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
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    seed_steps=100 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    bc_model_fp=/checkpoint/plancaster/outputs/modem/models/img/seed    
test

python train.py -m \
    task=franka-FrankaPickPlaceRandom_v2d  \
    suite=franka \
    exp_name=policy_plan_exp \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=1000 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    camera_views=[left_cam,right_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    bc_model_fp=/checkpoint/plancaster/outputs/modem/models/img/seed \
    save_freq=2000 \
    hydra/launcher=slurm
