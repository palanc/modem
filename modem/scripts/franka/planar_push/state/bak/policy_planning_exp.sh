#!/bin/bash
<<test
python train.py \
    task=franka-FrankaPlanarPush  \
    suite=franka \
    exp_name=test \
    discount=0.95 \
    train_steps=200000 \
    seed=5 \
    demos=3 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    camera_views=[top_cam,Franka-wrist_cam] \
    action_repeat=1 \
    seed_steps=100 \
    eval_episodes=3 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true  
test

python train.py -m \
    task=franka-FrankaPlanarPush  \
    suite=franka \
    exp_name=planar_push_state_smooth \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=1000 \
    img_size=0 \
    lr=1e-3 \
    batch_size=512 \
    camera_views=[top_cam,Franka-wrist_cam] \
    action_repeat=1 \
    plan_policy=true \
    bc_rollout=true \
    bc_q_pol=true \
    hydra/launcher=slurm
