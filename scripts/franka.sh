python train.py -m \
    task=franka-FrankaPickPlaceRandom_v2d  \
    suite=franka \
    exp_name=franka-default \
    discount=0.95 \
    train_steps=200000 \
    seed=1,2,3,4,5 \
    demos=100 \
    img_size=224 \
    lr=3e-4 \
    batch_size=256 \
    episode_length=100 \
    hydra/launcher=slurm