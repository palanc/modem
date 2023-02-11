python train.py -m \
    task=franka-FrankaPickPlaceRandom \
    suite=franka \
    exp_name=franka-default \
    seed=1,2,3,4,5 \
    hydra/launcher=slurm