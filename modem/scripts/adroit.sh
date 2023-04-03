python train.py -m \
    task=adroit-door,adroit-hammer,adroit-pen \
    suite=adroit \
    exp_name=adroit-all \
    seed=1,2,3,4,5 \
    hydra/launcher=slurm