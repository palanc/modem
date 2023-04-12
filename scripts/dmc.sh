
python train.py -m \
    task=walker-run,quadruped-run,humanoid-walk \
    suite=dmcontrol \
    demos=5 \
    exp_name=dmc \
    seed=1,2,3,4,5 \
    hydra/launcher=slurm \
    hydra.job.name=walk-quad-human \
    hydra.launcher.timeout_min=1200 \
    hydra.launcher.mem_gb=90