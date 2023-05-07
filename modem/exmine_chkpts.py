# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env, set_seed
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, get_demos,get_demos_h5, ReplayBuffer, linear_schedule, do_policy_rollout, trace2episodes
from termcolor import colored
from copy import deepcopy
import logger
import hydra
from robohive.logger.grouped_datasets import Trace

torch.backends.cudnn.benchmark = True
import pickle

def evaluate(paths, env, agent, cfg, step, env_step, video, traj_plot, policy_rollout=False,rollout_agent=None,q_pol_agent=None):
    """Evaluate a trained agent and optionally save a video."""
    if rollout_agent is None:
        rollout_agent = agent
    if q_pol_agent is None:
        q_pol_agent = agent

    episodes = trace2episodes(cfg, env, paths, exclude_fails=False, is_demo=False)
    record_actions = []
    eval_actions = []
    record_states = []
    for episode in episodes:
        agent_actions = []      
        plan_step = step
        planner = agent.plan

        for t in range(cfg.episode_length):
            obs = []
            s_idx = 0#t - (cfg.frame_stack-1)
            for _ in range(cfg.frame_stack):
                if s_idx < 0:
                    obs.append(episode.obs[0])
                else:
                    obs.append(episode.obs[s_idx])
                s_idx += 1
            obs = np.concatenate(obs, axis=1)
            state = episode.state[t]

            horizon = int(min(cfg.horizon, linear_schedule(cfg.horizon_schedule, step)))
            mean = rollout_agent.rollout_actions(obs, state, horizon)
            std = cfg.min_std* torch.ones(horizon, cfg.action_dim, device=agent.device)   
            action = planner(obs, state, mean=mean, std=std, eval_mode=True, step=plan_step, t0=t == 0, q_pol=q_pol_agent.model.pi, gt_rollout_start_state= None if not cfg.gt_rollout else env.unwrapped.get_env_state())   

            agent_actions.append(action.cpu().numpy())

        record_states.append(episode.state)
        record_actions.append(episode.action)
        eval_actions.append(np.stack(agent_actions, axis=0))

    print('Act diff: {}'.format(np.linalg.norm(np.linalg.norm(np.stack(eval_actions,axis=0)-np.stack(record_actions,axis=0), axis=0),axis=0)))
    if traj_plot:
        rollout_states = record_states
        rollout_actions = record_actions
        episode_states = record_states
        episode_actions = eval_actions

        traj_plot.save_traj(rollout_states, rollout_actions, 
                            episode_states, episode_actions, 
                            env_step)

def load_checkpt_fp(cfg, checkpt_dir, bc_dir):
    model_fp = None
    step = 0

    if checkpt_dir is not None:
        while True:
            if os.path.exists(checkpt_dir / f'{str(step)}.pt'):
                model_fp = checkpt_dir / f'{str(step)}.pt'
            else:
                step -= cfg.save_freq
                step = max(0, step)
                break
            step += cfg.save_freq
        
    if model_fp is None:
        assert(step==0)

    if model_fp is None and cfg.get('bc_model_fp', None) is not None:
        if os.path.exists(cfg.bc_model_fp):
            model_fp = cfg.bc_model_fp
        elif os.path.exists(cfg.bc_model_fp+str(cfg.seed)+'.pt'):
            model_fp = cfg.bc_model_fp+str(cfg.seed)+'.pt'
        else:
            assert False, 'Could not find bc model file path'   

    bc_save_step = 0
    if model_fp is None:
        max_bc_steps = 2 * cfg.demos * cfg.episode_length
        bc_save_int = max(int(0.1*max_bc_steps), 10000)
        bc_save_step = bc_save_int
        # Check if there are existing bc models
        while True:
            if os.path.exists(bc_dir / f'bc_{str(bc_save_step)}.pt'):
                model_fp = bc_dir / f'bc_{str(bc_save_step)}.pt'
            else:
                bc_save_step -= bc_save_int
                bc_save_step = max(0, bc_save_step)
                break
            bc_save_step += bc_save_int  

    return model_fp, step, bc_save_step if bc_save_step > 0 else None 

@hydra.main(config_name="config", config_path="cfgs")
def train(cfg: dict):
    """Training script for online TD-MPC."""
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    work_dir = Path(cfg.logging_dir) / "logs" / cfg.task / cfg.exp_name / str(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), work_dir)

    episode_dir = str(cfg.logging_dir) + f"/episodes/{cfg.task}/{cfg.exp_name}" 
    if (not os.path.isdir(episode_dir)):
        os.makedirs(episode_dir)

    env, agent = make_env(cfg), TDMPC(cfg)
    demo_buffer = ReplayBuffer(deepcopy(cfg)) if cfg.get("demos", 0) > 0 else None
    buffer = ReplayBuffer(cfg)
    L = logger.Logger(work_dir, cfg)
    print(agent.model)

    model_fp, start_step, bc_start_step = load_checkpt_fp(cfg, L._model_dir, L._model_dir)
    assert(start_step == 0 or bc_start_step is None)
    if model_fp is not None:
        print('Loading agent '+str(model_fp))
        agent.load(model_fp)        
        agent.model.eval()
        agent.model_target.eval()     

    bc_agent = TDMPC(cfg)
    bc_model_fp = model_fp
    if bc_model_fp is not None and start_step //cfg.action_repeat >= cfg.seed_steps:
        bc_model_fp, _, _ = load_checkpt_fp(cfg, None, L._model_dir)
    if bc_model_fp is not None:
        print('BC model fp: {}'.format(bc_model_fp))
        bc_agent.load(bc_model_fp)
        bc_agent.model.eval()
        bc_agent.model_target.eval()
 
    # Load past episodes

    load_idx = start_step//cfg.episode_length + 1 
    trace_fn = 'rollout'+f'{(load_idx):010d}.pickle'
    trace_path = episode_dir+'/'+trace_fn   

    paths = Trace.load(trace_path)
    evaluate(paths, env, agent, cfg, start_step, start_step, L.video, L.traj_plot, rollout_agent=bc_agent,q_pol_agent=bc_agent)
    
    L.finish()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
