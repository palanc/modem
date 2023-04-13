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

def evaluate(env, agent, cfg, step, env_step, video, traj_plot, policy_rollout=False,rollout_agent=None,q_pol_agent=None):
    """Evaluate a trained agent and optionally save a video."""
    if rollout_agent is None:
        rollout_agent = agent
    if q_pol_agent is None:
        q_pol_agent = agent

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_successes = []
    episode_start_states = []

    for i in range(cfg.eval_episodes):
        print('Episode {}'.format(i))
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        episode_start_states.append(env.unwrapped.get_env_state())
        states = [torch.tensor(env.state, dtype=torch.float32, device=agent.device)]
        obs_unstacked = [obs[0,-4:-1]]
        actions = []
        if video:
            if cfg.real_robot:
                video.init(env, enabled=(i < 5))
            else:
                video.init(env, enabled=(i == 0))
        plan_step = step
        planner = agent.plan
        while not done:
            if policy_rollout:
                plan_step = 0
            
            if cfg.plan_policy:
                horizon = int(min(cfg.horizon, linear_schedule(cfg.horizon_schedule, step)))
                mean = rollout_agent.rollout_actions(obs, env.state, horizon)
                std = cfg.min_std* torch.ones(horizon, cfg.action_dim, device=agent.device)   
                action = planner(obs, env.state, mean=mean, std=std, eval_mode=True, step=plan_step, t0=t == 0, q_pol=q_pol_agent.model.pi, gt_rollout_start_state= None if not cfg.gt_rollout else env.unwrapped.get_env_state())   
            else:
                action = planner(obs, env.state, eval_mode=True, step=plan_step, t0=t == 0, gt_rollout_start_state= None if not cfg.gt_rollout else env.unwrapped.get_env_state())
            
            obs, reward, done, info = env.step(action.cpu().numpy())
            states.append(torch.tensor(env.state, dtype=torch.float32, device=agent.device))
            obs_unstacked.append(obs[0,-4:-1])
            actions.append(env.base_env().last_eef_cmd)
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        ep_success = info.get('success', 0)

        if cfg.real_robot:
            states = torch.stack(states, dim=0)
            obs_unstacked = np.stack(obs_unstacked, axis=0)
            task_success, new_rewards = env.post_process_task(obs_unstacked, states, eval_mode=True)
            if task_success:
                ep_reward = torch.sum(new_rewards).item()

            ep_success = 1 if task_success else 0

        episode_states.append(np.stack([state.cpu().numpy() for state in states],axis=0))
        episode_actions.append(np.stack(actions, axis=0))
        episode_rewards.append(ep_reward)
        episode_successes.append(ep_success)
        if video:
            video.save(env_step)

    if traj_plot:
        if not cfg.real_robot:
            rollout_states, rollout_actions = do_policy_rollout(cfg, episode_start_states, rollout_agent)
        else:
            rollout_states, rollout_actions = episode_states, episode_actions
        traj_plot.save_traj(rollout_states, rollout_actions, 
                            episode_states, episode_actions, 
                            env_step)
    return np.nanmean(episode_rewards), np.nanmean(episode_successes)

def load_checkpt_fp(cfg, checkpt_dir, bc_dir):
    
    model_fps = []
    step = 0

    if checkpt_dir is not None:
        while True:
            if os.path.exists(checkpt_dir / f'{str(step)}.pt'):
                model_fps.append(checkpt_dir / f'{str(step)}.pt')
            else:
                step -= cfg.save_freq
                step = max(0, step)
                break
            step += cfg.save_freq  

    return model_fps

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

    env = make_env(cfg)
    demo_buffer = ReplayBuffer(deepcopy(cfg)) if cfg.get("demos", 0) > 0 else None
    buffer = ReplayBuffer(cfg)
    L = logger.Logger(work_dir, cfg)

    model_fps = load_checkpt_fp(cfg, L._model_dir, L._model_dir)      

    bc_agent = TDMPC(cfg)
    bc_model_fp = model_fps[0]
    if bc_model_fp is not None:
        bc_agent.load(bc_model_fp)

    step = 0

    step = cfg.save_freq*(len(model_fps)-1)
    model_fps = [model_fps[-1]]

    start_time = time.time()
    for model_fp in model_fps:
        assert(model_fp is not None)
        agent = TDMPC(cfg)
        print('Loading agent '+str(model_fp))
        agent.load(model_fp)  
       
        # Log training episode
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            "step": step,
            "env_step": env_step,
            "total_time": time.time() - start_time,
            "episode_reward": 0.0,
            "episode_success": 0,
        }

        if not cfg.real_robot or step >= cfg.seed_steps or cfg.plan_policy:

            assert((cfg.bc_rollout and cfg.bc_q_pol) or (not cfg.bc_rollout and not cfg.bc_q_pol))
            bc_eval_rew, bc_eval_succ = evaluate(env, agent, cfg, step, env_step, L.video, L.traj_plot, rollout_agent=bc_agent,q_pol_agent=bc_agent)
            eval_rew, eval_succ = 0.0, 0#evaluate(env, agent, cfg, step, env_step, L.video, L.traj_plot)
            print('BC Reward {}, BC Succ {}'.format(bc_eval_rew, bc_eval_succ))
            print('Ep Reward {}, Ep Succ {}'.format(eval_rew, eval_succ))
        else:
            eval_rew, eval_succ = -cfg.episode_length, 0.0
            bc_eval_rew, bc_eval_succ = -cfg.episode_length, 0.0
        common_metrics.update(
            {"episode_reward": eval_rew, "episode_success": eval_succ,
                "bc_episode_reward": bc_eval_rew, "bc_episode_success":bc_eval_succ}
        )
        L.log(common_metrics, category="eval")
        step += cfg.save_freq

    L.finish()
    print("Eval completed successfully")


if __name__ == "__main__":
    train()
