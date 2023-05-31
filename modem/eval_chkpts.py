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

def evaluate(env, agent, cfg, step, env_step, video, traj_plot=None, q_plot=None, policy_rollout=False):
    """Evaluate a trained agent and optionally save a video."""

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_successes = []
    episode_start_states = []

    ep_q_stats = []
    ep_q_success = []
    #for i in range(cfg.eval_episodes):
    ep_count = 0
    while ep_count < cfg.eval_episodes:
        print('Episode {}'.format(ep_count))
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        start_time = time.time()
        episode_start_states.append(env.unwrapped.get_env_state())
        states = [torch.tensor(env.state, dtype=torch.float32, device=agent.device)]
        obs_unstacked = [obs[0,-4:-1]]
        actions = []
        q_stats = []
        if video:
            if cfg.real_robot:
                video.init(env, enabled=(ep_count < 5))
            else:
                video.init(env, enabled=(ep_count == 0))
        while not done:              
            action, q_stat = agent.plan(obs, env.state, 
                                        eval_mode=True, 
                                        step=(0 if policy_rollout else step), 
                                        t=t)   
            
            obs, reward, done, info = env.step(action.cpu().numpy())
            states.append(torch.tensor(env.state, dtype=torch.float32, device=agent.device))
            obs_unstacked.append(obs[0,-4:-1])
            actions.append(env.base_env().last_eef_cmd)
            ep_reward += reward
            if q_stat is not None:
                q_stats.append(q_stat)
            if video:
                video.record(env)
            t += 1
        print('Episode length: {}'.format(time.time()-start_time))
        ep_success = np.sum(ep_reward)>=5#info.get('success', 0)

        if cfg.real_robot:
            states = torch.stack(states, dim=0)
            obs_unstacked = np.stack(obs_unstacked, axis=0)
            task_success, new_rewards, retry_episode = env.post_process_task(obs_unstacked, states, eval_mode=True)
            
            if retry_episode:
                print('Episode result unclear, retry')
                continue
            
            if task_success:
                ep_reward = torch.sum(new_rewards).item()

            ep_success = 1 if task_success else 0

        episode_states.append(np.stack([state.cpu().numpy() for state in states],axis=0))
        episode_actions.append(np.stack(actions, axis=0))
        episode_rewards.append(ep_reward)
        episode_successes.append(ep_success)
        if len(q_stats) > 0:
            ep_q_stats.append(q_stats)
            ep_q_success.append(ep_success)
        if video:
            video.save(env_step)
        ep_count += 1

    assert(len(episode_states)) == cfg.eval_episodes
    assert(len(episode_actions)) == cfg.eval_episodes
    assert(len(episode_rewards)) == cfg.eval_episodes
    assert(len(episode_successes)) == cfg.eval_episodes

    ep_uncertainty_weight = 0.0
    uncertainty_count = 0
    if len(ep_q_stats) > 0 and q_plot is not None:
        print('LOGGING Q VALS')
        q_plot.log_q_vals(env_step, ep_q_stats, ep_q_success)
        for i in range(len(ep_q_stats)):
            for j in range(len(ep_q_stats[i])):
                ep_uncertainty_weight += ep_q_stats[i][j]['uncertainty_weight']
                uncertainty_count += 1
        ep_uncertainty_weight = ep_uncertainty_weight/uncertainty_count
    if traj_plot:
        if not cfg.real_robot:
            rollout_states, rollout_actions = do_policy_rollout(cfg, episode_start_states, agent)
        else:
            rollout_states, rollout_actions = episode_states, episode_actions
        traj_plot.save_traj(rollout_states, rollout_actions, 
                            episode_states, episode_actions, 
                            env_step)
    return np.nanmean(episode_rewards), np.nanmean(episode_successes), ep_uncertainty_weight

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
    print('Model dir {}'.format(L._model_dir))
    model_fps = load_checkpt_fp(cfg, L._model_dir, L._model_dir)      

    step = 0

    #step = cfg.save_freq*(len(model_fps)-1)
    #model_fps = [model_fps[-1]]

    start_time = time.time()
    i = 0
    while i < len(model_fps):
    #for i in range(len(model_fps)):
        model_fp = model_fps[i]
        assert(model_fp is not None)
        agent = TDMPC(cfg)
        print('Loading agent '+str(model_fp))
        agent_data = torch.load(model_fp)
        if isinstance(agent_data, dict):
            agent.load(agent_data)
        else:
            agent = agent_data    
            agent.cfg = cfg  
        agent.model.eval()
        agent.model_target.eval()  
       
        # Log training episode
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            "step": step,
            "env_step": env_step,
            "total_time": time.time() - start_time,
            "episode_reward": 0.0,
            "episode_success": 0,
        }

        eval_rew, eval_succ, uncertainty_weight = evaluate(env, agent, cfg, step, env_step, L.video)
        print('Eval rew {}, eval succ {}'.format(eval_rew, eval_succ))
        while True:
            resp = input('c to continue, r to repeat')
            if resp == 'c':
                common_metrics.update(
                    {"episode_reward": eval_rew, "episode_success": eval_succ}
                )
                L.log(common_metrics, category="eval")
                step += cfg.save_freq
                i+= 1
                break
            elif resp == 'r':
                break
            else:
                pass
        

    L.finish()
    print("Eval completed successfully")


if __name__ == "__main__":
    train()
