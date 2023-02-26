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
from algorithm.helper import Episode, get_demos, ReplayBuffer
from termcolor import colored
from copy import deepcopy
import logger
import hydra

from mj_envs.utils.collect_modem_rollouts_real import check_grasp_success
from tasks.franka import recompute_real_rwd

torch.backends.cudnn.benchmark = True
import pickle

def evaluate(env, agent, cfg, step, env_step, video, policy_rollout=False):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    episode_successes = []
    for i in range(cfg.eval_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        states = [torch.tensor(env.state, dtype=torch.float32, device=agent.device)]
        if video:
            if cfg.real_robot:
                video.init(env, enabled=(i < 5))
            else:
                video.init(env, enabled=(i == 0))
        plan_step = step
        planner = agent.plan
        while not done:
            if policy_rollout:
                obs = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                state = torch.tensor(env.state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                action = torch.squeeze(agent.model.pi(agent.model.h(obs, state), cfg.min_std).detach(),dim=0)
            else:
                #action = agent.plan(obs, env.state, eval_mode=True, step=step, t0=t == 0)
                action = planner(obs, env.state, eval_mode=True, step=plan_step, t0=t == 0)
            obs, reward, done, info = env.step(action.cpu().numpy())
            states.append(torch.tensor(env.state, dtype=torch.float32, device=agent.device))
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        ep_success = info.get('success', 0)
        if cfg.real_robot:
            _, _, grasp_success, _, _ = check_grasp_success(env.base_env(), None)
            if grasp_success:
                states = torch.stack(states, dim=0)
                ep_reward = torch.sum(recompute_real_rwd(cfg, states)).item()
            ep_success = 1 if grasp_success else 0

        episode_rewards.append(ep_reward)
        episode_successes.append(ep_success)
        if video:
            video.save(env_step)
    return np.nanmean(episode_rewards), np.nanmean(episode_successes)

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

    episode_dir = str(cfg.logging_dir) + f"/episodes/{cfg.task}" 
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

    bc_agent = TDMPC(cfg)
    bc_model_fp = model_fp
    if bc_model_fp is not None and start_step //cfg.action_repeat >= cfg.seed_steps:
        bc_model_fp, _, _ = load_checkpt_fp(cfg, None, L._model_dir)
    if bc_model_fp is not None:
        bc_agent.load(bc_model_fp)

    # Load past episodes
    if cfg.real_robot:
        for i in range(start_step//cfg.episode_length):
            episode_fn = 'rollout'+f'{(i):010d}.pt'
            episode_path = episode_dir+'/'+episode_fn     
            with open(episode_path, "rb") as ep_file:
                buffer += pickle.load(ep_file)        
            print('Loaded episode {} of {}'.format(i+1,start_step//cfg.episode_length))
        print('Loaded {} rollouts'.format(len(buffer)//cfg.episode_length))

    # Load demonstrations
    if cfg.get("demos", 0) > 0:

        valid_demo_buffer = None
        if cfg.bc_only:
            valid_demo_buffer = buffer

        for i,episode in enumerate(get_demos(cfg)):
            if valid_demo_buffer is not None and i % 10 == 0:
                valid_demo_buffer += episode
            else:
                demo_buffer += episode
        print(colored(f"Loaded {cfg.demos} demonstrations", "yellow", attrs=["bold"]))
        print(colored("Phase 1: policy pretraining", "red", attrs=["bold"]))
        if model_fp is None or (bc_start_step is not None):
            if bc_start_step is None:
                bc_start_step = 0
            agent.init_bc(demo_buffer if cfg.get("demo_schedule", 0) != 0 else buffer, L, bc_start_step, valid_demo_buffer)
            bc_agent.load(copy.deepcopy(agent.state_dict()))
        print(colored("\nPhase 2: seeding", "green", attrs=["bold"]))

    if cfg.bc_only:
        L.save_model(agent, 0)
        print(colored(f'Model has been checkpointed', 'yellow', attrs=['bold']))
                        
        eval_rew, eval_succ = evaluate(env, agent, cfg, 0, 0, L.video, policy_rollout=True)
        common_metrics = {"env_step": 0, "episode_reward": eval_rew, "episode_success": eval_succ}
        L.log(common_metrics, category="eval")
        print('Eval reward: {}, Eval success: {}'.format(eval_rew, eval_succ))
        exit()

    # Run training
    start_time = time.time()
    start_step = start_step // cfg.action_repeat
    episode_idx = start_step // cfg.episode_length
    for step in range(start_step, start_step+cfg.train_steps + cfg.episode_length, cfg.episode_length):

        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs, env.state)

        planner = agent.plan
        plan_step = step
        while not episode.done:
            action = planner(obs, env.state, step=plan_step, t0=episode.first)
            obs, reward, done, info = env.step(action.cpu().numpy())
            episode += (obs, env.state, action, reward, done)

        assert (
            len(episode) == cfg.episode_length
        ), f"Episode length {len(episode)} != {cfg.episode_length}"
        
        if cfg.real_robot:
            print('Checking success')
            _, _, grasp_success, _, _ = check_grasp_success(env.base_env(), None)
            info['success'] = int(grasp_success)
            if grasp_success:
                print('Recomputing reward')
                episode.reward = recompute_real_rwd(cfg, episode.state)
                episode.cumulative_reward = torch.sum(episode.reward).item()

            episode_fn = 'rollout'+f'{(step//cfg.episode_length):010d}.pt'
            episode_path = episode_dir+'/'+episode_fn
            print('Saving')
            with open(episode_path, 'wb') as epf:
                pickle.dump(episode, epf)   
            print('Done saving')         
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps and len(buffer) >= cfg.seed_steps:
            if step == cfg.seed_steps:
                print(
                    colored(
                        "Seeding complete, pretraining model...",
                        "yellow",
                        attrs=["bold"],
                    )
                )
                num_updates = cfg.seed_steps
            else:
                num_updates = cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i, demo_buffer))
            if step == cfg.seed_steps:
                print(colored("Phase 3: interactive learning", "blue", attrs=["bold"]))

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            "episode": episode_idx,
            "step": step,
            "env_step": env_step,
            "total_time": time.time() - start_time,
            "episode_reward": episode.cumulative_reward,
            "episode_success": info.get("success", 0),
        }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category="train")

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            if not cfg.real_robot or step >= cfg.seed_steps:
                eval_rew, eval_succ = evaluate(env, agent, cfg, step, env_step, L.video)
            else:
                eval_rew, eval_succ = -cfg.episode_length, 0.0
            common_metrics.update(
                {"episode_reward": eval_rew, "episode_success": eval_succ}
            )
            L.log(common_metrics, category="eval")
            if cfg.save_model and env_step % cfg.save_freq == 0:# and env_step > 0:
                L.save_model(agent, env_step)
                print(colored(f"Model has been checkpointed", "yellow", attrs=["bold"]))

    L.finish()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
