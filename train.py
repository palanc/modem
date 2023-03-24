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
from algorithm.helper import Episode, get_demos,get_demos_h5, ReplayBuffer, linear_schedule, do_policy_rollout
from termcolor import colored
from copy import deepcopy
import logger
import hydra

from mj_envs.utils.collect_modem_rollouts_real import check_grasp_success
from mj_envs.utils.policies.heuristic_policy import HeuristicPolicyReal
from tasks.franka import recompute_real_rwd

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
            grasp_success, new_rewards = env.post_process_task(obs_unstacked, states, eval_mode=True)
            if grasp_success:
                ep_reward = torch.sum(new_rewards).item()

            #_, latest_img, grasp_success, _, _ = check_grasp_success(env=env.base_env(), obs=None, force_img=True)
            #if grasp_success:
            #    states = torch.stack(states, dim=0)
            #    ep_reward = torch.sum(recompute_real_rwd(cfg, states)).item()
            #elif reset_pi is not None:
            #    assert(latest_img is not None)
            #    reset_pi.update_grasps(img=latest_img)
            #    print('Executing reset policy (eval)')
            #    reset_pi.do_rollout(horizon=cfg.episode_length)
            #    check_grasp_success(env=env.base_env(), obs=None, just_drop=True)

            ep_success = 1 if grasp_success else 0

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
        #consec_train_fails = 0
        #RESET_PI_THRESH = 3
        #reset_pi = HeuristicPolicyReal(env=env.base_env(), 
        #                               seed=cfg.seed)

    # Load demonstrations
    if cfg.get("demos", 0) > 0:

        valid_demo_buffer = None
        if cfg.bc_only:
            valid_demo_buffer = buffer

        demos = get_demos(cfg) if not cfg.h5_demos else get_demos_h5(cfg, env)

        for i,episode in enumerate(demos):
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
            bc_agent.load(deepcopy(agent.state_dict()))
        print(colored("\nPhase 2: seeding", "green", attrs=["bold"]))

    if cfg.bc_only:
        L.save_model(agent, 0)
        print(colored(f'Model has been checkpointed', 'yellow', attrs=['bold']))
                        
        eval_rew, eval_succ = evaluate(env, agent, cfg, 0, 0, L.video, L.traj_plot, policy_rollout=True)
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
        horizon = int(min(cfg.horizon, linear_schedule(cfg.horizon_schedule, step)))
        while not episode.done:
            if cfg.plan_policy:
                if cfg.bc_rollout:
                    mean = bc_agent.rollout_actions(obs, env.state, horizon)
                else:
                    mean = agent.rollout_actions(obs, env.state, horizon)
                if cfg.bc_q_pol:
                    q_pol = bc_agent.model.pi
                else:
                    q_pol = agent.model.pi
                std = cfg.min_std * torch.ones(horizon, cfg.action_dim, device=bc_agent.device)   
                action = planner(obs, env.state, mean=mean, std=std, step=plan_step, t0=episode.first,q_pol=q_pol, gt_rollout_start_state= None if not cfg.gt_rollout else env.unwrapped.get_env_state())         
            else:
                action = planner(obs, env.state, step=plan_step, t0=episode.first, gt_rollout_start_state= None if not cfg.gt_rollout else env.unwrapped.get_env_state())
            obs, reward, done, info = env.step(action.cpu().numpy())
            episode += (obs, env.state, action, reward, done)

        assert (
            len(episode) == cfg.episode_length
        ), f"Episode length {len(episode)} != {cfg.episode_length}"
        
        if cfg.real_robot:
            print('Checking success')

            grasp_success, new_rewards = env.post_process_task(episode.obs[:,0,-4:-1].cpu().numpy(), episode.state, eval_mode=False)
            info['success'] = int(grasp_success)
            if grasp_success:
                episode.reward = new_rewards
                episode.cumulative_reward = torch.sum(episode.reward).item()


            #_, latest_img, grasp_success, _, _ = check_grasp_success(env.base_env(), obs=None, force_img=(consec_train_fails>=RESET_PI_THRESH-1))
            #info['success'] = int(grasp_success)
            #if grasp_success:
            #    print('Recomputing reward')
            #    episode.reward = recompute_real_rwd(cfg, episode.state)
            #    episode.cumulative_reward = torch.sum(episode.reward).item()
            #    consec_train_fails = 0
            #else:
            #    consec_train_fails += 1
            #    if consec_train_fails >= RESET_PI_THRESH:
            #        assert(latest_img is not None)
            #        reset_pi.update_grasps(img=latest_img)
            #        print('Executing reset policy (train)')
            #        reset_pi.do_rollout(horizon=cfg.episode_length)
            #        check_grasp_success(env.base_env(), obs=None, just_drop=True)
            #        consec_train_fails = 0

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

        if cfg.save_model and env_step % cfg.save_freq == 0:# and env_step > 0:
            L.save_model(agent, env_step)
            print(colored(f"Model has been checkpointed", "yellow", attrs=["bold"]))

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            if not cfg.real_robot or step >= cfg.seed_steps or cfg.plan_policy:

                assert((cfg.bc_rollout and cfg.bc_q_pol) or (not cfg.bc_rollout and not cfg.bc_q_pol))
                if cfg.bc_rollout:
                    bc_eval_rew, bc_eval_succ = evaluate(env, agent, cfg, step, env_step, L.video, L.traj_plot, rollout_agent=bc_agent,q_pol_agent=bc_agent)
                    if not cfg.real_robot:
                        eval_rew, eval_succ = evaluate(env, agent, cfg, step, env_step, L.video, L.traj_plot)
                    else:
                        eval_rew, eval_succ = -cfg.episode_length, 0.0
                else:
                    eval_rew, eval_succ = evaluate(env, agent, cfg, step, env_step, L.video, L.traj_plot)
                    if not cfg.real_robot:
                        bc_eval_rew, bc_eval_succ = evaluate(env, agent, cfg, step, env_step, L.video, L.traj_plot, rollout_agent=bc_agent,q_pol_agent=bc_agent)
                    else:
                        bc_eval_rew, bc_eval_succ = -cfg.episode_length, 0.0
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


    L.finish()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
