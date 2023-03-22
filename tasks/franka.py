# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import torch
import numpy as np
import gym
from gym.wrappers import TimeLimit
import mj_envs.envs.arms
import matplotlib.pyplot as plt
from enum import Enum 
from mj_envs.utils.policies.heuristic_policy import HeuristicPolicyReal
from mj_envs.utils.collect_modem_rollouts_real import check_grasp_success
from mj_envs.utils.color_threshold import ColorThreshold
import time

class FrankaTask(Enum):
    PickPlace=1
    BinPush=2
    HangPush=3
    PlanarPush=4

class FrankaWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)
        img_size = max(cfg.img_size, 10)
        self.camera_names = cfg.get("camera_views", ["view_1"])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(len(self.camera_names), self._num_frames * (3 if cfg.img_size <= 0 else 4), img_size, img_size),
            dtype=np.uint8,
        )
        
        if 'PickPlace' in cfg.task:
            self.franka_task = FrankaTask.PickPlace
        elif 'BinPush' in cfg.task:
            self.franka_task = FrankaTask.BinPush
        elif 'HangPush' in cfg.task:
            self.franka_task = FrankaTask.HangPush
        elif 'PlanarPush' in cfg.task:
            self.franka_task = FrankaTask.PlanarPush
        else:
            raise NotImplementedError()

        if self.franka_task == FrankaTask.PickPlace:
            act_low = -np.ones(7)
            act_high = np.ones(7)
        elif self.franka_task == FrankaTask.PlanarPush:
            act_low = -np.ones(5)
            act_high = np.ones(5)
        else:
            act_low = -np.ones(3)
            act_high = np.ones(3)
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
        
        if cfg.task.startswith('franka-FrankaPickPlace'):

            self.consec_train_fails = 0
            self.RESET_PI_THRESH = 3
            self.reset_pi = HeuristicPolicyReal(env=self.env, seed=self.cfg.seed)
        elif cfg.task.startswith('franka-FrankaPlanarPush') or cfg.task.startswith('franka-FrankaBinPush'):
            self.col_thresh = ColorThreshold(cam_name=self.camera_names[0], 
                                             left_crop=self.env.success_mask['left'], 
                                             right_crop=self.env.success_mask['right'], 
                                             top_crop=self.env.success_mask['top'], 
                                             bottom_crop=self.env.success_mask['bottom'], 
                                             thresh_val=self.env.success_mask['thresh'], 
                                             target_uv=np.array([0,0.55]),
                                             render=False)

    @property
    def state(self):
        return self._state_obs.astype(np.float32)

    @property
    def observation(self):
        return self._stacked_obs()

    def _get_state_obs(self, obs):

        qp = self.env.sim.data.qpos.ravel()
        qv = self.env.sim.data.qvel.ravel()
        grasp_pos = self.env.sim.data.site_xpos[self.env.grasp_sid].ravel()
        obj_err = self.env.sim.data.site_xpos[self.env.object_sid]-self.env.sim.data.site_xpos[self.env.grasp_sid]
        tar_err = self.env.sim.data.site_xpos[self.env.target_sid]-self.env.sim.data.site_xpos[self.env.object_sid]

        if self.cfg.img_size <= 0:
            manual = np.concatenate([qp,qv,grasp_pos,obj_err,tar_err])
            assert(np.isclose(obs[:manual.shape[0]], manual).all())
        else:
            manual = np.concatenate([qp[:9], qv[:9], grasp_pos])
            assert(np.isclose(obs[:9], qp[:9]).all())
            assert(np.isclose(obs[qp.shape[0]:qp.shape[0]+9], qv[:9]).all())
            assert(np.isclose(obs[qp.shape[0]+qv.shape[0]:qp.shape[0]+qv.shape[0]+3], grasp_pos).all())

        return manual


    def _get_pixel_obs(self):
        if self.cfg.img_size <= 0:
            return np.zeros((self.observation_space.shape[0],
                             self.observation_space.shape[1]//self._num_frames,
                             self.observation_space.shape[2],
                             self.observation_space.shape[3]), dtype=np.uint8)

        vis_obs_dict = self.env.get_visual_obs_dict(self.env.sim_obsd)
        img_views = []
        for camera_name in self.camera_names:
            rgb_key = 'rgb:'+camera_name+':'+str(self.cfg.img_size)+'x'+str(self.cfg.img_size)+':2d'
            #assert rgb_key in vis_obs_dict, 'Missing viz key {}, available {}'.format(rgb_key, vis_obs_dict.keys())
            rgb_img = vis_obs_dict[rgb_key].squeeze().transpose(2,0,1) # cxhxw
            
            #if self.cfg.task.startswith('franka-FrankaPlanarPush') or self.cfg.task.startswith('franka-FrankaBinPush'):
            #    rgb_img = rgb_img[::-1]
            
            depth_key = 'd:'+camera_name+':'+str(self.cfg.img_size)+'x'+str(self.cfg.img_size)+':2d'
            #assert depth_key in vis_obs_dict, 'Missing viz key {}, available {}'.format(depth_key, vis_obs_dict.keys())
            
            if self.cfg.real_robot:
                depth_img = np.array(np.clip(vis_obs_dict[depth_key],0,255), dtype=np.uint8) # 1xhxw
            else:
                depth_img = 255 *vis_obs_dict[depth_key] # 1xhxw
            
            img_views.append(np.concatenate((rgb_img, depth_img), axis=0))
        return np.stack(img_views, axis=0)     

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=1)

    def reset(self):

        if ((self.cfg.task.startswith('franka-FrankaPlanarPush') or 
             self.cfg.task.startswith('franka-FrankaBinPush')) 
            and len(self._frames)> 0):
            # Move to pre-reset
            print('moving to preset')
            if self.cfg.task.startswith('franka-FrankaPlanarPush'):
                preset_eef = np.array([-0.4, 0.5,1.25,1.0,np.arcsin(-0.74)])
            elif self.cfg.task.startswith('franka-FrankaBinPush'):
                preset_eef = np.array([-0.3, 0.5,1.25])
            else:
                raise NotImplementedError()
            preset_eef[:3] = 2*(((preset_eef[:3]-self.env.unwrapped.pos_limit_low[:3])/(self.env.unwrapped.pos_limit_high[:3]-self.env.unwrapped.pos_limit_low[:3]))-0.5)
            preset_step = 0
            while(preset_step < 50):
                self.step(preset_eef)
                preset_step += 1
            print('at preset')

            if self.cfg.task.startswith('franka-FrankaBinPush'):
                time.sleep(5) # Wait for ball to stop moving

        obs = self.env.reset()
        self._state_obs = self._get_state_obs(obs)
        obs = self._get_pixel_obs()
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._stacked_obs()

    def set_env_state(self, env_state):
        self.env.unwrapped.set_env_state(env_state)
        env_obs = self.env.get_obs()

        self._state_obs = self._get_state_obs(env_obs)
        obs = self._get_pixel_obs()    
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._stacked_obs()            

    def step(self, action):
        reward = 0

        if self.franka_task == FrankaTask.PickPlace:
            aug_action = np.zeros(action.shape[0]+1, dtype=action.dtype)
            aug_action[:3] = action[:3]
            aug_action[3] = 1.0
            aug_action[4] = -1.0
            aug_action[5] = np.arctan2(action[4], action[3])/np.pi
            aug_action[6] = 2*(int(action[5]>0.0)-0.5)
            aug_action[7] = action[6]
        else:
            aug_action = np.zeros(7, dtype=action.dtype)
            aug_action[:3] = action[:3]
            # Note that unset dimensions of aug_action will be clipped to a constant value
            # as specified in env/arms/franka/__init__.py
            if self.franka_task == FrankaTask.PlanarPush:
                aug_action[5] = np.arctan2(action[4], action[3])
                aug_action[5] = 2*(((aug_action[5] - self.env.pos_limit_low[5]) / (self.env.pos_limit_high[5] - self.env.pos_limit_low[5])) - 0.5)

        
        for _ in range(self.cfg.action_repeat):
            obs, r, _, info = self.env.step(aug_action)
            reward += r
        self._state_obs = self._get_state_obs(obs)
        obs = self._get_pixel_obs()
        self._frames.append(obs)
        info["success"] = info["solved"]
        if not self.cfg.dense_reward:
            reward = float(info["success"]) - 1.0
        if self.cfg.real_robot:
            reward = -1.0
        return self._stacked_obs(), reward, False, info

    def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
        if self.cfg.real_robot:
            return self._frames[-1][0,:3].transpose(1,2,0)
        else:
            return np.flip(
                self.env.env.sim.render(
                    mode="offscreen",
                    width=width,
                    height=height,
                    camera_name=self.camera_names[0],
                ),
                axis=0,
            )

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

    def base_env(self):
        return self.env


    def post_process_task(self, obs, states, eval_mode=True):
        assert(self.cfg.real_robot)
        success = False
        rewards = None
        if self.cfg.task.startswith('franka-FrankaPickPlaceRandom'):
            _, latest_img, success, _, _ = check_grasp_success(env=self.env, obs=None, force_img=eval_mode or (self.consec_train_fails>=self.RESET_PI_THRESH-1))
            if success:
                rewards = recompute_real_rwd(self.cfg, states)
                self.consec_train_fails = 0
            else:
                if not eval_mode:
                    self.consec_train_fails += 1
                
                if eval_mode or self.consec_train_fails >= self.RESET_PI_THRESH:
                    assert(latest_img is not None)
                    print('Executing reset policy ({})'.format('eval' if eval_mode else 'train'))
                    self.reset_pi.update_grasps(img=latest_img)
                    self.reset_pi.do_rollout(horizon=self.cfg.episode_length)
                    check_grasp_success(self.env, obs=None, just_drop=True)

                    if not eval_mode:
                        self.consec_train_fails = 0
        elif self.cfg.task.startswith('franka-FrankaPlanarPush') or self.cfg.task.startswith('franka-FrankaBinPush'):
            rewards = recompute_real_rwd(self.cfg, states, obs, self.col_thresh)
            success = torch.sum(rewards).item() >= -1*self.cfg.episode_length+4.999
        else:
            raise NotImplementedError()
        
        return success, rewards

def recompute_real_rwd(cfg, states, obs=None, col_thresh=None):
    rewards = torch.zeros(
        (cfg.episode_length,), dtype=torch.float32, device=states.device
    )-1.0
    if cfg.task.startswith('franka-FrankaPickPlaceRandom'):
        for i in reversed(range(cfg.episode_length)):
            if states[i+1, -1] > 1.05:
                rewards[i] = 0.0
            else:
                break
    elif cfg.task.startswith('franka-FrankaPlanarPush') or cfg.task.startswith('franka-FrankaBinPush'):
        assert(len(obs.shape)==4)
        assert(obs.shape[0]==cfg.episode_length+1)
        assert(obs.shape[1]==3)
        assert(obs.shape[2]==cfg.img_size and obs.shape[3]==cfg.img_size)
        imgs = obs.transpose(0,2,3,1)
        for t in range(1,obs.shape[0]):
            img_success, luv_angle = col_thresh.detect_success(imgs[t])
            if img_success:
                rewards[t-1] = 0
        print('Ep reward: {}'.format(torch.sum(rewards).item()))
    else:
        raise NotImplementedError()

    return rewards

def make_franka_env(cfg):
    env_id = cfg.task.split("-", 1)[-1] + "-v0"
    reward_mode = 'dense' if cfg.dense_reward else 'sparse'
    env = gym.make(env_id, **{'reward_mode': reward_mode, 'is_hardware': cfg.real_robot})
    env = FrankaWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)

    if cfg.real_robot and (cfg.task.startswith('franka-FrankaPlanarPush') or cfg.task.startswith('franka-FrankaBinPush')):
        env.reset()

    env.reset()
    cfg.state_dim = env.state.shape[0]
    return env
