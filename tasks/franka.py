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
        

    @property
    def state(self):
        return self._state_obs.astype(np.float32)

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
        obs = self.env.reset()
        self._state_obs = self._get_state_obs(obs)
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
            aug_action[5] = np.arctan2(action[4], action[3])/3.14
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
        reward = float(info["success"]) - 1.0
        if self.cfg.real_robot:
            reward = -1.0
        return self._stacked_obs(), reward, False, info

    def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
        if self.cfg.real_robot:
            return self._get_pixel_obs()[0,:3].transpose(1,2,0)
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
    
def recompute_real_rwd(cfg, states):
    assert(cfg.task.startswith('franka-FrankaPickPlaceRandom'))
    rewards = torch.zeros(
        (cfg.episode_length,), dtype=torch.float32, device=states.device
    )-1.0
    for i in reversed(range(cfg.episode_length)):
        if states[i+1, -1] > 1.05:
            rewards[i] = 0.0
        else:
            break
    return rewards

def make_franka_env(cfg):
    env_id = cfg.task.split("-", 1)[-1] + "-v0"
    env = gym.make(env_id, **{'reward_mode': 'sparse', 'is_hardware': cfg.real_robot})
    env = FrankaWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)
    env.reset()
    cfg.state_dim = env.state.shape[0]
    return env
