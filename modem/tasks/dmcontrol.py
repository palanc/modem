# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque, defaultdict
from typing import Any, NamedTuple
import numpy as np
import torch
import dm_env
from dm_env import StepType, specs
from dm_control.suite.wrappers import action_scale, pixels
from dm_control import suite
import gym


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape

        if len(pixels_shape) < 4:
            pixels_shape = np.expand_dims(pixels_shape,axis=0)
        elif len(pixels_shape>4):
            assert False
        assert(pixels_shape[0]==1 and len(pixels_shape)==4)
        
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[0],pixels_shape[3] * num_frames], pixels_shape[1:3]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=1)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]

        if len(pixels.shape) < 4:
            pixels = np.expand_dims(pixels, axis=0)
        elif len(pixels.shape) > 4:
            assert(False)
        assert(pixels.shape[0]==1 and len(pixels.shape)==4)

        return pixels.transpose(0,3,1,2).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class TimeStepToGymWrapper(object):
    def __init__(self, env, domain, task, cfg):
        obs_shp = env.observation_spec().shape
        act_shp = env.action_spec().shape
        assert(env.observation_spec().minimum==0)
        assert(env.observation_spec().maximum==255)
        self.observation_space = gym.spaces.Box(
            low=int(env.observation_spec().minimum),
            high=int(env.observation_spec().maximum),
            shape=obs_shp,
            dtype=np.uint8,
        )
        assert(env.action_spec().minimum==-1)
        assert(env.action_spec().maximum==1)
        self.action_space = gym.spaces.Box(
            low=int(env.action_spec().minimum),
            high=int(env.action_spec().maximum),
            shape=act_shp,
            dtype=env.action_spec().dtype,
        )
        self.env = env
        self.domain = domain
        self.task = task
        self.ep_len = cfg.episode_length
        self.t = 0

    @property
    def unwrapped(self):
        return self.env._env._env._env._env._env._env

    @property
    def reward_range(self):
        return None

    @property
    def metadata(self):
        return None

    @property
    def state(self):
        env = self.env._env._env._env._env._env._env
        state = env.task.get_observation(env.physics)
        state = np.concatenate([v.flatten() for v in state.values()])
        return state.astype(np.float32)

    def reset(self):
        self.t = 0
        return self.env.reset().observation

    def step(self, action):
        self.t += 1
        time_step = self.env.step(action)
        return (
            time_step.observation,
            time_step.reward,
            time_step.last() or self.t == self.ep_len,
            defaultdict(float),
        )

    def render(self, mode="rgb_array", width=384, height=384, camera_id=0):
        camera_id = dict(quadruped=2).get(self.domain, camera_id)
        try:
            return self.env.physics.render(height, width, camera_id)
        except:
            return self.env.render(mode, width, height, camera_id)

    def close(self, *args, **kwargs):
        pass


def get_state_dim(env):
    obs_shp = []
    for v in env.observation_spec().values():
        try:
            shp = v.shape[0]
        except:
            shp = 1
        obs_shp.append(shp)
    return int(np.sum(obs_shp))


def make_dmcontrol_env(cfg):
    domain, task = cfg.task.replace("-", "_").split("_", 1)
    domain = dict(cup="ball_in_cup").get(domain, domain)
    env = suite.load(
        domain, task, task_kwargs={"random": cfg.seed}, visualize_reward=False
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, cfg.action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    cfg.state_dim = get_state_dim(env)

    camera_id = dict(quadruped=2).get(domain, 0)
    render_kwargs = dict(height=cfg.img_size, width=cfg.img_size, camera_id=camera_id)
    env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
    env = FrameStackWrapper(env, cfg.get("frame_stack", 1))
    env = ExtendedTimeStepWrapper(env)
    env = TimeStepToGymWrapper(env, domain, task, cfg)

    return env
