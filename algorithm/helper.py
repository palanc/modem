# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from PIL import Image
from pathlib import Path
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from typing import Union
from env import make_env
import multiprocessing as mp
import concurrent.futures

__REDUCE__ = lambda b: "mean" if b else "none"


def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (
        (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x)
        .squeeze(0)
        .shape
    )


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def soft_update_params(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class NormalizeImg(nn.Module):
    """Module that divides (pixel) observations by 255."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0)


class Flatten(nn.Module):
    """Module that flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def enc(cfg):
    """Returns our MoDem encoder that takes a stack of 224x224 frames as input."""
    if cfg.img_size <= 0:
        return None
        
    C = int(cfg.obs_shape[0])
    layers = [
        NormalizeImg(),
        nn.Conv2d(C, cfg.num_channels, 7, stride=2),
        nn.ReLU(),
        nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2),
        nn.ReLU(),
        nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
        nn.ReLU(),
    ]
    out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
    layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
    return nn.Sequential(*layers)


def state_enc(cfg):
    """Returns a proprioceptive state encoder + modality fuse."""
    return (
        nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.enc_dim),
            nn.ELU(),
            nn.Linear(cfg.enc_dim, cfg.latent_dim),
        ),
        nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.enc_dim),
            nn.ELU(),
            nn.Linear(cfg.enc_dim, cfg.latent_dim),
        ),
    )


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]),
        act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]),
        act_fn,
        nn.Linear(mlp_dim[1], out_dim),
    )


def q(cfg, act_fn=nn.ELU()):
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(
        nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim),
        nn.LayerNorm(cfg.mlp_dim),
        nn.Tanh(),
        nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
        act_fn,
        nn.Linear(cfg.mlp_dim, 1),
    )


class RandomShiftsAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, cfg):
        super().__init__()
        self.pad = int(
            cfg.img_size / 21
        )  # maintain same padding ratio as in original implementation

    def forward(self, x):
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, cfg, init_obs, init_state=None):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.obs = torch.empty(
            (cfg.episode_length + 1, *init_obs.shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self.obs[0] = torch.tensor(init_obs, dtype=torch.uint8, device=self.device)
        self.state = torch.empty(
            (cfg.episode_length + 1, *init_state.shape),
            dtype=torch.float32,
            device=self.device,
        )
        self.state[0] = torch.tensor(
            init_state, dtype=torch.float32, device=self.device
        )
        self.action = torch.empty(
            (cfg.episode_length, cfg.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward = torch.empty(
            (cfg.episode_length,), dtype=torch.float32, device=self.device
        )
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0

    @classmethod
    def from_trajectory(cls, cfg, obs, states, action, reward, done=None):
        """Constructs an episode from a trajectory."""
        episode = cls(cfg, obs[0], states[0])
        episode.obs[1:] = torch.tensor(
            obs[1:], dtype=episode.obs.dtype, device=episode.device
        )
        episode.state[1:] = torch.tensor(
            states[1:], dtype=episode.state.dtype, device=episode.device
        )
        episode.action = torch.tensor(
            action, dtype=episode.action.dtype, device=episode.device
        )
        episode.reward = torch.tensor(
            reward, dtype=episode.reward.dtype, device=episode.device
        )
        episode.cumulative_reward = torch.sum(episode.reward)
        episode.done = True
        episode._idx = cfg.episode_length
        return episode

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, state, action, reward, done):
        self.obs[self._idx + 1] = torch.tensor(
            obs, dtype=self.obs.dtype, device=self.obs.device
        )
        self.state[self._idx + 1] = torch.tensor(
            state, dtype=self.state.dtype, device=self.state.device
        )
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1


def get_demos(cfg):
    fps = glob.glob(str(Path(cfg.demo_dir) / "demonstrations" / f"{cfg.task}/*.pt"))
    episodes = []
    for fp in fps:

        if len(episodes) >= cfg.demos:
            break 

        data = torch.load(fp)
        frames_dir = Path(os.path.dirname(fp)) / "frames"
        assert frames_dir.exists(), "No frames directory found for {}".format(fp)
        
        frames = []
        for fn in data["frames"]:
            if isinstance(fn, list):
                assert(len(fn) == 2)
                assert(cfg.camera_view in str(fn[0]) and 'rgb' in str(fn[0]))
                assert(cfg.camera_view in str(fn[1]) and 'depth' in str(fn[1]))
                rgb_image = np.array(Image.open(frames_dir/fn[0])).transpose(2,0,1)
                depth_image = np.expand_dims(np.array(Image.open(frames_dir/fn[1])),axis=0)
                frames.append(np.concatenate([rgb_image,depth_image],axis=0))             
            else:
                img = np.array(Image.open(frames_dir / fn))
                frames.append(img.transpose(2,0,1))
        
        obs = np.stack(frames)

        if cfg.task.startswith("mw-"):
            state = torch.tensor(data["states"], dtype=torch.float32)
            state = torch.cat((state[:, :4], state[:, 18 : 18 + 4]), dim=-1)
        elif cfg.task.startswith("adroit-"):
            state = torch.tensor(data["states"], dtype=torch.float32)
            if cfg.task == "adroit-door":
                state = np.concatenate([state[:, :27], state[:, 29:32]], axis=1)
            elif cfg.task == "adroit-hammer":
                state = state[:, :36]
            elif cfg.task == "adroit-pen":
                state = np.concatenate([state[:, :24], state[:, -9:-6]], axis=1)
            else:
                raise NotImplementedError()
        elif cfg.task.startswith('franka-'):
            qp = data['states']['qp']
            qv = data['states']['qv']
            grasp_pos = data['states']['grasp_pos']
            obj_err = data['states']['object_err']
            tar_err = data['states']['target_err']
            if cfg.img_size <= 0:
                state = np.concatenate([qp,qv,grasp_pos,obj_err,tar_err],axis=1)
            else:
                state = np.concatenate([qp[:,:9],qv[:,:9],grasp_pos], axis=1)
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = torch.tensor(data["states"], dtype=torch.float32)
        actions = np.array(data["actions"], dtype=np.float32).clip(-1, 1)

        if cfg.task.startswith('franka-'):
            assert actions.shape[1] == 8
            aug_actions = np.zeros((actions.shape[0],actions.shape[1]-1),dtype=np.float32)
            aug_actions[:,:3] = actions[:,:3]
            aug_actions[:,3] = np.cos(3.14*actions[:,5])
            aug_actions[:,4] = np.sin(3.14*actions[:,5])
            aug_actions[:,5:] = actions[:,6:]
            actions = aug_actions

        if cfg.task.startswith("mw-") or cfg.task.startswith("adroit-"):
            rewards = (
                np.array(
                    [
                        _data[
                            "success" if "success" in _data.keys() else "goal_achieved"
                        ]
                        for _data in data["infos"]
                    ],
                    dtype=np.float32,
                )
                - 1.0
            )
        elif cfg.task.startswith('franka-'):
            rewards = np.array([_data['success' if 'success' in _data.keys() else 'goal_achieved'] for _data in data['infos']], dtype=np.float32) - 1.
        else:  # use dense rewards for DMControl
            rewards = np.array(data["rewards"])
        episode = Episode.from_trajectory(cfg, obs, state, actions, rewards)
        episodes.append(episode)

    assert(len(episodes)==cfg.demos)
    return episodes

def gather_paths_parallel(
    cfg,
    start_state: dict,
    act_list: np.ndarray,
    base_seed: int,
    paths_per_cpu: int,
    num_cpu: int = 1,
):
    """Parallel wrapper around the gather paths function."""

    if num_cpu == 1:
        input_dict = dict(
            cfg=cfg,
            start_state=start_state,
            act_list=act_list,
            base_seed=base_seed
        )
        return generate_paths(**input_dict)

    # do multiprocessing only if necessary
    input_dict_list = []

    for i in range(num_cpu):
        cpu_seed = base_seed + i * paths_per_cpu
        input_dict = dict(
            cfg=cfg,
            start_state=start_state,
            act_list=act_list,
            base_seed=cpu_seed
        )
        input_dict_list.append(input_dict)

    results = _try_multiprocess_mp(
        func=generate_paths,
        input_dict_list=input_dict_list,
        num_cpu=num_cpu,
        max_process_time=300,
        max_timeouts=4,
    )
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths


def _try_multiprocess_mp(
    func: callable,
    input_dict_list: list,
    num_cpu: int = 1,
    max_process_time: int = 500,
    max_timeouts: int = 4,
    *args,
    **kwargs,
):
    """Run multiple copies of provided function in parallel using multiprocessing."""

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [
        pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list
    ]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        #pool.terminate()
        pool.join()
        return _try_multiprocess_mp(
            func, input_dict_list, num_cpu, max_process_time, max_timeouts - 1
        )

    pool.close()
    #pool.terminate()
    pool.join()
    return results

def generate_paths(
    cfg,
    start_state: dict,
    act_list: list,
    base_seed: int
) -> list:
    """Generates perturbed action sequences and then performs rollouts.

    Args:
        env:
            Environment, either as a gym env class, callable, or env_id string.
        start_state:
            Dictionary describing a single start state. All rollouts begin from this state.
        base_act:
            A numpy array of base actions to which we add noise to generate action sequences for rollouts.
        filter_coefs:
            We use these coefficients to generate colored for action perturbation
        base_seed:
            Seed for generating random actions and rollouts.
        num_paths:
            Number of paths to rollout.
        env_kwargs:
            Any kwargs needed to instantiate the environment. Used only if env is a callable.

    Returns:
        A list of paths that describe the resulting rollout. Each path is a list.
    """
    set_seed(base_seed)

    paths = do_env_rollout(cfg, start_state, act_list)
    return paths

def set_seed(seed=None):
    """
    Set all seeds to make results reproducible
    :param seed: an integer to your choosing (default: None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

def do_env_rollout(
    cfg,
    start_state: dict,
    act_list: list
) -> list:
    """Rollout action sequence in env from provided initial states.

    Instantiates requested environment. Sets environment to provided initial states.
    Then rollouts out provided action sequence. Returns result in paths format (list of dicts).

    Args:
        env:
            Environment, either as a gym env class, callable, or env_id string.
        start_state:
            Dictionary describing a single start state. All rollouts begin from this state.
        act_list:
            List of numpy arrays containing actions that will be rolled out in open loop.
        env_kwargs:
            Any kwargs needed to instantiate the environment. Used only if env is a callable.

    Returns:
        A list of paths that describe the resulting rollout. Each path is a list.
    """

    # Not all low-level envs are picklable. For generalizable behavior,
    # we have the option to instantiate the environment within the rollout function.
    e = make_env(cfg)
    e.reset()
    e.unwrapped.real_step = False  # indicates simulation for purpose of trajectory optimization
    paths = []
    H = act_list[0].shape[0]  # horizon
    N = len(act_list)  # number of rollout trajectories (per process)
    for i in range(N):
        if cfg.task.startswith('adroit-') or cfg.task.startswith('franka-'):
            e.unwrapped.set_env_state(start_state)
        else:
            e.unwrapped.physics.set_state(start_state)
        obs = []
        act = []
        rewards = []
        env_infos = []
        states = []

        for k in range(H):
            s, r, d, ifo = e.step(act_list[i][k])

            act.append(act_list[i][k])
            obs.append(s)
            rewards.append(r)
            env_infos.append(ifo)            
            if cfg.task.startswith('adroit-') or cfg.task.startswith('franka-'):
                states.append(e.unwrapped.get_env_state())
            else:
                states.append(e.unwrapped.physics.get_state())
            
        path = dict(
            observations=np.array(obs),
            actions=np.array(act),
            rewards=np.array(rewards),
            env_infos=stack_tensor_dict_list(env_infos),
            states=states,
        )
        paths.append(path)

    return paths

def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret

def stack_tensor_list(tensor_list):
    return np.array(tensor_list)

class ReplayBuffer(object):
    """
    Storage and sampling functionality for training MoDem.
    Uses prioritized experience replay by default.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.capacity = 2 * cfg.train_steps + 1
        obs_shape = (cfg.obs_shape[0]//cfg.frame_stack, *cfg.obs_shape[-2:])
        self._state_dim = cfg.state_dim
        self._obs = torch.empty(
            (self.capacity + 1, *obs_shape), dtype=torch.uint8, device=self.device
        )
        self._last_obs = torch.empty(
            (self.capacity // cfg.episode_length, *cfg.obs_shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self._action = torch.empty(
            (self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device
        )
        self._reward = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._state = torch.empty(
            (self.capacity, self._state_dim), dtype=torch.float32, device=self.device
        )
        self._last_state = torch.empty(
            (self.capacity // cfg.episode_length, self._state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._priorities = torch.ones(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._eps = 1e-6
        self.idx = 0

    def __len__(self):
        return self.idx

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        obs = episode.obs[:-1, -(self.cfg.obs_shape[0]//self.cfg.frame_stack):]
        if episode.obs.shape[1] == (self.cfg.obs_shape[0]//self.cfg.frame_stack):
            last_obs = episode.obs[-self.cfg.frame_stack :].view(
                self.cfg.obs_shape[0], *self.cfg.obs_shape[-2:]
            )
        else:
            last_obs = episode.obs[-1]
        self._obs[self.idx : self.idx + self.cfg.episode_length] = obs
        self._last_obs[self.idx // self.cfg.episode_length] = last_obs
        self._action[self.idx : self.idx + self.cfg.episode_length] = episode.action
        self._reward[self.idx : self.idx + self.cfg.episode_length] = episode.reward
        states = torch.tensor(episode.state, dtype=torch.float32)
        self._state[
            self.idx : self.idx + self.cfg.episode_length, : self._state_dim
        ] = states[:-1]
        self._last_state[
            self.idx // self.cfg.episode_length, : self._state_dim
        ] = states[-1]
        max_priority = (
            1.0
            if self.idx == 0
            else self._priorities[: self.idx].max().to(self.device).item()
        )
        mask = (
            torch.arange(self.cfg.episode_length)
            >= self.cfg.episode_length - self.cfg.horizon
        )
        new_priorities = torch.full(
            (self.cfg.episode_length,), max_priority, device=self.device
        )
        new_priorities[mask] = 0
        self._priorities[self.idx : self.idx + self.cfg.episode_length] = new_priorities
        self.idx = (self.idx + self.cfg.episode_length) % self.capacity

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        obs = torch.empty(
            (self.cfg.batch_size, self.cfg.obs_shape[0], *arr.shape[-2:]),
            dtype=arr.dtype,
            device=torch.device("cuda"),
        )
        obs[:, -(self.cfg.obs_shape[0]//self.cfg.frame_stack):] = arr[idxs].cuda(non_blocking=True)
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg.frame_stack):
            mask[_idxs % self.cfg.episode_length == 0] = False
            _idxs[mask] -= 1
            obs[:, -(i + 1) * (self.cfg.obs_shape[0]//self.cfg.frame_stack) : -i * (self.cfg.obs_shape[0]//self.cfg.frame_stack)] = arr[_idxs].cuda(non_blocking=True)
        return obs.float()

    def sample(self):
        probs = self._priorities[: self.idx] ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(
                total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=True
            )
        ).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()
        obs = (
            self._get_obs(self._obs, idxs)
            if self.cfg.frame_stack > 1
            else self._obs[idxs].cuda(non_blocking=True)
        )
        next_obs_shape = ((self.cfg.obs_shape[0]//self.cfg.frame_stack) * self.cfg.frame_stack, *self._last_obs.shape[-2:])
        next_obs = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *next_obs_shape),
            dtype=obs.dtype,
            device=obs.device,
        )
        action = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *self._action.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )
        reward = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size),
            dtype=torch.float32,
            device=self.device,
        )
        state = self._state[idxs, : self._state_dim]
        next_state = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *state.shape[1:]),
            dtype=state.dtype,
            device=state.device,
        )
        for t in range(self.cfg.horizon + 1):
            _idxs = idxs + t
            next_obs[t] = (
                self._get_obs(self._obs, _idxs + 1)
                if self.cfg.frame_stack > 1
                else self._obs[_idxs + 1].cuda(non_blocking=True)
            )
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
            next_state[t] = self._state[_idxs + 1, : self._state_dim]

        mask = (_idxs + 1) % self.cfg.episode_length == 0
        next_obs[-1, mask] = (
            self._last_obs[_idxs[mask] // self.cfg.episode_length]
            .to(next_obs.device, non_blocking=True)
            .float()
        )
        state = state.cuda(non_blocking=True)
        next_state[-1, mask] = (
            self._last_state[_idxs[mask] // self.cfg.episode_length, : self._state_dim]
            .to(next_state.device)
            .float()
        )
        next_state = next_state.cuda(non_blocking=True)
        next_obs = next_obs.cuda(non_blocking=True)
        action = action.cuda(non_blocking=True)
        reward = reward.unsqueeze(2).cuda(non_blocking=True)
        idxs = idxs.cuda(non_blocking=True)
        weights = weights.cuda(non_blocking=True)

        return obs, next_obs, action, reward, state, next_state, idxs, weights


def linear_schedule(schdl, step):
    """Outputs values following a linear decay schedule"""
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)
