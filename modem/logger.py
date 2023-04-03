# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import datetime
import re
import numpy as np
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

CONSOLE_FORMAT = [
    ("episode", "E", "int"),
    ("env_step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]
AGENT_METRICS = [
    "consistency_loss",
    "reward_loss",
    "value_loss",
    "total_loss",
    "weighted_loss",
    "pi_loss",
    "grad_norm",
]

CAT_TO_COLOR = {
    "train": "blue",
    "eval": "green",
}


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    """Pretty-printing of run information. Call at start of training."""
    prefix, color, attrs = "  ", "green", ["bold"]

    def limstr(s, maxlen=32):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v)
        )

    kvs = [
        ("task", cfg.task_title),
        ("train steps", f"{int(cfg.train_steps*cfg.action_repeat):,}"),
        ("observations", "x".join([str(s) for s in cfg.obs_shape])),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
    div = "-" * w
    print(div)
    for k, v in kvs:
        pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """Return a wandb-safe group name for logging. Optionally returns group name as list."""
    lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)

class TrajectoryPlotter:
    def __init__(self, wandb, cfg):
        self.cfg = cfg
        self._wandb = wandb
        if cfg.task.startswith('franka-'):
            if cfg.img_size > 0:
                self.state_labels = ['qp0','qp1','qp2','qp3','qp4','qp5','qp6','qp7','qp8',
                                        'qv0','qv1','qv2','qv3','qv4','qv5','qv6','qv7','qv8',
                                        'eef_x','eef_y','eef_z',
                                        'eef_qw', 'eef_qx', 'eef_qy', 'eef_qz']
                self.state_enabled = [1,1,1,1,1,1,1,1,0,
                                        0,0,0,0,0,0,0,0,0,
                                        1,1,1,
                                        0,0,0,0]
            else:
                self.state_labels = ['qp0','qp1','qp2','qp3','qp4','qp5','qp6','qp7','qp8',
                                     'qp9','qp10','qp11','qp12','qp13','qp14','qp15',
                                     'qv0','qv1','qv2','qv3','qv4','qv5','qv6','qv7','qv8',
                                     'qv9','qv10','qv11','qv12','qv13','qv14',
                                     'eef_x','eef_y','eef_z',
                                     'eef_qw', 'eef_qx', 'eef_qy', 'eef_qz',
                                     'obj_err_x','obj_err_y','obj_err_z',
                                     'tar_err_x','tar_err_y','tar_err_z',]
                self.state_enabled = [1,1,1,1,1,1,1,1,0,
                                      0,0,0,0,0,0,0,
                                      0,0,0,0,0,0,0,0,0,
                                      0,0,0,0,0,0,
                                      1,1,1,
                                      0,0,0,0,
                                      0,0,0,
                                      0,0,0]              
            if 'BinPick' in cfg.task:
                self.act_labels = ['act_x','act_y','act_z','act_cos', 'act_sin','act_grasp']
                self.act_enabled = [1,1,1,1,1,1]
                self.act_limits = {'low': np.array([0.368, -0.25, 0.9, -1, -1, 0.0]),
                                   'high': np.array([0.72, 0.25, 1.3, 1, 1, 0.835])}
            elif 'PlanarPush' in cfg.task:
                self.act_labels = ['act_x','act_y','act_z','act_cos','act_sin']
                self.act_enabled = [1,1,1,1,1]
                self.act_limits = {'low': np.array([0.3, -0.4, 0.865, -1, -1]),
                                   'high': np.array([0.8, 0.4, 0.965, 1, 1])}                
            elif 'BinPush' in cfg.task:
                self.act_labels = ['act_x','act_y','act_z']
                self.act_enabled = [1,1,1]
                self.act_limits = {'low': np.array([0.315, -0.3, 0.89]),
                                   'high': np.array([0.695, 0.275, 1.175])}
            elif 'HangPush' in cfg.task:
                self.act_labels = ['act_x','act_y','act_z']
                self.act_enabled = [1,1,1]
                self.act_limits = {'low': np.array([0.3, -0.1, 1.25]),
                                   'high': np.array([0.8, 0.1, 1.5])}                
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        assert(len(self.state_labels)==len(self.state_enabled))  
        assert(len(self.act_labels)==len(self.act_enabled))
        assert(len(self.act_labels)==self.act_limits['low'].shape[0])
        assert(len(self.act_labels)==self.act_limits['high'].shape[0])
        self.cols = 3
        self.rows = (np.sum(np.array(self.state_enabled))+np.sum(np.array(self.act_enabled)))
        self.rows =  (self.rows//self.cols) if (self.rows%self.cols==0) else ((self.rows//self.cols)+1)

    def save_traj(self, policy_states, policy_actions, mppi_states, mppi_actions, step):
        assert(len(policy_states) >= 1)
        assert(len(policy_actions) >= 1)
        assert(len(mppi_states)>=1)
        assert(len(mppi_actions)>=1)
        assert(len(policy_states) == len(mppi_states))
        assert(len(policy_actions) == len(mppi_actions))

        plt.clf()
        fig, axs = plt.subplots(self.rows, self.cols, figsize=(15, 15))
        plt.tight_layout(h_pad=3)
        plot_row = 0
        plot_col = 0

        # Generate colors
        colors = []
        for i in range(len(mppi_states)):
            colors.append((0.8*np.random.rand(),
                           0.8*np.random.rand(),
                           0.8*np.random.rand()))
        t = np.arange(mppi_states[0].shape[0])
        for i in range(len(self.state_labels)):
            if not self.state_enabled[i]:
                continue
            axs[plot_row, plot_col].set_title(self.state_labels[i]+' (idx {})'.format(i))
            axs[plot_row, plot_col].set_xlabel('Step')
            for j in range(len(mppi_states)):
                #axs[plot_row, plot_col].plot(t, mppi_states[j][:,i])
                axs[plot_row, plot_col].plot(t, policy_states[j][:,i], color=colors[j])
                axs[plot_row, plot_col].fill_between(t, 
                                                     policy_states[j][:,i]+ np.abs(policy_states[j][:,i]-mppi_states[j][:,i]),
                                                     policy_states[j][:,i]- np.abs(policy_states[j][:,i]-mppi_states[j][:,i]),
                                                     color=(*colors[j],0.3))
            plot_col += 1
            if plot_col >= self.cols:
                plot_row += 1
                plot_col = 0

        t = np.arange(mppi_actions[0].shape[0])
        # unnormalize actions
        #for i in range(len(mppi_actions)):
        #    mppi_actions[i] = (0.5*mppi_actions[i]+0.5)*(self.act_limits['high']-self.act_limits['low'])+self.act_limits['low']
        
        for i in range(len(self.act_labels)):
            if not self.act_enabled[i]:
                continue
            axs[plot_row, plot_col].set_title(self.act_labels[i]+' (idx {})'.format(i))
            axs[plot_row, plot_col].set_xlabel('Step')
            axs[plot_row, plot_col].set_ylim(bottom=self.act_limits['low'][i],top=self.act_limits['high'][i])
            for j in range(len(mppi_actions)):
                #axs[plot_row, plot_col].plot(t, mppi_actions[j][:,i])
                axs[plot_row, plot_col].plot(t, policy_actions[j][:,i], color=colors[j])
                axs[plot_row, plot_col].fill_between(t, 
                                                     policy_actions[j][:,i]+ np.abs(policy_actions[j][:,i]-mppi_actions[j][:,i]),
                                                     policy_actions[j][:,i]- np.abs(policy_actions[j][:,i]-mppi_actions[j][:,i]),
                                                     color=(*colors[j],0.3))                
            plot_col += 1
            if plot_col >= self.cols:
                plot_row += 1
                plot_col = 0

        self._wandb.log({'Trajectories': self._wandb.Image(plt)}, step=step, commit=False)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, root_dir, wandb, render_size=384, fps=15):
        self.save_dir = (root_dir / "eval_video") if root_dir else None
        self._wandb = wandb
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir and self._wandb and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                height=self.render_size,
                width=self.render_size,
                camera_id=0,
            )
            if frame is not None:
                self.frames.append(frame)

    def save(self, step, key="videos/eval_video"):
        if self.enabled and len(self.frames) > 0:
            frames = np.stack(self.frames).transpose(0, 3, 1, 2)
            self._wandb.log(
                {key: self._wandb.Video(frames, fps=self.fps, format="mp4")}, step=step
            )


class Logger(object):
    """Primary logger object. Logs either locally or using wandb."""

    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_model = cfg.save_model
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._eval = []
        print_run(cfg)
        project, entity = cfg.get("wandb_project", "none"), cfg.get(
            "wandb_entity", "none"
        )
        import wandb

        wandb.init(
            project=project,
            entity=entity,
            name=str(cfg.seed),
            group=self._group,
            tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
            dir=self._log_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        self._wandb = wandb
        self._video = (
            VideoRecorder(log_dir, self._wandb)
            if self._wandb and cfg.save_video
            else None
        )
        self._traj_plot = TrajectoryPlotter(self._wandb, cfg) if self._wandb else None

    @property
    def video(self):
        return self._video

    @property
    def traj_plot(self):
        return self._traj_plot

    @property
    def model_dir(self):
        return self._model_dir

    def save_model(self, agent, identifier):
        if self._save_model:
            fp = self._model_dir / f"{str(identifier)}.pt"
            agent.save(fp)
            if self._wandb:
                artifact = self._wandb.Artifact(
                    self._group + "-" + str(self._seed) + "-" + str(identifier),
                    type="model",
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def finish(self):
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "grey")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "grey")} {value:.01f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "grey")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            pieces.append(f"{self._format(disp_k, d.get(k, 0), ty):<26}")
        print("   ".join(pieces))

    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if self._wandb is not None:
            if category in {"train", "eval"}:
                xkey = "env_step"
            for k, v in d.items():
                self._wandb.log({category + "/" + k: v}, step=d[xkey])
        if category == "eval":
            keys = ["env_step", "episode_reward"]
            self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
            pd.DataFrame(np.array(self._eval)).to_csv(
                self._log_dir / "eval.log", header=keys, index=None
            )
        self._print(d, category)
