""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
import robohive
from PIL import Image
from pathlib import Path
import click
import numpy as np
import pickle
import os
import torch
import time 

DESC = '''
Collect successful policy rollouts
'''

def load_class_from_str(module_name, class_name):
    try:
        m = __import__(module_name, globals(), locals(), class_name)
        return getattr(m, class_name)
    except (ImportError, AttributeError):
        return None

# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='none')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-n', '--num_rollouts', type=int, help='number of rollouts to save', default=100)
@click.option('-sp', '--sparse_reward', type=bool, default=True)
@click.option('-pp', '--policy_path', type=str, default=None )
def collect_rollouts_cli(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts, sparse_reward, policy_path):
    collect_rollouts(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts, sparse_reward, policy_path)

def collect_rollouts(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts, sparse_reward, policy_path):

    # seed and load environments
    np.random.seed(seed)
    if sparse_reward:
        env = gym.make(env_name, **{'reward_mode': 'sparse'})
    else:
        env = gym.make(env_name)        
    env.seed(seed)
 
    # resolve policy and outputs
    assert(policy_path is not None)
    policy_tokens = policy_path.split('.')
    pi = load_class_from_str('.'.join(policy_tokens[:-1]), policy_tokens[-1])

    if pi is not None:
        pi = pi(env, seed)
    else:
        pi = pickle.load(open(policy_path, 'rb'))
        if output_dir == './': # overide the default
            output_dir, pol_name = os.path.split(policy_path)
            output_name = os.path.splitext(pol_name)[0]
        if output_name is None:
            pol_name = os.path.split(policy_path)[1]
            output_name = os.path.splitext(pol_name)[0]

    # resolve directory
    if (os.path.isdir(output_dir) == False):
        os.mkdir(output_dir)

    rollouts = 0
    successes = 0

    while successes < num_rollouts:

        # examine policy's behavior to recover paths
        paths = env.examine_policy_new(
            policy=pi,
            horizon=env.spec.max_episode_steps,
            num_episodes=1,
            frame_size=(640,480),
            mode=mode,
            output_dir=output_dir+'/',
            filename=output_name+str(successes),
            camera_name=camera_name,
            render=render)

        rollouts += 1

        # evaluate paths
        success_percentage = env.env.evaluate_success(paths)
        if success_percentage > 0.5:
            file_name = output_dir + '/' + output_name + f'{(successes+seed):06d}'+ '.pickle'
            
            paths.close()
            paths.save(trace_name=file_name, verify_length=True, f_res=np.float64)

            successes += 1

            if 'env_infos' in paths[0]:
                reward = np.sum(paths[0]['env_infos']['solved'] * 1.0)
            elif 'env_infos' in paths[0]:
                reward = np.sum([ _data['solved']*1.0 for _data in paths[0]['env_infos']])
            elif 'env_infos/solved' in paths[0]:
                reward = np.sum(paths[0]['env_infos/solved'] * 1.0)
            else:
                raise NotImplementedError()

            print('Success {} ({}/{}), reward {}'.format(successes/rollouts,
                                                         successes,
                                                         rollouts,
                                                         reward))

if __name__ == '__main__':
    collect_rollouts_cli()