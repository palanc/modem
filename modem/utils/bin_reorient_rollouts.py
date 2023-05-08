""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
from robohive.envs.arms.bin_reorient_v0 import BinReorientPolicy
#from move_utils import update_grasps, check_grasp_success, open_gripper
from move_utils import check_reorient_success, update_grasps, graspcenter2pose
from PIL import Image
from pathlib import Path
import click
import numpy as np
import pickle
import time
import os
import glob
import threading
import copy
import cv2
import random
import torch

# python bin_reorient_rollouts.py -e FrankaBinReorientReal_v2d-v0 -m exploration -r none -o /tmp -on robopen08_
DESC='script for collecting bin picking demos on real robot'
# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='none')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-n', '--num_rollouts', type=int, help='number of rollouts to save', default=-1)
@click.option('-fc','--force_check', type=bool, default=False)
def main(env_name, mode, seed, render, camera_name, output_dir, output_name, num_rollouts, force_check):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name, **{'reward_mode': 'sparse', 'is_hardware':True})
    env.seed(seed)

    pi = BinReorientPolicy(env, seed)

    # resolve directory
    if (os.path.isdir(output_dir) == False):
        os.mkdir(output_dir)

    if output_name is None:
        output_name = 'heuristic'

    fps = glob.glob(str(Path(output_dir) / f'{output_name}*'))
    if len(fps) > 0:
        input('Output name {} found in {}, continue?'.format(output_name,output_dir))

    rollouts = 0
    successes = 0

    real_tar_pos = np.array([0.5, 0.0, 1.1])
    latest_img  = None
    grasp_centers = None
    filtered_boxes = None
    img_masked = None
    print('Starting')

    env.reset()
    obs = env.get_obs(update_proprioception=True, update_exteroception=True)
    _, reset_img = check_reorient_success(env, obs)

    while num_rollouts <= 0 or successes < num_rollouts:

        grasp_centers, filtered_boxes, img_masked = update_grasps(img=reset_img,
                                                                out_dir=output_dir+'/debug',
                                                                min_pixels=400,
                                                                luv_thresh=True,
                                                                limit_yaw=False)
        real_obj_pos, real_yaw = graspcenter2pose(grasp_centers[-1], xy_offset=[-0.145,0.065])
        pi.set_real_obj_pos(real_obj_pos)
        pi.set_real_yaw(real_yaw)
        paths = env.examine_policy_new(
            policy=pi,
            horizon=env.spec.max_episode_steps,
            num_episodes=1,
            frame_size=(640,480),
            mode=mode,
            output_dir=output_dir+'/',
            filename=output_name,
            camera_name=camera_name,
            render=render)   
        print('Passed yaw {}'.format(real_yaw))
        reorient_success, reset_img = check_reorient_success(env, obs,out_dir=output_dir+'/debug')

        if reorient_success:
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = output_dir + '/' + output_name + '{}_paths.pickle'.format(time_stamp)
            paths.close()
            paths.save(trace_name=file_name, verify_length=True, f_res=np.float64)            
            successes += 1

    '''
    while num_rollouts <= 0 or successes < num_rollouts:

        if latest_img is not None and (grasp_centers is None or len(grasp_centers) <= 0):
            grasp_centers, filtered_boxes, img_masked = update_grasps(img=latest_img,
                                                                      out_dir=output_dir+'/debug')

        if grasp_centers is not None and len(grasp_centers) > 0:
            real_obj_pos = np.array([grasp_centers[-1][0],grasp_centers[-1][1], 0.91])
            real_yaw = grasp_centers[-1][2]
            if output_dir is not None:

                rec_img = img_masked.copy()
                x,y,w,h = filtered_boxes[-1]
                cv2.rectangle(rec_img, (x,y), (x+w, y+h), (255,0,0), 1)

                rec_img_fn = os.path.join(output_dir,'debug/masked_latest.png')
                cv2.imwrite(rec_img_fn, cv2.cvtColor(rec_img, cv2.COLOR_RGB2BGR))

            grasp_centers.pop()
            filtered_boxes.pop()
            print('Vision obj pos')
        else:
            real_obj_pos = np.random.uniform(low=[0.368, -0.25, 0.91], high=[0.72, 0.25, 0.91])
            real_yaw = np.random.uniform(low = -3.14, high = 0)
            print('Random obj pos')            
        
        #obs = env.reset()
        #obs, env_info = open_gripper(env, obs)
        print('real obj pos {}'.format(real_obj_pos))
        pi.set_real_obj_pos(real_obj_pos)
        pi.set_real_tar_pos(real_tar_pos)
        pi.set_real_yaw(real_yaw)

        paths = env.examine_policy_new(
            policy=pi,
            horizon=env.spec.max_episode_steps,
            num_episodes=1,
            frame_size=(640,480),
            mode=mode,
            output_dir=output_dir+'/',
            filename=output_name,
            camera_name=camera_name,
            render=render)        
        obs = env.get_obs(update_exteroception=True)
        
        rollouts += 1

        mean_diff, new_img, grasp_success, pre_drop_img, post_drop_img = check_grasp_success(env, obs, 
                                                                                            force_img=force_check or grasp_centers is None or len(grasp_centers) <= 0 )
        
        if new_img is not None:
            latest_img = new_img

        # Determine success
        if grasp_success:

            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = output_dir + '/' + output_name + '{}_paths.pickle'.format(time_stamp)
            paths.close()
            paths.save(trace_name=file_name, verify_length=True, f_res=np.float64)
            successes += 1

        print('Success {} ({}/{})'.format(successes/rollouts,successes,rollouts))
    '''

if __name__ == '__main__':
    main()
