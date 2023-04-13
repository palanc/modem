import click
import numpy as np
import time
import os
from robohive.logger.grouped_datasets import Trace
'''
RENDER_KEYS = ["env_infos/visual_dict/rgb:left_cam:240x424:2d",
               "env_infos/visual_dict/d:left_cam:240x424:2d",
               "env_infos/visual_dict/rgb:right_cam:240x424:2d",
               "env_infos/visual_dict/d:right_cam:240x424:2d",
               "env_infos/visual_dict/rgb:top_cam:240x424:2d",
               "env_infos/visual_dict/d:top_cam:240x424:2d",
               "env_infos/visual_dict/rgb:Franka_wrist_cam:240x424:2d",
               "env_infos/visual_dict/d:Franka_wrist_cam:240x424:2d"]
'''
'''
RENDER_KEYS = ["env_infos/visual_dict/rgb:top_cam:240x424:2d",
               "env_infos/visual_dict/d:top_cam:240x424:2d"]
'''
RENDER_KEYS = ["env_infos/visual_dict/rgb:left_cam:240x424:2d",
               "env_infos/visual_dict/d:left_cam:240x424:2d",
               "env_infos/visual_dict/rgb:right_cam:240x424:2d",
               "env_infos/visual_dict/d:right_cam:240x424:2d",]

FPS = 12.5

DESC='Render a trace into a .mp4'
@click.command(help=DESC)
@click.option('-tp', '--trace_path', type=str, help='absolute path to trace to load', required=True)
def main(trace_path):
    paths = Trace.load(trace_path)
    paths.close()
    output_dir = '/'.join(trace_path.split('/')[:-1])
    paths.render(output_dir=output_dir, output_format="mp4", groups=":", datasets=RENDER_KEYS, input_fps=FPS)

if __name__ == '__main__':
    main()