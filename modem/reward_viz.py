import os
import hydra
from cfg_parse import parse_cfg
import os
import signal
import site
from collections import deque
import time

import capnp
import a0
import numpy as np
import cv2
from env import make_env
import robohive
from modem.utils.color_threshold import ColorThreshold

filepath = os.path.dirname(os.path.abspath(__file__))
latest_img = None

def onmsg(pkt):
    global latest_img
    headers = dict(pkt.headers)
    #print(f'[headers]')
    #print(dump(headers, indent=2))
    if 'rostype' in headers:
        rostype=headers["rostype"]
        print(f'[payload] ros:{rostype}')
        pkg, msg_type = rostype.split('/') 
        msgs_capnp = capnp.load(os.path.join(filepath, f'../msgs/def/{pkg}.capnp'), imports=site.getsitepackages())

        msg = getattr(msgs_capnp, msg_type).from_bytes(pkt.payload)
        msg_dict = msg.to_dict(verbose=True)
        print(dump(msg_dict, indent=4))
    else:
        #print('[payload] raw')
        img = np.frombuffer(pkt.payload,dtype=np.uint8)[-305280:].reshape(240,424,3)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if latest_img is None:
            latest_img = img
        #print(f'  {byte_to_string(pkt.payload)}\n')

@hydra.main(config_name="config", config_path="cfgs")
def reward_viz(cfg: dict):
    global latest_img

    cfg = parse_cfg(cfg)
    cfg.real_robot = False
    env = make_env(cfg)
    camera_name = env.camera_names[0]
    left_crop = env.cfg.left_crops[0]
    top_crop = env.cfg.top_crops[0]
    topic_name = None
    img_size = 224
    for name, device in env.unwrapped.robot.robot_config.items():
        if name != camera_name:
            continue
        topic_name = device['interface']['rgb_topic']
        break
    assert(topic_name is not None)
    print('topic name {}'.format(topic_name))
    sub = a0.Subscriber(topic_name, a0.INIT_MOST_RECENT, a0.ITER_NEXT, onmsg)
    #print('left {} right {} top {} bottpm {}'.format(env.success_mask['left'],env.success_mask['right'],env.success_mask['top'],env.success_mask['bottom']))
    col_thresh = ColorThreshold(cam_name=env.camera_names[0], 
                                left_crop=env.cfg.success_mask_left, 
                                right_crop=env.cfg.success_mask_right, 
                                top_crop=env.cfg.success_mask_top, 
                                bottom_crop=env.cfg.success_mask_bottom, 
                                thresh_val=env.cfg.success_thresh, 
                                target_uv=np.array(env.cfg.success_uv),
                                render=True)
    max_angle = 0
    min_angle = np.inf
    try:
        while True:
            if latest_img is not None:
                latest_img = latest_img[top_crop:top_crop+img_size, left_crop:left_crop+img_size, :]
                success, luv_angle = col_thresh.detect_success(latest_img)
                if luv_angle > max_angle:
                    max_angle = luv_angle
                if luv_angle < min_angle:
                    min_angle = luv_angle
                print('Success {}, angle {}, max {}, min {}'.format(success, luv_angle, max_angle, min_angle))
                #cv2.imshow("image", latest_img)
                #cv2.waitKey(1)
                latest_img = None
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    reward_viz()