import os
import gym
import cv2
import numpy as np

MAX_GRIPPER_OPEN = 0.0002
MIN_GRIPPER_CLOSED = 0.8228
DROP_ZONE = np.array([0.53, 0.0, 1.1])
DROP_ZONE_PERTURB = np.array([0.025, 0.025,0.0])
OUT_OF_WAY = np.array([0.3438, -0.9361,  0.0876, -2.8211,  0.0749,  0.5144,  1.8283])

PIX_FROM_LEFT = 73-116
PIX_FROM_TOP = 58-16
DIST_FROM_CENTER = 1.0668/2
DIST_FROM_BASE = 0.72#0.7493
#Y_SCALE = -0.5207/152 # 0.5207 is length of bin 
#X_SCALE = 1.0668/314 # 1.0668 is width of bin
SCALE = 0.0034

MASK_START_X = 148-116 #40
MASK_END_X = 313-116 #400
MASK_START_Y = 57-16 #30
MASK_END_Y = 170-16 #220

DIFF_THRESH = 0.15

def is_moving(prev, cur, tol):
    return np.linalg.norm(cur-prev) > tol

def cart_move(action, env):
    last_pos = None
    action = 2*(((action-env.pos_limits['eef_low'])/(np.abs(env.pos_limits['eef_high']-env.pos_limits['eef_low'])+1e-8))-0.5)

    for _ in range(1000):
        obs, _, _, env_info = env.step(action, update_exteroception=True)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        pos = obs_dict['qp'][0,0,:7]
        if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
            break
        last_pos = pos

def move_joint_config(env, config):
    last_pos = None

    jnt_low = env.sim.model.jnt_range[:env.sim.model.nu, 0]
    jnt_high = env.sim.model.jnt_range[:env.sim.model.nu, 1]

    config = 2*(((config-jnt_low)/(np.abs(jnt_high-jnt_low)+1e-8))-0.5)

    for _ in range(1000):

        obs, _, _, env_info = env.step(config, update_exteroception=True)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        pos = obs_dict['qp'][0,0,:7]
        if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
            break
        last_pos = pos

    return obs, env_info

def open_gripper(env, obs):
    obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
    open_qp = obs_dict['qp'][0,0,:9].copy()
    open_qp[7:9] = 0.0
    
    jnt_low = env.sim.model.jnt_range[:env.sim.model.nu, 0]
    jnt_high = env.sim.model.jnt_range[:env.sim.model.nu, 1]    
    release_action = 2*(((open_qp - jnt_low)/(jnt_high-jnt_low))-0.5)
    
    start_time = time.time()

    while((obs_dict['qp'][0,0,7] > MAX_GRIPPER_OPEN) and time.time()-start_time < 30.0):
        
        obs, _, done, env_info = env.step(release_action, update_exteroception=True)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
    return obs

def check_grasp_success(env, obs, force_img=False, just_drop=False):
    failed_grasp = False
    if obs is None:
        obs = env.get_obs(update_exteroception=True)
    
    obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
    if obs_dict['qp'][0,0,7] < MAX_GRIPPER_OPEN:
        failed_grasp = True
        print('Policy didnt close gripper')
        if not force_img:
            return None, None, False, None, None

    if obs_dict['grasp_pos'][0,0,2] < 1.0:
        failed_grasp = True
        print('Policy didnt lift gripper')
        obs = open_gripper(env, obs)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        if not force_img:
            return None, None, False, None, None

    print('moving up')
    des_grasp_pos = obs_dict['grasp_pos'][0,0,:].copy()
    des_grasp_xy = des_grasp_pos[:2]
    des_grasp_height = 1.2        
    move_up_tries = 4
    for i in range(1,move_up_tries+1):
        
        move_up_steps = 0
        while(obs_dict['grasp_pos'][0,0,2] < 1.1 and move_up_steps < 25):
            move_up_action = np.concatenate([des_grasp_pos, [3.14,0.0,0.0,obs_dict['qp'][0,0,7]]])
            des_grasp_pos[:2] = ((move_up_tries-i)/(move_up_tries))*des_grasp_xy+((i)/(move_up_tries))*(DROP_ZONE[:2]) 
            des_grasp_pos[:2] = min(max(des_grasp_pos[:2], obs_dict['grasp_pos'][0,0,1]-0.1),obs_dict['grasp_pos'][0,0,1]+0.1) 
            move_up_action[2] = min(1.2, obs_dict['grasp_pos'][0,0,2]+0.1)
            move_up_action = 2*(((move_up_action - env.pos_limits['eef_low']) / (np.abs(env.pos_limits['eef_high'] - env.pos_limits['eef_low'])+1e-8)) - 0.5)
            obs, _, done, _ = env.step(move_up_action, update_exteroception=True)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1))) 
            move_up_steps += 1 
        if obs_dict['grasp_pos'][0,0,2] >= 1.1:
            break

    obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))      

    print('Grip Width {}'.format(obs_dict['qp'][0,0,7]))
    grip_width = obs_dict['qp'][0,0,7]
    mean_diff = 0.0

    if (grip_width < MAX_GRIPPER_OPEN or grip_width > MIN_GRIPPER_CLOSED):
        failed_grasp = True
        obs = open_gripper(env, obs)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        if not force_img:
            return None, None, False, None, None

    pre_drop_img = None
    # Get top cam key
    top_cam_key = None
    for key in obs_dict.keys():
        if 'top' in key:
            top_cam_key = key
            break
    assert(top_cam_key is not None)

    if not just_drop and not failed_grasp:

        obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]])) 

        # Wait for stabilize
        time.sleep(3)
        obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]]))   

        pre_drop_img = env_info['obs_dict'][top_cam_key]

    if just_drop or not failed_grasp:

        print('Moving to drop zone')
        drop_zone_pos = np.random.uniform(low=DROP_ZONE-DROP_ZONE_PERTURB, high=DROP_ZONE+DROP_ZONE_PERTURB)
        drop_zone_yaw = -np.pi*np.random.rand()
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        last_pos = None
        drop_zone_steps = 0
        while(drop_zone_steps < 100):
            drop_zone_action = np.concatenate([drop_zone_pos, [3.14,0.0,drop_zone_yaw,obs_dict['qp'][0,0,7]]])
            drop_zone_action = 2*(((drop_zone_action - env.pos_limits['eef_low']) / (np.abs(env.pos_limits['eef_high'] - env.pos_limits['eef_low'])+1e-8)) - 0.5)
            obs, _, done, _ = env.step(drop_zone_action, update_exteroception=True)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))    
            pos = obs_dict['qp'][0,0,:7]
            if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
                break
            last_pos = pos         
            drop_zone_steps += 1       

        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        if np.linalg.norm(obs_dict['grasp_pos'][0,0,:2] - drop_zone_pos[:2]) > 0.1:
            #return None, None
            print('Rand drop failed, moving to init qpos for drop')
            obs, env_info = move_joint_config(env, np.concatenate([env.init_qpos[:7], [obs_dict['qp'][0,0,7]]]))
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))

        open_qp = obs_dict['qp'][0,0,:8].copy()
        open_qp[7:8] = 0.0

        print('Releasing')
        extra_time = 25
        start_time = time.time()
        while(((obs_dict['qp'][0,0,7] > 0.001) or extra_time > 0) and time.time()-start_time < 30.0):
            release_action = 2*(((open_qp - env.jnt_low)/(env.jnt_high-env.jnt_low))-0.5)
            obs, _, done, env_info = env.step(release_action, update_exteroception=True)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
            extra_time -= 1

        drop_pos = obs_dict['grasp_pos'][0,0]

        drop_x = int(PIX_FROM_LEFT + (drop_pos[1]+DIST_FROM_CENTER)/SCALE)
        drop_y = int(PIX_FROM_TOP + (DIST_FROM_BASE-drop_pos[0])/SCALE)

        print('drop_pos x: {}, drop_pos y: {}'.format(drop_zone_pos[0], drop_zone_pos[1]))
        if pre_drop_img is not None:
            success_mask = np.zeros(pre_drop_img.shape, dtype=np.uint8)
            success_start_x = max(MASK_START_X, drop_x - 30)
            success_end_x = min(MASK_END_X, drop_x + 30)
            success_start_y = max(MASK_START_Y, drop_y - 30)
            success_end_y = min(MASK_END_Y, drop_y+30)
            
            print('drop_x {}, drop_y {}, start_x {}, end_x {}, start_y {}, end_y {}'.format( drop_x, drop_y, success_start_x, success_end_x, success_start_y, success_end_y))
            success_mask[success_start_y:success_end_y, success_start_x:success_end_x, :] = 255
            pre_drop_img = cv2.bitwise_and(pre_drop_img, success_mask)

    latest_img = None
    post_drop_img = None
    if (not just_drop and not failed_grasp) or force_img:
        print('moving out of way')
        obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]])) 
        time.sleep(3)
        obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]]))

        latest_img = env_info['obs_dict'][top_cam_key].copy()

    if pre_drop_img is not None and latest_img is not None and success_mask is not None:
        post_drop_img = cv2.bitwise_and(latest_img, success_mask)
        mean_diff = np.mean(np.abs(post_drop_img.astype(float)-pre_drop_img.astype(float)))
        print('Mean img diff: {}'.format(mean_diff))    
    else:
        mean_diff =  0.0

    return mean_diff, latest_img, mean_diff > DIFF_THRESH, pre_drop_img, post_drop_img

def update_grasps(img, obj_pos_low, obj_pos_high, out_dir=None):
    if out_dir is not None and not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    bin_mask = np.zeros(img.shape, dtype=np.uint8)

    bin_mask[MASK_START_Y:MASK_END_Y, MASK_START_X:MASK_END_X, :] = 255
    img_masked = cv2.bitwise_and(img, bin_mask)
    #img_masked_fn = os.path.join(out_dir, out_name+'_masked.png')
    #cv2.imwrite(img_masked_fn, img_masked)

    gray_img = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 
                                    255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    15,
                                    15)
    _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)

    # first box is background
    boxes = boxes[1:]
    filtered_boxes = []
    rec_img = img_masked.copy()
    for x,y,w,h,pixels in boxes:
        if pixels > 15 and h > 4 and w > 4:#pixels < 1000 and h < 40 and w < 40 and h > 4 and w > 4:
            filtered_boxes.append((x,y,w,h))
            cv2.rectangle(rec_img, (x,y), (x+w, y+h), (255,0,0), 1)

    if out_dir is not None:
        rec_img_fn = os.path.join(out_dir,'masked_all_recs.png')
        cv2.imwrite(rec_img_fn, cv2.cvtColor(rec_img, cv2.COLOR_RGB2BGR))
    
    
    random.shuffle(filtered_boxes)


    #for x,y,w,h in filtered_boxes:
    #    cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,0,255), 1)

    #rec_img_fn = os.path.join(out_dir, out_name+'recs.png')
    #cv2.imwrite(rec_img_fn, img_masked)

    grasp_centers = []
    for x,y,w,h in filtered_boxes:

        grasp_x = SCALE * (PIX_FROM_TOP - (y+(h/2.0))) + DIST_FROM_BASE
        grasp_y = SCALE * ((x+(w/2.0)) - PIX_FROM_LEFT) - DIST_FROM_CENTER

        if (grasp_x >= obj_pos_low[0] and grasp_x <= obj_pos_high[0] and
            grasp_y >= obj_pos_low[1] and grasp_y <= obj_pos_high[1]):
            grasp_centers.append((grasp_x,grasp_y))
    return grasp_centers, filtered_boxes, img_masked