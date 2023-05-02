import os
import gym
import cv2
import numpy as np
import random
import time
from sklearn.decomposition import PCA

MAX_GRIPPER_OPEN = 0.0002
MIN_GRIPPER_CLOSED = 0.8228
DROP_ZONE = np.array([0.53, 0.0, 1.1])
# DROP_ZONE = np.array([0.575, 0.0, 1.1]) - better yaw setting
DROP_ZONE_PERTURB = np.array([0.025, 0.025,0.0])
OUT_OF_WAY = np.array([0.3438, -0.9361,  0.0876, -2.8211,  0.0749,  0.5144, -1.57])

PIX_FROM_LEFT = 73
PIX_FROM_TOP = 58
DIST_FROM_CENTER = 1.0668/2
DIST_FROM_BASE = 0.7493#0.72
X_SCALE = 0.5207/152 # 0.5207 is length of bin 
Y_SCALE = 1.0668/314 # 1.0668 is width of bin

MASK_START_X = 148 #40
MASK_END_X = 313 #400
MASK_START_Y = 57 #30
MASK_END_Y = 170 #220

DIFF_THRESH = 0.15

OBJ_POS_LOW = [0.368, -0.25, 0.91] #[-0.35,0.25,0.91]
OBJ_POS_HIGH = [0.72, 0.25, 0.91] #[0.35,0.65,0.91]

def get_drop_zone_limits():
    low = DROP_ZONE - DROP_ZONE_PERTURB
    high = DROP_ZONE + DROP_ZONE_PERTURB
    return low, high

def is_moving(prev, cur, tol):
    return np.linalg.norm(cur-prev) > tol

def cart_move(action, env):
    last_pos = None
    action = 2*(((action-env.pos_limits['eef_low'])/(np.abs(env.pos_limits['eef_high']-env.pos_limits['eef_low'])+1e-8))-0.5)

    for _ in range(1000):
        obs, _, _, env_info = env.unwrapped.step(action, update_exteroception=True)
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

        obs, _, _, env_info = env.unwrapped.step(config, update_exteroception=True)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        pos = obs_dict['qp'][0,0,:7]
        if last_pos is not None and not is_moving(last_pos, pos, 0.0001):
            break
        last_pos = pos

    return obs, env_info

def open_gripper(env, obs):
    env_info = env.get_env_infos()
    obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
    open_qp = obs_dict['qp'][0,0,:8].copy()
    open_qp[7] = 0.0
    
    jnt_low = env.sim.model.jnt_range[:env.sim.model.nu, 0]
    jnt_high = env.sim.model.jnt_range[:env.sim.model.nu, 1]    
    release_action = 2*(((open_qp - jnt_low)/(jnt_high-jnt_low))-0.5)
    
    start_time = time.time()

    while((obs_dict['qp'][0,0,7] > MAX_GRIPPER_OPEN) and time.time()-start_time < 30.0):
        obs, _, done, env_info = env.unwrapped.step(release_action, update_exteroception=True)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
    return obs, env_info

def move_to(action, env):

    last_qp = None
    obs_dict = None
    while last_qp is None or np.linalg.norm(obs_dict['qp'][0,0,:env.sim.model.nu]-last_qp) > 0.01:
        move_action =  2*(((action - env.pos_limits['eef_low']) / (np.abs(env.pos_limits['eef_high'] - env.pos_limits['eef_low'])+1e-8)) - 0.5)
        
        if obs_dict is not None:
            last_qp = obs_dict['qp'][0,0,:env.sim.model.nu].copy()

        obs, _, done, _ = env.unwrapped.step(move_action, update_exteroception=True)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1))) 

def check_reorient_success(env, obs, out_dir='/tmp'):
    jnt_low = env.sim.model.jnt_range[:env.sim.model.nu, 0]
    jnt_high = env.sim.model.jnt_range[:env.sim.model.nu, 1]    
    #OUT_OF_WAY_HAND = np.array([0.0,1.5,0.0,0.0,
    #                            0.0,1.5,0.0,
    #                            0.0,1.5,0.0])
    OUT_OF_WAY_HAND = np.array([0.0,-0.75,0.0,-0.2,
                                0.0,-0.75,-0.2,
                                0.0,-0.75,-0.2])
    KNOCK_HAND = np.array([0.0,1.5,0.0,-0.2,
                                0.75,1.5,-0.2,
                                -0.75,1.5,-0.2])
    DRAG_HAND = np.array([0.57,1.5,0.2,0.5, # Thumb
                            0.0,1.5,0.0,     # Middle
                            -0.0,1.5,0.2])    # Pinky

    if obs is None:
        obs = env.get_obs(update_exteroception=True)    
    obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
    env_info = env.get_env_infos()

    # Get top cam key
    top_rgb_cam_key = None
    top_d_cam_key = None
    for key in env_info['visual_dict'].keys():
        if 'top' in key :
            if key[:4] == 'rgb:':
                top_rgb_cam_key = key
            elif key[:2] == 'd:':
                top_d_cam_key = key        

    assert(top_rgb_cam_key is not None and top_d_cam_key is not None)

    out_of_way_jnts = np.concatenate([OUT_OF_WAY, OUT_OF_WAY_HAND])
    out_of_way_jnts[5] = 1.57
    out_of_way_jnts[6] = -3*np.pi/4

    reorient_success = None
    knocked_over = False
    while not knocked_over:

        # Check for success
        obs, env_info = move_joint_config(env, out_of_way_jnts) 
        # Wait for stabilize
        time.sleep(3)
    
        obs, env_info = move_joint_config(env, out_of_way_jnts)       
        
        success_img = np.array(np.clip(env_info['visual_dict'][top_d_cam_key],0,255), dtype=np.uint8).reshape((240,424,1))
        bin_mask = np.zeros(success_img.shape, dtype=np.uint8)
        bin_mask[MASK_START_Y:MASK_END_Y, MASK_START_X:MASK_END_X, :] = 255
        success_img = cv2.bitwise_and(success_img, bin_mask)

        top_pixels = np.logical_and(success_img>0, success_img <= 45)
        thresh_val = top_pixels.sum()
        knocked_over = thresh_val < 250
        #from matplotlib import pyplot as plt
        #weights = np.ones_like(success_img)
        #weights[success_img == 0] = 0
        #plt.clf()
        #plt.xlim([0,100])
        #_ = plt.hist(success_img.flatten(), bins=255, weights=weights.flatten())
        #plt.savefig(out_dir+'/hist.png')

        y_top, x_top = np.where(top_pixels)
        y_top = np.mean(y_top)
        x_top = np.mean(x_top)
        if not knocked_over:
            success_img[(int(y_top)-5):(int(y_top)+5), (int(x_top)-5):(int(x_top)+5)] = 255
        success_img_fn = os.path.join(out_dir,'success_img.png')    
        cv2.imwrite(success_img_fn, success_img)

        if reorient_success is None:
            reorient_success = not knocked_over
            print('Reorient success {}'.format(reorient_success))
        
        obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, OUT_OF_WAY_HAND])) 

        if knocked_over:
            break

        knock_x = X_SCALE * (PIX_FROM_TOP - (y_top)) + DIST_FROM_BASE
        knock_y = Y_SCALE * (PIX_FROM_LEFT - (x_top)) + DIST_FROM_CENTER               
        knock_dir = np.array([0.55-knock_x, 0.0-knock_y])
        knock_dir = knock_dir / np.linalg.norm(knock_dir)
        knock_yaw = np.arctan2(knock_dir[0], knock_dir[1])
        preknock_action = np.concatenate([[knock_x, knock_y, 1.125, 3.14, 0.0, knock_yaw],
                                          OUT_OF_WAY_HAND])
        move_to(preknock_action, env)

        preknock_action = np.concatenate([[knock_x, knock_y, 1.125, 3.14, 0.0, knock_yaw],
                                           KNOCK_HAND])
        
        move_to(preknock_action, env)        

        knock_x = knock_x + 0.15*knock_dir[0]
        knock_y = knock_y + 0.15*knock_dir[1]
        knock_action = np.concatenate([[knock_x, knock_y, 1.125, 3.14, 0.0, knock_yaw],
                                           KNOCK_HAND])

        move_to(knock_action, env)       

    # Move knocked over bottle to middle
    while True:

        obs, env_info = move_joint_config(env, out_of_way_jnts) 
        # Wait for stabilize
        time.sleep(3)
        obs, env_info = move_joint_config(env, out_of_way_jnts)   
        reset_img = env_info['visual_dict'][top_rgb_cam_key]            
        break
        grasp_centers, filtered_boxes, img_masked = update_grasps(img=reset_img,
                                                                  out_dir=out_dir+'/debug',
                                                                  min_pixels=400,
                                                                  luv_thresh=True,
                                                                  limit_yaw=False)

        grasp_x = grasp_centers[0]
        grasp_y = grasp_centers[1]
        assert(False,"UPDATE YAW")
        grasp_yaw = grasp_centers[2]+np.pi/4

        drag_dir = np.array([0.55-grasp_x, 0.0-grasp_y])
        if np.linalg.norm(drag_dir) < 0.025:
            break

        predrag_action = np.concatenate([[grasp_x, grasp_y, 0.875, 3.14, 0.0, grasp_yaw],
                                           OUT_OF_WAY_HAND])
        
        move_to(predrag_action, env)        

        predrag_action = np.concatenate([[grasp_x, grasp_y, 0.875, 3.14, 0.0, grasp_yaw],
                                           DRAG_HAND])
        
        move_to(predrag_action, env)       
        grasp_yaw = np.random.uniform(0.0, np.pi)
        drag_action = np.concatenate([[0.55, 0.0, 0.925, 3.14, 0.0, grasp_yaw],
                                           DRAG_HAND])
        
        move_to(drag_action, env)       

        drag_action = np.concatenate([[0.55, 0.0, 0.925, 3.14, 0.0, grasp_yaw],
                                           OUT_OF_WAY_HAND])
        
        move_to(drag_action, env)       

    return reorient_success, reset_img

def check_grasp_success(env, obs, force_img=False, just_drop=False):
    jnt_low = env.sim.model.jnt_range[:env.sim.model.nu, 0]
    jnt_high = env.sim.model.jnt_range[:env.sim.model.nu, 1]    
    
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
        obs, env_info = open_gripper(env, obs)
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
            des_grasp_pos[:2] = np.clip(des_grasp_pos[:2], obs_dict['grasp_pos'][0,0,:2]-0.1,obs_dict['grasp_pos'][0,0,:2]+0.1) 
            move_up_action[2] = min(1.2, obs_dict['grasp_pos'][0,0,2]+0.1)
            move_up_action = 2*(((move_up_action - env.pos_limits['eef_low']) / (np.abs(env.pos_limits['eef_high'] - env.pos_limits['eef_low'])+1e-8)) - 0.5)
            obs, _, done, _ = env.unwrapped.step(move_up_action, update_exteroception=True)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1))) 
            move_up_steps += 1 
        if obs_dict['grasp_pos'][0,0,2] >= 1.1:
            break

    obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))      
    env_info = env.get_env_infos()

    print('Grip Width {}'.format(obs_dict['qp'][0,0,7]))
    grip_width = obs_dict['qp'][0,0,7]
    mean_diff = 0.0

    if (grip_width < MAX_GRIPPER_OPEN or grip_width > MIN_GRIPPER_CLOSED):
        failed_grasp = True
        obs, env_info = open_gripper(env, obs)
        obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
        if not force_img:
            return None, None, False, None, None

    pre_drop_img = None
    # Get top cam key
    top_cam_key = None
    for key in env_info['visual_dict'].keys():
        if 'top' in key:
            top_cam_key = key
            break
    assert(top_cam_key is not None)

    if not just_drop and not failed_grasp:

        obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]])) 

        # Wait for stabilize
        time.sleep(3)
        obs, env_info = move_joint_config(env, np.concatenate([OUT_OF_WAY, [obs_dict['qp'][0,0,7]]]))   

        pre_drop_img = env_info['visual_dict'][top_cam_key]

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
            obs, _, done, _ = env.unwrapped.step(drop_zone_action, update_exteroception=True)
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
            release_action = 2*(((open_qp - jnt_low)/(jnt_high-jnt_low))-0.5)
            obs, _, done, env_info = env.unwrapped.step(release_action, update_exteroception=True)
            obs_dict = env.obsvec2obsdict(np.expand_dims(obs, axis=(0,1)))
            extra_time -= 1

        drop_pos = obs_dict['grasp_pos'][0,0]

        drop_x = int(PIX_FROM_LEFT + (drop_pos[1]+DIST_FROM_CENTER)/Y_SCALE)
        drop_y = int(PIX_FROM_TOP + (DIST_FROM_BASE-drop_pos[0])/X_SCALE)

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

        latest_img = env_info['visual_dict'][top_cam_key].copy()

    if pre_drop_img is not None and latest_img is not None and success_mask is not None:
        post_drop_img = cv2.bitwise_and(latest_img, success_mask)
        mean_diff = np.mean(np.abs(post_drop_img.astype(float)-pre_drop_img.astype(float)))
        print('Mean img diff: {}'.format(mean_diff))    
    else:
        mean_diff =  0.0

    return mean_diff, latest_img, mean_diff > DIFF_THRESH, pre_drop_img, post_drop_img

def update_grasps(img, out_dir=None, min_pixels=9, luv_thresh=False, limit_yaw=True):
    if out_dir is not None and not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    bin_mask = np.zeros(img.shape, dtype=np.uint8)

    bin_mask[MASK_START_Y:MASK_END_Y, MASK_START_X:MASK_END_X, :] = 255
    img_masked = cv2.bitwise_and(img, bin_mask)
    img_masked_fn = os.path.join(out_dir, 'img_masked.png')
    gray_img = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

    if luv_thresh:
        luv_img = cv2.cvtColor(np.array(img_masked).astype('float32')/255, cv2.COLOR_RGB2Luv)
        from matplotlib import pyplot as plt
        plt.hist(luv_img[:,:,0].flatten(),bins=255,range=(1,luv_img[:,:,0].max()))
        plt.savefig(out_dir+'/luv_img_hist.png')
        binary_img = np.zeros_like(gray_img)
        binary_img[luv_img[:,:,0] > 7] = 255

    else:
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
        #if pixels > 15 and h > 4 and w > 4:
        if pixels > min_pixels and h > 3 and w > 3:
            print(pixels)
            filtered_boxes.append((x,y,w,h))
            cv2.rectangle(rec_img, (x,y), (x+w, y+h), (255,0,0), 1)

    if out_dir is not None:
        bin_img_fn = os.path.join(out_dir, 'binary_thresh.png')
        cv2.imwrite(bin_img_fn, binary_img)

        rec_img_fn = os.path.join(out_dir,'masked_all_recs.png')
        cv2.imwrite(rec_img_fn, cv2.cvtColor(rec_img, cv2.COLOR_RGB2BGR))
    
    
    random.shuffle(filtered_boxes)


    #for x,y,w,h in filtered_boxes:
    #    cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,0,255), 1)

    #rec_img_fn = os.path.join(out_dir, out_name+'recs.png')
    #cv2.imwrite(rec_img_fn, img_masked)

    grasp_centers = []
    pca = PCA(n_components=1)
    for i, (x,y,w,h) in enumerate(filtered_boxes):

        grasp_x = X_SCALE * (PIX_FROM_TOP - (y+(h/2.0))) + DIST_FROM_BASE
        grasp_y = Y_SCALE * (PIX_FROM_LEFT - (x+(w/2.0))) + DIST_FROM_CENTER

        # Compute yaw
        if luv_thresh:
            yaw_thresh = binary_img[y:y+h, x:x+w]
        else:
            _, yaw_thresh = cv2.threshold(gray_img[y:y+h, x:x+w],
                                        min(int(np.mean(gray_img[y:y+h, x:x+w])),60),
                                        255, 
                                        cv2.THRESH_BINARY)
        pca.fit(np.transpose(np.nonzero(yaw_thresh > 128)))
        yaw_img = img.copy()
        yaw_img[y:y+h, x:x+w,:] = yaw_thresh[:,:,np.newaxis]
        yaw_img = cv2.line(yaw_img, 
                           (int(x+w/2.0), int(y+h/2.0)), 
                           (int(x+w/2.0+25*pca.components_[0][1]),int(y+h/2.0+25*pca.components_[0][0])),
                           (255,0,0),
                           2)
        yaw_img_fn = os.path.join(out_dir,'yaw_thresh{}.png'.format(i))
        cv2.imwrite(yaw_img_fn, yaw_img)

        yaw = np.arctan2(pca.components_[0][0], pca.components_[0][1]) + np.pi/4.0
        if limit_yaw:
            while yaw > 0:
                yaw -= np.pi
            while yaw < -np.pi:
                yaw += np.pi
        print('Predicted yaw {}'.format(yaw))

        if (grasp_x >= OBJ_POS_LOW[0] and grasp_x <= OBJ_POS_HIGH[0] and
            grasp_y >= OBJ_POS_LOW[1] and grasp_y <= OBJ_POS_HIGH[1]):
            grasp_centers.append((grasp_x,grasp_y, yaw))
    return grasp_centers, filtered_boxes, img_masked

def test_update_grasps():
    # Load image
    IMG_DIR = '/mnt/nfs_code/robopen_users/plancaster/robohive_base/modem/modem/utils/test/example_top_cam'
    img = cv2.imread(IMG_DIR+'/img_masked04.png')
    update_grasps(img, IMG_DIR)

if __name__ == '__main__':
    test_update_grasps()
