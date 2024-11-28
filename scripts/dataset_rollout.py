#!/home/george/miniforge3/envs/polymetis-local/bin/python
import os
import sys
import rospy
import tf2_ros
import torch
from torch.nn import functional as F
import numpy as np
import click
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import yaml
import h5py

import hydra
import dill

# from pathlib import Path
# home = Path.home()
sys.path.append('/home/george/diffusion_policy')

from franka_env import FrankaEnv
from sklearn.mixture import GaussianMixture
from OCR.models.gmm_grad import GMMGradient
from OCR.utils.utils_3d_compat import to_obj_pose, gen_keypoints, abs_traj, obs_quat_to_rot6d, \
                                        abs_se3_vector, deabs_se3_vector, rotation_6d_to_matrix, \
                                            quaternion_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion
from diffusion_policy.common.pytorch_util import dict_apply



def load_policy(ckpt, device, output_dir):
    # load checkpoint
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    return policy, cfg


def load_recovery_gradient(rec_cfg):
    gmms = []
    for i in range(3):
        gmms.append(GaussianMixture(n_components=rec_cfg['gmm']["n_components"]))

    gmms_params = np.load(rec_cfg['gmm']['params_used'], allow_pickle=True)
    for i in range(len(gmms_params)):
        for (key,val) in gmms_params[str(i)][()].items():
            setattr(gmms[i], key, val)

    rec_grad_generator = GMMGradient(gmms_params)
    return rec_grad_generator


def add_obs(new_obs_dict, past_obs_dict, n_obs_steps):
    for key in new_obs_dict.keys():
        new_obs = new_obs_dict[key][None]
        past_obs = past_obs_dict[key][0]
        if len(past_obs) < 1:
            past_obs = np.repeat(new_obs, n_obs_steps, 0)
        else:
            old_obs = past_obs[:-1]
            past_obs = np.vstack((new_obs,old_obs))
        past_obs_dict[key] = past_obs[None].astype(np.float32)
    return past_obs_dict


def generate_kp_traj(kp_start, recovery_vec, horizon, delay, alpha=0.01):
    n_kp,d_kp = kp_start.shape
    kp_base = np.repeat([kp_start], horizon, axis=0) # horizon,n_kp,D
    mean_recovery_vec = recovery_vec.mean(axis=0) * alpha
    motion_vecs = np.repeat([mean_recovery_vec], horizon-delay, axis=0) 
    motion_vecs = np.vstack((np.zeros((delay, d_kp)),motion_vecs)) # horizon,D
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), n_kp, axis=0).reshape(horizon, n_kp, d_kp)
    return kp_base + vecs



@click.command()
@click.option('-da', '--dataset_path', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(dataset_path, output_dir, device):
    recovery_config_path = '/home/george/diffusion_policy/OCR/config/bottle_recovery.yaml'
    with open(recovery_config_path, 'r') as f:
        recovery_config = yaml.safe_load(f)

    # load translator policy from checkpoint
    translator_policy, translator_cfg = load_policy(recovery_config['translator_policy']['checkpoint_used'], device, output_dir)
    # load base policy from checkpoint
    base_policy, base_cfg = load_policy(recovery_config['base_policy']['checkpoint_used'], device, output_dir)
    
    # rec_grad_generator = load_recovery_gradient(recovery_config)

    env = FrankaEnv()
    rate = rospy.Rate(7.0)

    OOD_THRESHOLD = 30
    action_horizon = 12
    n_obs_steps = base_cfg.n_obs_steps
    delay = 16
    gripper = -1
    
    past_obs = {
        'object': [[]],
        'agentview_image': [[]],
        'robot0_eef_pos': [[]],
        'robot0_eef_quat': [[]],
        'robot0_gripper_qpos': [[]],
    }

    dataset = h5py.File(dataset_path, 'r')
    demo_num = 0
    demo_data = dataset['data'][f'demo_{demo_num}']
    counter = 0

    # env policy rollout
    obs = env.reset(np.hstack((demo_data['actions'][0,:3], R.from_rotvec(demo_data['actions'][0,3:6]).as_quat())))
    # obs = env.reset()
    # past_obs = add_obs(obs, past_obs, n_obs_steps)
    while not rospy.is_shutdown():
        try:

            cur_obj_poses = to_obj_pose(demo_data['obs']['object']\
                                        [counter*translator_cfg.n_obs_steps:(counter+1)*translator_cfg.n_obs_steps,:7])
            cur_kps = gen_keypoints(cur_obj_poses) # H,n_kp,D_kp
            # densities, rec_vectors = rec_grad_generator(cur_kp)
            # print(np.mean(densities))
            
            # if np.mean(densities) < OOD_THRESHOLD:
            if True:
                ### Case: ODD
                # rec_vectors = rec_vectors.reshape(cur_kp.shape[1:])
                # kp_traj = generate_kp_traj(cur_kp[0], rec_vectors, horizon=16, delay=delay, alpha=0.0001) # H,n_kp,D_kp
                # if delay > 0: delay -= 1
                abs_kp = abs_traj(cur_kps, cur_obj_poses[0])

                cur_rot6d = matrix_to_rotation_6d(R.from_quat(obs['robot0_eef_quat']).as_matrix()[None])[0]
                cur_se3 = np.concatenate((obs['robot0_eef_pos'], cur_rot6d))[None]
                cur_action = np.hstack((abs_se3_vector(cur_se3, cur_obj_poses[0]), np.array([[gripper]])))

                # cur_rot6d = matrix_to_rotation_6d(R.from_rotvec(demo_data['actions'][counter*n_obs_steps][3:6]).as_matrix()[None])[0]
                # cur_se3 = np.concatenate((demo_data['actions'][counter*n_obs_steps][:3], cur_rot6d))[None]
                # cur_action = np.hstack((abs_se3_vector(cur_se3, cur_obj_poses[0]), np.array([[gripper]])))

                np_obs_dict = {
                    'obs': abs_kp.reshape(translator_cfg.horizon,-1)[None].astype(np.float32),
                    'init_action': cur_action.astype(np.float32)
                }
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = translator_policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                np_action = np_action_dict['action_pred'].squeeze(0)
                detrans_np_action = deabs_se3_vector(np_action[:,:9], cur_obj_poses[0])
                detrans_np_action = np.hstack((detrans_np_action[:,:3], 
                                                matrix_to_quaternion(rotation_6d_to_matrix(detrans_np_action[:,3:9]))[:,[1,2,3,0]],
                                                np_action[:,9:]))
                # detrans_np_action = np.hstack((detrans_np_action[:,:9], 
                #                                 np_action[:,9:]))

                vis_traj(cur_kps, 
                         detrans_np_action, 
                         demo_data['obs']['object'][counter*translator_cfg.n_obs_steps:(counter+1)*translator_cfg.n_obs_steps], 
                         demo_data['actions'][counter*translator_cfg.n_obs_steps:(counter+1)*translator_cfg.n_obs_steps], 
                         np.hstack((obs['robot0_eef_pos'],obs['robot0_eef_quat'])))
                # vis_traj(abs_kp, 
                #          np.hstack((np_action[:,:3], 
                #             matrix_to_quaternion(rotation_6d_to_matrix(np_action[:,3:9]))[:,[1,2,3,0]],
                #             np_action[:,9:])), 
                #          np.hstack((obs['robot0_eef_pos'],obs['robot0_eef_quat'])))

                # step env and render
                for i in range(action_horizon):
                    act = detrans_np_action[i]
                    # act = np.hstack((act[:3], R.from_matrix(rotation_6d_to_matrix(act[3:9][None])[0]).as_quat(), act[9:]))
                    gripper = act[-1]
                    print(gripper)
                    obs = env.step(act)
                    past_obs = add_obs(obs, past_obs, n_obs_steps)
                    rate.sleep()

            else:
                ## Case: ID
                np_obs_dict = {
                    'obs': past_obs,
                }
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = base_policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # action = np.hstack((action[:,:3], 
                #                 rot6d2quat.forward(action[:,3:9])[:,[1,2,3,0]],
                #                 action[:,9:]))
                
                # step env and render
                for i in range(action.shape[0]):
                    act = action[i]
                    act = np.hstack((act[:3], R.from_matrix(rotation_6d_to_matrix(act[3:9][None])[0]).as_quat(), act[9:]))
                    gripper = act[-1]
                    obs = env.step(act)
                    past_obs = add_obs(obs, past_obs, n_obs_steps)
                    rate.sleep()
            counter += 1
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Transform not available")
            

def draw_frame_axis(T, ax, color, length=0.05, alpha=1.0):
    if ax is None:
        return
    
    x_axis = T_multi_vec(T, np.array([length,    0,    0]))
    y_axis = T_multi_vec(T, np.array([0,    length,    0]))
    z_axis = T_multi_vec(T, np.array([0,    0,    length]))

    center = T_multi_vec(T, np.array([0.0, 0.0, 0.0]))
    stack_x = np.vstack((center, x_axis))
    stack_y = np.vstack((center, y_axis))
    stack_z = np.vstack((center, z_axis))

    ax.plot(stack_x[:,0], stack_x[:,1], stack_x[:,2], color=color, alpha=alpha)
    ax.plot(stack_y[:,0], stack_y[:,1], stack_y[:,2], color=color, alpha=alpha)
    ax.plot(stack_z[:,0], stack_z[:,1], stack_z[:,2], color=color, alpha=alpha)

def T_multi_vec(T, vec):
    vec = vec.flatten()
    return (T @ np.append(vec, 1.0).reshape(-1,1)).flatten()[:3]

def posquat_to_pose(x):
    pose = np.eye(4)
    pose[:3,:3] = R.from_quat(x[3:7]).as_matrix()
    pose[:3,3] = x[:3]
    return pose

def posrotvec_to_pose(x):
    pose = np.eye(4)
    pose[:3,:3] = R.from_rotvec(x[3:6]).as_matrix()
    pose[:3,3] = x[:3]
    return pose

def vis_traj(object_kps, action_poses, true_objects, true_actions, cur_action):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    fig_lims = 0.6

    def animate(args):
        object_kp, action_pose, true_object, true_action, cur_action= args
        action_pose = posquat_to_pose(action_pose[:7])
        cur_pose = posquat_to_pose(cur_action[:7])
        true_action_pose = posrotvec_to_pose(true_action[:6])
        true_object_pose = posquat_to_pose(true_object[:7])
        ax.cla()
        for i in range(len(object_kp)):
            ax.scatter(object_kp[i,0], object_kp[i,1], object_kp[i,2], color=plt.cm.rainbow(0/len(object_kp)), s=15)
        draw_frame_axis(action_pose, ax, 'green', 0.15)
        draw_frame_axis(cur_pose, ax, 'purple', 0.15)
        draw_frame_axis(true_object_pose, ax, 'blue', 0.15)
        draw_frame_axis(true_action_pose, ax, 'red', 0.15)
        ax.set_xlim(-fig_lims, fig_lims)
        ax.set_ylim(-fig_lims, fig_lims)
        ax.set_zlim(-fig_lims, fig_lims)

    cur_actions = np.repeat(cur_action[None], len(action_poses), 0)
    ani = FuncAnimation(fig, animate, frames=zip(object_kps,action_poses,true_objects,true_actions,cur_actions), interval=100, save_count=sys.maxsize)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
