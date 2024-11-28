#!/home/george/miniforge3/envs/polymetis-local/bin/python
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image as Imgs
from cv_bridge import CvBridge
import cv2

import torch
import numpy as np
from polymetis import RobotInterface
from polymetis import GripperInterface

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt


class FrankaEnv:
    def __init__(self, 
                #  home=[-0.14, -0.02, -0.05, -1.57, 0.0, 1.50, -0.91]): # base
                #  home=[0.4, -0.02, -0.05, -1.57, 0.0, 1.50, -0.91]): # trans
                home=[-0.45, -0.2, -0.05, -1.70, 0.0, 1.50, -0.91]): # apple
        # init ros node
        rospy.init_node('franka_env', anonymous=True)

        # init tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_publisher = tf2_ros.TransformBroadcaster()

        # init realsense sub
        self.obs = {
            'object': np.empty(0),
            'agentview_image': np.empty(0),
            'robot0_eef_pos': np.empty(0),
            'robot0_eef_quat': np.empty(0),
            'robot0_gripper_qpos': np.empty(0),
        }
        self.color_subscriber = rospy.Subscriber("/camera/color/image_raw", Imgs, self.set_color_callback)
        self.rgb_resize_shape = (120,160)

        # init robot & gripper
        self.robot = RobotInterface(
            ip_address="10.0.0.2",
            port="50051",
        )
        self.gripper = GripperInterface(
            ip_address="10.0.0.2",
            port="50052",
        )
        self.home = torch.Tensor(home)
        self.controller_on = False

        # init transforms
        self.t_w_fb, self.R_w_fb = self.get_pose_from_tf('world', 'franka_base')

    def reset(self, reset_pose=np.empty(0)):
        self.robot.set_home_pose(self.home)
        self.robot.go_home()
        if reset_pose.any():
            reset_pose = torch.tensor(reset_pose).float()
            self.robot.move_to_ee_pose(position=reset_pose[:3], orientation=reset_pose[3:7], time_to_go=2.0)
            print('Custom Reset Pose')
        self.gripper.goto(speed=0.5, force=0.1, width=0.09)
        self.gripper_state = False
        if self.controller_on:
            self.robot.terminate_current_policy()
        self.robot.start_cartesian_impedance()
        self.controller_on = True
        
        self.publish_data()
        self.update_obs()
        return self.obs
        
    def step(self,action):
        action = torch.tensor(action).float()
        self.robot.update_desired_ee_pose(position=action[:3], orientation=action[3:7])
        if action[-1] < 0 and self.gripper_state: 
            self.gripper.goto(speed=0.5, force=0.1, width=0.09)
            self.gripper_state = False
            print('release')
        elif action[-1] > 0 and not self.gripper_state: 
            self.gripper.grasp(speed=1.0, force=28, grasp_width=0.04, epsilon_inner=0.2, epsilon_outer=0.2)
            self.gripper_state = True
            print('grab')

        
        self.publish_data()
        self.update_obs()
        return self.obs
    
    def update_obs(self):
        # updates everything but image
        while not self.tf_buffer.can_transform('true_franka_base', 'object', rospy.Time(0), rospy.Duration(1e-3)) and not self.obs['agentview_image'].any() :
            continue
        t_obj, R_obj = self.get_pose_from_tf('true_franka_base', 'object')

        ee_pos, ee_quat = self.robot.get_ee_pose()
        if not self.gripper_state:
            gripper_qpos = 0.0808534
        else:
            gripper_qpos = self.gripper.get_state().width


        self.obs['object'] = np.concatenate((t_obj, R_obj.as_quat(), np.zeros(7)))
        self.obs['robot0_eef_pos'] = np.array(ee_pos)
        self.obs['robot0_eef_quat'] = np.array(ee_quat)
        self.obs['robot0_gripper_qpos'] = np.array([gripper_qpos,-gripper_qpos])

    def set_color_callback(self, msg):
        color = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.rgb_resize_shape[1],self.rgb_resize_shape[0]))
        self.obs['agentview_image'] = rgb
        
    def publish_pose(self, pose_base, pose_name, pose):
        msg = TransformStamped()
        
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = pose_base
        msg.child_frame_id = pose_name
        
        msg.transform.translation.x = pose[0]
        msg.transform.translation.y = pose[1]
        msg.transform.translation.z = pose[2]

        msg.transform.rotation.x = pose[3]
        msg.transform.rotation.y = pose[4]
        msg.transform.rotation.z = pose[5]
        msg.transform.rotation.w = pose[6]

        self.tf_publisher.sendTransform(msg)
    
    def fb_to_world_transform(self, pos_fb, rot_fb):
        fb_to_world = np.eye(4)
        fb_to_world[:3,3] = np.array(self.t_w_fb)
        fb_to_world[:3,:3] = self.R_w_fb.as_matrix()

        pose_fb = np.eye(4)
        pose_fb[:3,3] = np.array(pos_fb)
        pose_fb[:3,:3] = rot_fb.as_matrix()

        pose_world = fb_to_world @ pose_fb
        pos_world = pose_world[:3,3]
        rot_world = R.from_matrix(pose_world[:3,:3])

        return pos_world, rot_world

    def get_pose_from_tf(self, frame0, frame1):
        transform = self.tf_buffer.lookup_transform(frame0, frame1, rospy.Time(0), rospy.Duration(1e-3))

        # update time
        self.cur_time = transform.header.stamp
        
        pos = torch.Tensor([transform.transform.translation.x, 
                            transform.transform.translation.y, 
                            transform.transform.translation.z])
        rot = R.from_quat([transform.transform.rotation.x, 
                           transform.transform.rotation.y, 
                           transform.transform.rotation.z,
                           transform.transform.rotation.w])
        if frame0 == 'world' and frame1 == 'franka_base':
            rot_offset = R.from_euler('z', 180, degrees=True)
            ### Marker Center to Tag Center Offset ###
            # x: -10mm
            # y: 0mm
            # z: -38mm
            ### Tag Center to Frank Base Offset ###
            # x: 116mm
            # y: 0mm
            # z: -31mm
            pos_offset = (torch.Tensor([116, 0, -31]) + torch.Tensor([-10, 0, -38])) * 1e-3
            rot = rot_offset * rot
            pos = pos_offset + pos
        return pos, rot
    
    def publish_data(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        ee_pos_world, ee_rot_world = self.fb_to_world_transform(ee_pos, R.from_quat(ee_quat))
        self.publish_pose('world', 'ee', pose=np.concatenate((ee_pos_world, ee_rot_world.as_quat())))
        self.publish_pose('world', 'true_franka_base', pose=np.concatenate((self.t_w_fb, self.R_w_fb.as_quat())))


