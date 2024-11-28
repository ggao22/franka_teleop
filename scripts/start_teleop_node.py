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

import pygame
from scipy.spatial.transform import Rotation as R

import click
import h5py


class FrankaTeleop:
    def __init__(self, 
                 home=[-0.45, -0.2, -0.05, -1.70, 0.0, 1.50, -0.91],
                 rate=1000.0,
                 demo_output_file=None):
        # init ros node
        rospy.init_node('franka_teleop', anonymous=True)

        # init tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_publisher = tf2_ros.TransformBroadcaster()

        # init realsense sub
        self.rgb = None
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
        self.new_home = torch.Tensor(home)
        self.robot.set_home_pose(self.new_home)
        self.gripper_state = False
        self.rate = rospy.Rate(rate)  # Hz
        self.controller_on = False

        # init transforms
        self.init_transforms()

        # init pygame
        pygame.init()
        screen = pygame.display.set_mode((480,480))

        # init demo recording assets
        if demo_output_file:
            self.demo_output_file = demo_output_file
            self.obs = {
                'object': [],
                'agentview_image': [],
                'robot0_eef_pos': [],
                'robot0_eef_quat': [],
                'robot0_gripper_qpos': [],
            }
            self.actions = []
            self.demo_idx = 0
            self.is_recording = False
            with h5py.File(demo_output_file, 'a') as f:
                if "data" not in f:
                    f.create_group("data")
                else:
                    if len(f['data']):
                        self.demo_idx = max([int(s.split("demo_")[1]) for s in f['data'].keys()]) + 1


    def init_transforms(self):
        if self.controller_on:
            self.robot.terminate_current_policy()
        self.robot.go_home()
        self.t_w_fb, self.R_w_fb = self.get_pose_from_tf('world', 'franka_base')
        self.t_fb_ee_0, quat_fb_ee_0 = self.robot.get_ee_pose()
        self.R_fb_ee_0 = R.from_quat(list(quat_fb_ee_0))
        self.t_w_m_0, self.R_w_m_0 = self.get_pose_from_tf('world', 'franka_teleop')
        self.R_ee_m_0 = self.world_to_ee_rot_transform(self.R_w_m_0)
        self.robot.start_cartesian_impedance()
        self.controller_on = True

    def set_color_callback(self, msg):
        color = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        self.rgb = cv2.resize(rgb, (self.rgb_resize_shape[1],self.rgb_resize_shape[0]))

    def key_callback(self, data):
        self.key = data.data
        
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
    

    def world_to_ee_rot_transform(self, rot):
        return self.R_fb_ee_0.inv() * self.R_w_fb.inv() * rot
    

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
            # rot_offset = R.from_euler('z', 180, degrees=True)
            rot_offset = R.from_euler('z', 0, degrees=True) 

            ### Marker Center to Tag Center Offset ###
            # x: +16mm
            # y: 0mm
            # z: -18mm
            ### Tag Center to Frank Base Offset ###
            # x: -120mm
            # y: 0mm
            # z: -30mm
            pos_offset = (torch.Tensor([-120, 0, -30]) + torch.Tensor([16, 10, -18])) * 1e-3
            rot = rot_offset * rot
            pos = pos_offset + pos
        return pos, rot

    
    def teleop_control(self):
        t_w_m_k, R_w_m_k = self.get_pose_from_tf('world', 'franka_teleop')
                
        self.delta_pos = torch.Tensor(self.R_w_fb.inv().apply(t_w_m_k - self.t_w_m_0)) + self.t_fb_ee_0
        R_ee_m_k = self.world_to_ee_rot_transform(R_w_m_k)
        self.delta_rot = self.R_fb_ee_0 * R_ee_m_k * self.R_ee_m_0.inv()
        self.delta_quat = torch.Tensor(self.delta_rot.as_quat())

        self.robot.update_desired_ee_pose(position=self.delta_pos, orientation=self.delta_quat)

        pygame.event.pump()
        self.keys = pygame.key.get_pressed()
        if self.keys[pygame.K_SPACE] and not self.gripper_state: 
            # press "SPACE" to close gripper
            self.gripper.grasp(speed=1.0, force=20, grasp_width=0.04, epsilon_inner=0.2, epsilon_outer=0.2)
            self.gripper_state = True
        elif self.keys[pygame.K_RALT] and self.gripper_state:
            # press "RIGHT ALT" to OPEN gripper
            self.gripper.goto(speed=0.5, force=0.1, width=0.08)
            self.gripper_state = False
        
    
    def publish_data(self):
        delta_pos_world, delta_rot_world =  self.fb_to_world_transform(self.delta_pos, self.delta_rot)
        self.publish_pose('world', 'ee', pose=np.concatenate((delta_pos_world, delta_rot_world.as_quat())))
        self.publish_pose('world', 'true_franka_base', pose=np.concatenate((self.t_w_fb, self.R_w_fb.as_quat())))


    def start_teleop(self):
        while not rospy.is_shutdown():
            try:
                self.teleop_control()
                self.publish_data()
                a,b = self.get_pose_from_tf('true_franka_base', 'object')
                c,d = self.delta_pos, self.delta_rot
                print(np.array(a)-np.array(c))
                print(a)

                if self.keys[pygame.K_k]:
                    # press "K" to exit
                    exit(0)

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform not available")
                
            self.rate.sleep()


    def save_demonstration(self):
        with h5py.File(self.demo_output_file, 'a') as f:
            data_group = f['data']
            demo_group = data_group.create_group(f"demo_{self.demo_idx}")
            demo_group.create_dataset("actions", data=np.array(self.actions), compression="gzip")

            obs_group = demo_group.create_group("obs")
            for key, array in self.obs.items():
                obs_group.create_dataset(key, data=np.array(array), compression="gzip")
        
        print(f"Saved demo {self.demo_idx}")
        self.demo_idx += 1
        [self.obs[key].clear() for key in self.obs.keys()]
        self.actions.clear()

    
    def start_demo_recorder(self):

        print('Demo Recorder Ready!')
        
        while not rospy.is_shutdown():
            try:
                self.teleop_control()
                self.publish_data()

                if self.keys[pygame.K_k]:
                    # press "K" to exit
                    exit(0)
                elif self.keys[pygame.K_b] and not self.is_recording:
                    self.init_transforms()
                    self.is_recording = True
                    print('Recording!')
                elif self.keys[pygame.K_s] and self.is_recording:
                    self.is_recording = False
                    print('Stopped.')
                    self.save_demonstration()
                elif self.keys[pygame.K_BACKSPACE] and self.is_recording:
                    # Delete the most recent recorded episode
                    if click.confirm('Are you sure to drop an episode?'):
                        [self.obs[key].clear() for key in self.obs.keys()]
                        self.actions.clear()
                        self.is_recording = False
                        print("Deleted")
                
                if self.is_recording:
                    ee_pos, ee_quat = self.robot.get_ee_pose()
                    t_obj, R_obj = self.get_pose_from_tf('true_franka_base', 'object')
                    gripper_qpos = self.gripper.get_state().width
                    
                    self.actions.append(np.concatenate((self.delta_pos, self.delta_rot.as_rotvec(), [1.0 if self.gripper_state else -1.0])))
                    self.obs['object'].append(np.concatenate((t_obj, R_obj.as_quat(), np.zeros(7))))
                    self.obs['agentview_image'].append(self.rgb)
                    self.obs['robot0_eef_pos'].append(np.array(ee_pos))
                    self.obs['robot0_eef_quat'].append(np.array(ee_quat))
                    self.obs['robot0_gripper_qpos'].append([gripper_qpos,-gripper_qpos])

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform not available")
                
            self.rate.sleep()

        

if __name__ == '__main__':
    try:
        ft = FrankaTeleop()
        ft.start_teleop()
    except rospy.ROSInterruptException:
        pass
