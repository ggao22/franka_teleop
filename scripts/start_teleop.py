#!/home/george/miniforge3/envs/polymetis-local/bin/python
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

import torch
import numpy as np
from polymetis import RobotInterface
from polymetis import GripperInterface

import pygame
from scipy.spatial.transform import Rotation as R


class FrankaTeleop:
    def __init__(self):
        # init ros node
        rospy.init_node('franka_teleop', anonymous=True)

        # init tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pose_msgs = {'true_franka_base': TransformStamped(),
                          'ee': TransformStamped()}
        self.tf_publisher = tf2_ros.TransformBroadcaster()

        # init robot & gripper
        self.robot = RobotInterface(
            ip_address="10.0.0.2",
            port="50051",
        )
        self.gripper = GripperInterface(
            ip_address="10.0.0.2",
            port="50052",
        )
        self.new_home = torch.Tensor([-0.14, -0.02, -0.05, -1.57, 0.05, 1.50, -0.91])
        self.robot.set_home_pose(self.new_home)
        self.robot.go_home()
        self.gripper_state = False
        self.rate = rospy.Rate(1000.0)  # 1000Hz

        # init transforms
        self.t_w_fb, self.R_w_fb = self.get_pose_from_tf('world', 'franka_base')
        self.t_fb_ee_0, quat_fb_ee_0 = self.robot.get_ee_pose()
        self.R_fb_ee_0 = R.from_quat(list(quat_fb_ee_0))
        self.t_w_m_0, self.R_w_m_0 = self.get_pose_from_tf('world', 'franka_teleop')
        self.R_ee_m_0 = self.world_to_ee_rot_transform(self.R_w_m_0)

        # init pygame
        pygame.init()
        screen = pygame.display.set_mode((640,480))

    
    def publish_pose(self, pose_name, pose):
        msg = self.pose_msgs[pose_name] 
        
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
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
            rot_offset = R.from_euler('z', 180, degrees=True)
            ### Marker Center to Tag Center Offset ###
            # x: -8mm
            # y: 0mm
            # z: -36mm
            ### Tag Center to Frank Base Offset ###
            # x: 116mm
            # y: 0mm
            # z: -38mm
            pos_offset = (torch.Tensor([116, 0, -38]) + torch.Tensor([-8, 0, -36])) * 1e-3
            rot = rot_offset * rot
            pos = pos_offset + pos
        return pos, rot

    
    def start_teleop(self):
        self.robot.start_cartesian_impedance()
        while not rospy.is_shutdown():
            try:
                t_w_m_k, R_w_m_k = self.get_pose_from_tf('world', 'franka_teleop')
                
                delta_pos = torch.Tensor(self.R_w_fb.inv().apply(t_w_m_k - self.t_w_m_0)) + self.t_fb_ee_0
                R_ee_m_k = self.world_to_ee_rot_transform(R_w_m_k)
                delta_rot = self.R_fb_ee_0 * R_ee_m_k * self.R_ee_m_0.inv()
                delta_quat = torch.Tensor(delta_rot.as_quat())

                self.robot.update_desired_ee_pose(position=delta_pos, orientation=delta_quat)

                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE] and not self.gripper_state: 
                    # press "SPACE" to close gripper
                    self.gripper.grasp(speed=1.0, force=10, grasp_width=0.04, epsilon_inner=0.2, epsilon_outer=0.2)
                    self.gripper_state = True
                elif keys[pygame.K_RALT] and self.gripper_state:
                    # press "RIGHT ALT" to OPEN gripper
                    self.gripper.goto(speed=0.5, force=0.1, width=0.08)
                    self.gripper_state = False
                if keys[pygame.K_k]:
                    # press "K" to exit
                    exit(0)

                delta_pos_world, delta_rot_world =  self.fb_to_world_transform(delta_pos, delta_rot)
                self.publish_pose('ee', pose=np.concatenate((delta_pos_world, delta_rot_world.as_quat())))
                self.publish_pose('true_franka_base', pose=np.concatenate((self.t_w_fb, self.R_w_fb.as_quat())))

                t_tfb_obj, R_tfb_obj = self.get_pose_from_tf('true_franka_base', 'object')

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform not available")
                
            self.rate.sleep()

        

if __name__ == '__main__':
    try:
        ft = FrankaTeleop()
        ft.start_teleop()
    except rospy.ROSInterruptException:
        pass
