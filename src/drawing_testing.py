#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import time
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
import tf 
from hlpr_manipulation_utils.arm_moveit2 import ArmMoveIt
from hlpr_manipulation_utils.manipulator import Gripper
from rl_movement_utils import go_to_relative, cartesian_velocity_req

"""
Start pose:
[1.2279195296113203, 1.7713559258535105, 3.2761400792344495, 1.2205710692154648, 2.9774074408058633, 4.849935391732884, 1.3943493877200237]

"""
rospy.init_node("draw_test")

start_pose = [1.2279195296113203, 1.7713559258535105, 3.2761400792344495, 1.2205710692154648, 2.9774074408058633, 4.849935391732884, 1.3943493877200237]
arm = ArmMoveIt("j2s7s300_link_base")
gripper = Gripper()
arm.move_to_joint_pose(start_pose)

# # # gripper.open()
# # # time.sleep(5)
# # # gripper.close(block=False)

duration = 1.4
# go_down = [0.0,0.0,-.1,0.0,0.0,0.0]
# go_up = [0.0,0.0,.3,0.0,0.0,0.0]
go_left = [0.1,0.0,0.0,0.0,0.0,0.0]
# go_right = [-0.1,0.0,0.0,0.0,0.0,0.0]

# # cartesian_velocity_req(go_up, duration)
# # cartesian_velocity_req(go_right, duration)
# # cartesian_velocity_req(go_down, duration)
# cartesian_velocity_req(go_left, duration)
# # for i in range(5):
# #     cartesian_velocity_req(go_right, duration)