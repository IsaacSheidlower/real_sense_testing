#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
import pyrealsense2 as rs2
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
import tf 
from hlpr_manipulation_utils.arm_moveit2 import ArmMoveIt

depth_image_topic = "/camera/depth/image_rect_raw"
depth_info_topic = "/camera/depth/camera_info"
world_frame = "base_link"
depth_frame = "camera_depth_frame"
camera_frame = "/camera_color_optical_frame"
arm_frame = "/j2s7s300_link_base"
eef_frame = "/j2s7s300_ee_link"


def depthInfoCB(camera_intrinsics, camerainfo):
        if camera_intrinsics:
            pass
        else:
            camera_intrinsics = rs2.intrinsics()
            camera_intrinsics.width = camerainfo.width
            camera_intrinsics.height = camerainfo.height
            camera_intrinsics.ppx = camerainfo.K[2]
            camera_intrinsics.ppy = camerainfo.K[5]
            camera_intrinsics.fx = camerainfo.K[0]
            camera_intrinsics.fy = camerainfo.K[4]
            if (camerainfo.distortion_model == "plumb_bob"):
                camera_intrinsics.model = rs2.distortion.brown_conrady
            elif (camerainfo.distortion_model == "equidistant"):
                camera_intrinsics.model = rs2.distortion.kannala_brandt4
            camera_intrinsics.coeffs = [i for i in camerainfo.D]
            return camera_intrinsics


rospy.init_node("cam_testing")
camera_intrinsics = None
camera_info = rospy.wait_for_message(depth_info_topic, CameraInfo)
camera_intrinsics = depthInfoCB(camera_intrinsics, camera_info)

depth_image = rospy.wait_for_message(depth_image_topic, Image)
bridge = CvBridge()
depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
# print(depth_image)
# depth_point = [depth_image[0][1], depth_image[0][0]]
# print(depth_point)

img = rospy.wait_for_message('/camera/color/image_raw', Image)
img2 = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
bridge = CvBridge()
cv_image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
img2 = bridge.imgmsg_to_cv2(img2, desired_encoding='passthrough')

img = cv_image
print("camera:" , img.shape)
print("depth: ", img2.shape)
def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 210, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(thresh, 0, 0)
    img_dilate = cv2.dilate(img_canny, None, iterations=7)
    return cv2.erode(img_dilate, None, iterations=7)

def get_contours(img):
    contours = cv2.findContours(process(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cv2.imwrite('img_new.jpg', img[y:y+h,x:x+w])

    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    y = y+15
    x = x+15
    w = w-40
    h = h-40
    new_img = img[y:y+h,x:x+w]
    return new_img, x, y, w, h


rect_img, x_shift,y_shift,w_shift,h_shift = get_contours(img)


original_img = img.copy()
# cv2.imshow('img', original_img[x_shift:y_shift+h_shift,x_shift:x_shift+w_shift])
# cv2.imshow("img_processed", rect_img)
# cv2.waitKey(0)

img = rect_img
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.size)
# cv2.imshow("img_processed", rect_img)
# cv2.waitKey(0)


inputImage = img.copy()
#inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
print(inputImage.shape)
ksize = (1, 1)
  
# Using cv2.blur() method 
inputImage = cv2.blur(inputImage, ksize) 
edges = cv2.Canny(inputImage,150,200,apertureSize = 3)
print(edges.shape)
minLineLength = 30
maxLineGap = 5
lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
for o in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[o]:
        #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
        pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
        cv2.polylines(inputImage, [pts], True, (0,255,0))

print(inputImage.shape)
font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(inputImage,"Tracks Detected", (500, 250), font, 0.5, 255)
# cv2.imshow("Trolley_Problem_Result", inputImage)
# cv2.imshow('edge', edges)

# cv2.waitKey(0)
# print(edges)

kernel = np.ones((2,2),np.uint8)
dilated_img = cv2.dilate(edges, kernel, iterations = 2)

canvas = dilated_img.copy() # Canvas for plotting contours on
print(canvas.size)
canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB) # create 3 channel image so we can plot contours in color

contours = cv2.findContours(dilated_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[1]

# loop through the contours and check through their hierarchy, if they are inner contours
# more here: https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
# for i,cont in enumerate(contours):
#     # look for hierarchy[i][3]!=-1, ie hole boundaries
#     if ( hierarchy[0][i][3] != -1 ):
#         #cv2.drawContours(canvas, cont, -1, (0, 180, 0), 1) # plot inner contours GREEN
#         cv2.fillPoly(canvas, pts =[cont], color=(0, 180, 0)) # fill inner contours GREEN
#         print(cont)
#     else:
#         cv2.drawContours(canvas, cont, -1, (255, 0, 0), 1) # plot all others BLUE, for completeness

#print(contours)
dist2 = cv2.pointPolygonTest(contours[0], (204, 38), True)
#print(dist2)
# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)

# img = edges.copy()
canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
# print(canvas.shape, edges.shape)

skeleton = medial_axis(canvas).astype(np.uint8)
x_cor,y_cor = np.where(skeleton>0)
# print(x_cor,y_cor)
# print(skeleton)
# cv2.imshow("result", skeleton*255)
# cv2.waitKey(0)

color_img = cv2.cvtColor(skeleton*255,cv2.COLOR_GRAY2RGB)


def merge(list1, list2):
    merged_list = [[list1[i], list2[i]] for i in range(0, len(list1))]
    return merged_list

def closest_point(arr, x, y):
    dist = (arr[:, 0] - x)**2 + (arr[:, 1] - y)**2
    return arr[dist.argmin()]

def sort_by_nearest(list1):
    list1 = list1.copy()
    start_point = list1[0]
    sorted_list = []
    sorted_list.insert(0, list(start_point))
    #print(start_point)
    #print(np.where(np.all(list1 == start_point, axis=1)))
    list1 = np.delete(list1, np.where(list1 == start_point), axis=0)
    while list1.size > 0:
        nearest_point = closest_point(list1, start_point[0], start_point[1])
        sorted_list.append(list(nearest_point))
        list1 = np.delete(list1, np.where(np.all(list1 == start_point, axis=1)), axis=0)
        start_point = nearest_point

    # remove duplicates
    c_set = set(tuple(t) for t in sorted_list)
    c = [ list(t) for t in c_set ]
    c.sort(key = lambda t: sorted_list.index(t))
    return sorted_list


# def sort_by_nearest(list1):
#     list1 = list1.copy()
#     start_point = list1[0]
#     sorted_list = []
#     sorted_list.insert(0, list(start_point))
#     # print(start_point)
#     # print(np.argwhere(list1 == start_point))
#     list1.remove(start_point)
#     while list1 != []:
#         nearest_point = closest_point(np.array(list1), start_point[0], start_point[1])
#         sorted_list.append(nearest_point)
#         start_point = nearest_point
#         #list1 = np.delete(list1, np.where(list1 == start_point), axis=0)
#         print(start_point)
#         list1.remove(start_point)

#     return sorted_list


cordinates = merge(x_cor, y_cor)
#print(cordinates)
cordinates = sorted(cordinates, key=lambda k: [k[1], k[0]])
cordinates = np.array(cordinates)


cordinates = sort_by_nearest(cordinates)
cordinates = np.array(cordinates)
# print(original_img.shape, color_img.shape)
cordinates[:,0] += y_shift

cordinates[:,1] += x_shift

mean, stdev = np.mean(cordinates, axis=0), np.std(cordinates, axis=0)
# Mean: [811.2  76.4  88. ]
# Std: [152.97764543   6.85857128  29.04479299]
## Find Outliers
outliers = ((np.abs(cordinates[:,0] - mean[0]) < 2*stdev[0])
            * (np.abs(cordinates[:,1] - mean[1]) < 2*stdev[1]))

cordinates = cordinates[outliers]
# is this th depth in meters? It seems like yes
# print(cordinates[5])
# print("depth_sample", img2[265,162])
# print("cor", cordinates[50][0], cordinates[50][1])

# for corr in cordinates:
#     print(img2[corr[0], corr[1]]/1000.0)
#     if img2[corr[0], corr[1]]/1000 == 0 or img2[corr[0], corr[1]]/1000 ==1:
#         continue
#     else:
#         print(img2[corr[0], corr[1]]/1000)
# from IPython import embed
# embed()
# # print(bbox_coords)
# print(cordinates[:,0])

# for cor in cordinates:
#     #canvas = cv2.circle(color_img, (cor[1], cor[0]), 1, (0,0,255), -1)
#     canvas = cv2.circle(original_img, (cor[1], cor[0]), 1, (0,0,255), -1)
#     cv2.imshow('canvas', canvas)
#     cv2.waitKey(5)

depth_arr = np.array(img2, dtype=np.float32)
tf_listener = tf.TransformListener()
tf_listener.waitForTransform(arm_frame, camera_frame, rospy.Time(), rospy.Duration(4.0))
for cor in cordinates:
    p = rs2.rs2_deproject_pixel_to_point(camera_intrinsics, [cor[0], cor[1]], depth_arr[cor[1],cor[0]])
    p = [entry / 1000.0 for entry in p]
    poi = PoseStamped()
    poi.pose.position.x = p[0]
    poi.pose.position.y = p[1]
    poi.pose.position.z = p[2]
    poi.pose.orientation = Quaternion(0.,0.,0.,1.)
    poi.header.frame_id = camera_frame


    goal_world = tf_listener.transformPose(arm_frame, poi)
    print(goal_world)
#p = rs2.rs2_deproject_pixel_to_point(camera_intrinsics, [cordinates[10][0], cordinates[10][1]], img2[cordinates[10][1],cordinates[10][0]])


#p = Point(cordinates[50][0]/1000.0, cordinates[50][1]/1000.0, img2[cordinates[50][1],cordinates[50][0]]/1000.0)

p = [entry / 1000.0 for entry in p]
print("POINT, ", p)

poi = PoseStamped()
poi.pose.position.x = p[0]
poi.pose.position.y = p[1]
poi.pose.position.z = p[2]
poi.pose.orientation = Quaternion(0.,0.,0.,1.)
poi.header.frame_id = camera_frame

tf_listener = tf.TransformListener()
"""
    x: 0.650393009186
    y: 0.314593672752
    z: 0.42180532217
    w: 0.547813892365

    x: 0.224111557007
    y: -0.544560670853
    z: 0.0815567523241
  orientation: 
    x: 0.536402761936
    y: 0.483809828758
    z: 0.25174665451
    w: 0.644068241119

  position: 
    x: 0.210650160909
    y: -0.265275985003
    z: 0.503405988216
  orientation: 
    x: 0.649025440216
    y: 0.315696060658
    z: 0.423214763403
    w: 0.547714471817
"""
# q = Quaternion()
# q.x = 0.53640#rot[0]
# q.y = 0.4838 #rot[1]
# q.z = 0.2517#rot[2]
# q.w = 0.6440#rot[3]
# # q = Quaternion()
# # q.x = 0.#rot[0]
# # q.y = 0.#rot[1]
# # q.z = 0. #rot[2]
# # q.w = 1.0#rot[3]

# print(poi)
tf_listener.waitForTransform(arm_frame, camera_frame, rospy.Time(), rospy.Duration(4.0))
goal_world = tf_listener.transformPose(arm_frame, poi)
# the x-y is correct but there is still a z problem


        # plan = self.moveit.plan_ee_pos(goal_arm)
        # self.moveit.move_through_waypoints(plan)
#goal_world.pose.orientation = q

# goal_world.pose.position.x=0.217083305719
# goal_world.pose.position.y=-0.747633460875
# goal_world.pose.position.z=0.408275365829


# goal_world.pose.orientation.x=0.474602401257
# goal_world.pose.orientation.y=0.599907696247
# goal_world.pose.orientation.z=-0.337281763554
# goal_world.pose.orientation.w=0.548729717731

print(goal_world)
#tf_listener.waitForTransform(arm_frame, eef_frame, rospy.Time(), rospy.Duration(4.0))
#goal_world = tf_listener.transformPose(eef_frame, goal_world)

#goal_world.pose.position.z+=.2
# print(goal_world)

# arm = ArmMoveIt()
# print(tf_listener.transformPose(eef_frame, goal_world))

#arm.move_to_ee_pose(goal_world.pose)
#print(arm.get_IK(goal_world.pose))