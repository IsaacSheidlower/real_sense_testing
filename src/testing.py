#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
import pyrealsense2 as rs2
from pyrealsense2 import pipeline_profile as Profile

depth_image_topic = "/camera/depth/image_rect_raw"
depth_info_topic = "/camera/depth/camera_info"

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

profile = Profile()
print(profile.get_device())
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
depth_min = 0.11 #meter
depth_max = 1.0 #meter

# rs2.rs2_project_color_pixel_to_depth_pixel([])
depth_image = rospy.wait_for_message(depth_image_topic, Image)
bridge = CvBridge()
depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
print(depth_image.shape)
depth_point = [depth_image[0][1], depth_image[0][0]]
print(depth_point)

img = rospy.wait_for_message('/camera/color/image_raw', Image)
bridge = CvBridge()
cv_image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')

img = cv_image

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 163, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(thresh, 0, 0)
    img_dilate = cv2.dilate(img_canny, None, iterations=7)
    return cv2.erode(img_dilate, None, iterations=7)

def get_contours(img):
    contours, _ = cv2.findContours(process(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cv2.imwrite('img_new.jpg', img[y:y+h,x:x+w])

    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    y = y+20
    x = x+20
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
print(img.size)


inputImage = img.copy()
#inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
print(inputImage.shape)
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

kernel = np.ones((3,3),np.uint8)
dilated_img = cv2.dilate(edges, kernel, iterations = 2)

canvas = dilated_img.copy() # Canvas for plotting contours on
print(canvas.size)
canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB) # create 3 channel image so we can plot contours in color

contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

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
#cordinates[:,0] -= w_shift
cordinates[:,1] += x_shift
#cordinates[:,1] -= h_shift

# old_width = original_img.shape[1]
# old_height = original_img.shape[0]
# new_width = color_img.shape[1]
# new_height = color_img.shape[0]
# scale_x = old_width/new_width
# scale_y = old_height/new_height

# print('scale:', scale_x, scale_y)

# cordinates = [[int(x/scale_x), int(y/scale_y)] for x,y in cordinates]
# # print(bbox_coords)
# # print(cordinates[:,0])
# for cor in cordinates:
#     #canvas = cv2.circle(color_img, (cor[1], cor[0]), 1, (0,0,255), -1)
#     canvas = cv2.circle(original_img, (cor[1], cor[0]), 1, (0,0,255), -1)
#     cv2.imshow('canvas', canvas)
#     cv2.waitKey(50)
