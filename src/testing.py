#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis


rospy.init_node("cam_testing")
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


rect_img, x,y,w,h = get_contours(img)
# cv2.imshow("img_processed", rect_img)
# cv2.waitKey(0)

img = rect_img
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.size)
# #converted = convert_hls(img)
# image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([350,55,100])
# black_mask = cv2.inRange(image, lower_black, upper_black)
# # yellow color mask
# lower = np.uint8([0, 200, 0])
# upper = np.uint8([255, 255, 255])
# white_mask = cv2.inRange(image, lower, upper)
# # combine the mask
# mask = cv2.bitwise_or(white_mask, black_mask)
# result = img.copy()
# # cv2.imshow("mask",mask) 
# # cv2.waitKey(0)

# height,width = mask.shape
# skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# temp_nonzero = np.count_nonzero(mask)
# while(np.count_nonzero(mask) != 0 ):
#     eroded = cv2.erode(mask,kernel)
#     cv2.imshow("eroded",eroded)   
#     temp = cv2.dilate(eroded,kernel)
#     cv2.imshow("dilate",temp)
#     temp = cv2.subtract(mask,temp)
#     skel = cv2.bitwise_or(skel,temp)
#     mask = eroded.copy()
 
# # cv2.imshow("skel",skel)
# # cv2.waitKey(0)

# edges = cv2.Canny(skel, 50, 150)
# cv2.imshow("edges",edges)
# lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=30,maxLineGap=30)
# i = 0
# for x1,y1,x2,y2 in lines[0]:
#     i+=1
#     cv2.line(result,(x1,y1),(x2,y2),(255,0,0),1)
#     print(i)

# cv2.imshow("res",result)
# cv2.waitKey(0)

"""try 2"""
# img = rect_img
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# kernel_size = 5
# blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
# low_threshold = 50
# high_threshold = 150
# edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# rho = 1  # distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # angular resolution in radians of the Hough grid
# threshold = 15  # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 50  # minimum number of pixels making up a line
# max_line_gap = 20  # maximum gap in pixels between connectable line segments
# line_image = np.copy(img) * 0  # creating a blank to draw lines on

# # Run Hough on edge detected image
# # Output "lines" is an array containing endpoints of detected line segments
# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                     min_line_length, max_line_gap)

# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

# cv2.imshow("lines",lines_edges)
# cv2.waitKey(0)

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

print(contours)
dist2 = cv2.pointPolygonTest(contours[0], (204, 38), True)
print(dist2)
# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)

# img = edges.copy()
canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
print(canvas.shape, edges.shape)

skeleton = medial_axis(canvas).astype(np.uint8)
x_cor,y_cor = np.where(skeleton>0)
print(x_cor,y_cor)
print(skeleton)
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
print(cordinates)
for cor in cordinates:
    canvas = cv2.circle(color_img, (cor[1], cor[0]), 1, (0,0,255), -1)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(50)
