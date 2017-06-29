#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
from geometry_msgs.msg import PointStamped
import tf
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt


def get_red_thresh_img(p_img):

    # Red_Thresholds HSV

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    return cv2.inRange(p_img, lower_red1, upper_red1)


def get_green_thresh_img(p_img):
    lower_green = np.array([60, 60, 46])
    upper_green = np.array([97, 255, 255])

    # """
    # in BGR
    # lower_green = np.array([70,108,128])
    # upper_green = np.array([163,255,255])
    # """ #in HSV
    # Threshold the HSV image to get only single color portions

    return cv2.inRange(p_img, lower_green, upper_green)


def get_contours(thresh_img):
    (_, contours, hierarchy) = cv2.findContours(thresh_img,
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_squares(contours):
    coordi = []

    # coordi = np.array([],dtype=object)
    # coordi = np.ndarray([],dtype=object)

    for x in range(len(contours)):
        contourarea = cv2.contourArea(contours[x])  # get area of contour
        if contourarea > 100:  # Discard contours with a small area as this may just be noise
            # The below 2 functions help you to approximate the contour to a nearest polygon
            arclength = cv2.arcLength(contours[x], True)
            approxcontour = cv2.approxPolyDP(contours[x], 0.1* arclength, True)
            # Check for Square
            if len(approxcontour) == 4:
                # Find the coordinates of the polygon with respect to he camera frame in pixels
                rect_cordi = cv2.minAreaRect(contours[x])
                obj_x = int(rect_cordi[0][0])
                obj_y = int(rect_cordi[0][1])
                # print(approxcontour)
                coordi.append((obj_x, obj_y))
            else:
                continue
    return coordi


def hsv(img):

    # return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def estimate_car_coordinates(g_sq, r_sq):

    # implement some mechanism to reduce error and estimate position of the car such that it starts at (0,0) and plots similar to

    return (0, 0)


def process_image(image):
    global initial_cordi_set
    global initial_x_y
    # Processes image and publishes the image with coordinates
    hsv_img = hsv(image)
    g_thresh_img = get_green_thresh_img(hsv_img)
    g_contours = get_contours(g_thresh_img)
    g_squares = get_squares(g_contours)  # coordinates of square

    r_thresh_img = get_red_thresh_img(hsv_img)
    r_contours = get_contours(r_thresh_img)
    r_squares = get_squares(r_contours)  # coordinates of square

    car_cordi = estimate_car_coordinates(g_squares, r_squares)
    if (len(g_squares) != 0) and (initial_cordi_set == True):
        #TODO - These coordinates are transformed to match with origin and orientation of odom coordinates
        file2write.write(str(image_time))
        file2write.write(",")
        #TODO Added '- sign to match with the Version 2 car'

        file2write.write(str((initial_x_y[0] - g_squares[0][0])/one_m_pixel ))
        file2write.write(",")
        file2write.write(str((initial_x_y[1] - g_squares[0][1])/one_m_pixel ))
        file2write.write("\n")
    elif (len(g_squares) != 0) and initial_cordi_set == False:
        initial_cordi_set = True
        initial_x_y[0] = g_squares[0][0]
        initial_x_y[1] = g_squares[0][1]



    #,g_squares[1]
    """
    print g_squares
    print 'g_squares :: '
    print 'r_squares :: '
    print r_squares
    """
    for pt in r_squares:
        cv2.circle(image, pt, 15, (0, 0, 255), -1)
    for pt in g_squares:
        cv2.circle(image, pt, 15, (0, 255, 0), -1)
    return image


class cam_tracker:
    def __init__(self):
        self.image_sub = \
            rospy.Subscriber('/app/camera/rgb/image_raw/compressed',
                             CompressedImage, self.callback)
        self.track_pub = rospy.Publisher('/track_img', Image,
                queue_size=1)
        self.bridge = CvBridge()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter('color_vid_out.avi', fourcc, 20.0, (1280,
                720))

        # self.video = cv2.VideoWriter("tester.avi", cv2.CV_FOURCC('M','J','P','G'), 10.0,(1280, 720),True)
    # Callback function for subscribed image

    def callback(self, data):
        global video
        global image_time
        #TODO this time gap is because of linux time sync. Check the error
        image_time = data.header.stamp.secs + 3470
        np_arr = np.fromstring(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        np_arr = np.fromstring(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        final_img = process_image(cv_image)
        #self.video.write(final_img)
        #image_message = self.bridge.cv2_to_imgmsg(final_img, encoding="passthrough")
        #self.track_pub.publish(image_message)


# Main function for the node
file2write=open("vid_cordi.txt",'w')
initial_cordi_set = False
initial_x_y = [0,0]
one_m_pixel = 279.0

image_time =0
def main(args):
    rospy.init_node('cam_tracker', anonymous=False)
    ic = cam_tracker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print 'Shutting Down hsv_picker Node'
        cv2.destroyAllWindows()
        video.release()


if __name__ == '__main__':
    main(sys.argv)
