#!/usr/bin/env python2
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


def nothing(x):
    pass

cv2.namedWindow("camera", 0)
cv2.resizeWindow("camera", 640,480)
cv2.moveWindow("camera", 400,30)
cv2.waitKey(1)
#print()'Create trackbars')
cv2.createTrackbar('HLow','camera',0,255,nothing)
cv2.createTrackbar('SLow','camera',0,255,nothing)
cv2.createTrackbar('VLow','camera',0,255,nothing)

cv2.createTrackbar('HHigh','camera',0,255,nothing)
cv2.createTrackbar('SHigh','camera',0,255,nothing)
cv2.createTrackbar('VHigh','camera',0,255,nothing)

bridge = CvBridge()

#img  = cv2.imread("vx.jpg")
#cv2.imshow('vieww', img)
#cv2.waitKey(0)
#cv_image = Image.new('RGB', (640, 480))

def callback(data):
    global cv_image
    bridge = CvBridge()
    #Use the below code for Turtlebot compressed image
    #np_arr = np.fromstring(data.data, np.uint8)
    #cv_image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    #If subscribing to Drone use the below line
    print 'callback'
    np_arr = np.fromstring(data.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    #"""
    HLow = cv2.getTrackbarPos('HLow','camera')
    SLow = cv2.getTrackbarPos('SLow','camera')
    VLow = cv2.getTrackbarPos('VLow','camera')
    HHigh = cv2.getTrackbarPos('HHigh','camera')
    SHigh = cv2.getTrackbarPos('SHigh','camera')
    VHigh = cv2.getTrackbarPos('VHigh','camera')
    img = cv_image.copy()
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #convert img to HSV and store result in imgHSVyellow
    lower = np.array([HLow, SLow, VLow]) #np arrays for upper and lower thresholds
    upper = np.array([HHigh, SHigh, VHigh])
    imgthreshed = cv2.inRange(imgHSV, lower, upper) #threshold imgHSV
    plt.imshow(imgthreshed)
    cv2.waitKey(1)
    #imgthreshed = cv2.blur(imgthreshed,(3,3))
    #print imgthreshed.shape
    #cv2.imshow("camera", imgthreshed)
    #cv2.imshow("View.jpg", cv_image)
    #cv2.waitKey(1)
    """
    print 'show start'
    print 'wait start'
    print 'wait done'

    """
    #cv2.imwrite("View.jpg", cv_image)



def main(args):
    rospy.init_node('HSV_Picker', anonymous = False)
    image_sub = rospy.Subscriber("/app/camera/rgb/image_raw/compressed",CompressedImage, callback)
    rate = rospy.Rate(10)

    #print cv_image.shape
    #global cv_image
    #cv2.imshow("View.jpg", cv_image)
    while not rospy.is_shutdown():
        #print 'show start'
        #cv2.imshow("View.jpg", cv_image)
        #print 'show end'
        #cv2.waitKey(1)
        rospy.spin()
        rate.sleep()
    """
    rospy.spin()
    #print cv_image.shape
    print 'show start'
    cv2.imshow("View.jpg", cv_image)
    print 'show end'
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
    """

if __name__ == '__main__':
    main(sys.argv)
