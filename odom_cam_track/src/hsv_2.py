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
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#video = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))
#bridge = CvBridge()

class hsv_picker:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/app/camera/rgb/image_raw/compressed",CompressedImage, self.callback)
        self.track_pub = rospy.Publisher('/track_img', Image,queue_size=1)
        self.bridge = CvBridge()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))
        #self.video = cv2.VideoWriter("tester.avi", cv2.CV_FOURCC('M','J','P','G'), 10.0,(1280, 720),True)
	#Callback function for subscribed image
    def callback(self,data):
        global video
        #print 'wait start'
        np_arr = np.fromstring(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        HLow = cv2.getTrackbarPos('HLow','camera')
        SLow = cv2.getTrackbarPos('SLow','camera')
        VLow = cv2.getTrackbarPos('VLow','camera')
        HHigh = cv2.getTrackbarPos('HHigh','camera')
        SHigh = cv2.getTrackbarPos('SHigh','camera')
        VHigh = cv2.getTrackbarPos('VHigh','camera')
        """ #Red
        HLow = 0#cv2.getTrackbarPos('HLow','camera')
        SLow = 100#cv2.getTrackbarPos('SLow','camera')
        VLow = 100#cv2.getTrackbarPos('VLow','camera')
        HHigh = 16#cv2.getTrackbarPos('HHigh','camera')
        SHigh = 255#cv2.getTrackbarPos('SHigh','camera')
        VHigh = 255#cv2.getTrackbarPos('VHigh','camera')
        """
        #Green
        HLow = 60#cv2.getTrackbarPos('HLow','camera')
        SLow = 60#cv2.getTrackbarPos('SLow','camera')
        VLow = 46#cv2.getTrackbarPos('VLow','camera')
        HHigh = 97#cv2.getTrackbarPos('HHigh','camera')
        SHigh = 255#cv2.getTrackbarPos('SHigh','camera')
        VHigh = 255#cv2.getTrackbarPos('VHigh','camera')
        #img = cv_image.copy()
        imgHSV = cv2.cvtColor(cv_image,cv2.COLOR_BGR2HSV) #convert img to HSV and store result in imgHSVyellow
        lower = np.array([HLow, SLow, VLow]) #np arrays for upper and lower thresholds
        upper = np.array([HHigh, SHigh, VHigh])
        imgthreshed = cv2.inRange(imgHSV, lower, upper) #threshold imgHSV
        anded = np.bitwise_and(gray, imgthreshed)
        anded_t = np.bitwise_and(anded, 255)
        #print 'nw_img'
        #print imgthreshed

        image_message = self.bridge.cv2_to_imgmsg(anded, "mono8")
        self.track_pub.publish(image_message)
        #self.video.write(cv_image)
        #plt.imshow(imgthreshed)
        """
        print 'wait start'
        print 'wait done'
        """

#Main function for the node
def main(args):
    rospy.init_node('hsv_picker', anonymous = False)
    ic = hsv_picker()
    #"""
    try:
        #plt.show(block=True)
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down hsv_picker Node')
        cv2.destroyAllWindows()
        video.release()
    #"""

if __name__ == '__main__':
	main(sys.argv)
