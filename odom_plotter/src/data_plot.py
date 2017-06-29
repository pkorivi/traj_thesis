#!/usr/bin/env python
import random
import time
import roslib
import rospy
import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
import math

x = np.array([])
y = np.array([])
z = np.array([]) #sequence number
speed = open("speed.txt",'w')
yaw_val   = open("yaw.txt",'w')
cordi   = open("cordi.txt",'w')
count =0
init = False
initial_yaw = 0.0
vx = 0.0
vy = 0.0
delta_head = 0.0
th = 0.0
x = 0.0
y = 0.0
def speed_callback(data):
    global vx
    vx = -(data.linear.x / (9.54929659643*5.5))*(0.031)
    #speed.write(str(data.linear.x))
    """
    speed.write(str(vx))
    speed.write("\n")
    """
def yaw_callback(data):
    global count,init,initial_yaw,delta_head,th
    if init == False:
        initial_yaw = (data.data*3.14)/180.0
        init = True
        #print(initial_yaw)
    else:
        count = count+1
        delta_head= (data.data*3.14)/180.0 - initial_yaw;
        if delta_head > 3.14:
            delta_head = delta_head - 6.28
        elif delta_head < -3.14:
            delta_head = delta_head + 6.28
        th = delta_head
        """
        yaw_val.write(str(count))
        yaw_val.write(",")
        yaw_val.write(str(delta_head))
        yaw_val.write("\n")
        """

def main(args):
    global x,y
    rospy.init_node('data_plotter', anonymous = False)
    image_sub0 = rospy.Subscriber("/model_car/yaw",Float32, yaw_callback)
    image_sub1 = rospy.Subscriber("/motor_control/twist",Twist, speed_callback)
    #image_sub = rospy.Subscriber("/model_car/yaw",Float32, yaw_callback)
    #ani = animation.FuncAnimation(fig, animate, interval=200000)
    #plt.show()
    current_time = rospy.get_rostime()
    last_time = rospy.get_rostime()
    r = rospy.Rate(1000) # 10hz
    while not rospy.is_shutdown():
        #rospy.spin()
        current_time = rospy.get_rostime()
        dt = (current_time - last_time).secs;
        last_time = current_time
        delta_x = (vx * math.cos(th) - vy * math.sin(th)) * dt;
        delta_y = (vx * math.sin(th) + vy * math.cos(th)) * dt;
        x += delta_x;
        y += delta_y;
        cordi.write(str(current_time.secs))
        cordi.write(",")
        cordi.write(str(dt))
        cordi.write(",")
        cordi.write(str(th))
        cordi.write("\n")
        r.sleep()

    #plt.show()


if __name__ == '__main__':
    main(sys.argv)
