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

x = np.array([])
y = np.array([])
z = np.array([]) #sequence number
#file2write=open("plot_data.txt",'w')


fig = plt.figure()
#ax1 = fig.gca(projection='3d')
#ax1 = fig.add_subplot(1,1,1,projection='3d')
ax1 = fig.add_subplot(111, projection='3d')


def animate(args):
    #pullData = open("sampleText.txt","r").read()
    #dataArray = pullData.split('\n')
    ax1.clear()
    #ax1.plot(x,y,zs = z)
    ax1.plot(x,y,z, '-b')

def odom_callback(data):
    global x,y,z
    x=np.concatenate((x,[data.pose.pose.position.x]))
    y=np.concatenate((y,[data.pose.pose.position.y]))
    z=np.concatenate((z,[data.header.seq]))
    """
    x.append(data.pose.pose.position.x)
    y.append(data.pose.pose.position.y)
    z.append(int(data.header.seq))
    """
    """
    file2write.write(str(data.pose.pose.position.x))
    file2write.write(",")
    file2write.write(str(data.pose.pose.position.y))
    file2write.write("\n")
    """

def main(args):
    rospy.init_node('odom_plotter', anonymous = False)
    image_sub = rospy.Subscriber("/odom",Odometry, odom_callback)
    ani = animation.FuncAnimation(fig, animate, interval=200000)
    plt.show()
    rospy.spin()
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
