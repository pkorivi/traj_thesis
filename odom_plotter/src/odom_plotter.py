#!/usr/bin/env python
import random
import time
import roslib
import rospy
import sys
from matplotlib import pyplot as plt
from matplotlib import animation
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16
from std_msgs.msg import Float32

x = []
y = []
file2write=open("plot_data.txt",'w')

#fig = plt.figure()
#plt.axis([-2,2,-2,2])
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def animate(args):
    #pullData = open("sampleText.txt","r").read()
    #dataArray = pullData.split('\n')
    ax1.clear()
    ax1.plot(x,y)

def odom_callback(data):
    x.append(data.pose.pose.position.x)
    y.append(data.pose.pose.position.y)
    """
    file2write.write(str(data.pose.pose.position.x))
    file2write.write(",")
    file2write.write(str(data.pose.pose.position.y))
    file2write.write("\n")
    """
    #print(data.pose.pose.position.x,data.pose.pose.position.y)

#def animate(args):
#    return plt.plot(x, y, color='g')


#anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000)
#plt.show()

#ani = animation.FuncAnimation(fig, animate, interval=1000)
#plt.show()

def main(args):
    rospy.init_node('odom_plotter', anonymous = False)
    image_sub = rospy.Subscriber("/odom",Odometry, odom_callback)
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
    rospy.spin()
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
