#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

rospy.init_node("cmd_remap")
cmd_pub = rospy.Publisher("/Warthog/cmd_vel", Twist, queue_size=100)
def cmd_cb(data):
    global cmd_pub
    cmd_pub.publish(data)
rospy.Subscriber("/warthog_velocity_controller/cmd_vel", Twist, cmd_cb)
rospy.spin()
