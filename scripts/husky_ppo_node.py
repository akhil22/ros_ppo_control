#!/usr/bin/env python
import rospy
from ppo_control import PPOControl
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
#from ilqr.msg import ins
import tensorflow as tf
import utm
import matplotlib.pyplot as plt
import numpy as np
import math
def simulate_warthog(X, v, w, dt):
    xcurr = X[0] + v*math.cos(X[2])*dt
    ycurr = X[1] + v*math.sin(X[2])*dt
    thcurr = X[2]+w*dt
    print(v,w)
    return [xcurr, ycurr, thcurr]
class HuskyPPONode:
    def __init__(self):
        self.warthog_ppo = PPOControl()
        vel_topic = rospy.get_param('~vel_topic', 'warthog_velocity_controller/cmd_vel')
        odom_topic = rospy.get_param('~odom_topic', 'warthog_velocity_controller/odom')
        ins_topic = rospy.get_param('~ins_topic', 'vectronav/fix')
        frozen_graph_path = rospy.get_param('~frozen_graph_path', "/home/sai/hdd1/ml-master/ml-agents/config/ppo/results/wlong_path40/3DBall/frozen_graph_def.pb")
        waypoint_file_path = rospy.get_param('~waypoint_file_path', "ww.txt")
        self.warthog_ppo.read_tf_frozen_graph(frozen_graph_path)
        self.warthog_ppo.read_waypoint_file(waypoint_file_path)
        self.twist_pub = rospy.Publisher(vel_topic, Twist, queue_size = 10)
        rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        #rospy.Subscriber(ins_topic, ins, self.ins_cb)
        self.cx = [i[0] for i in self.warthog_ppo.waypoints_list]
        self.cy = [i[1] for i in self.warthog_ppo.waypoints_list]
        print(self.warthog_ppo.waypoints_list)
        #plt.plot(self.cx, self.cy, '+b')
        plt.show()
    def odom_cb(self, data):
        v = data.twist.twist.linear.x
        w = data.twist.twist.angular.z
        self.warthog_ppo.set_twist([v, w])
    def ins_cb(self, data):
        lat = data.LLA.x
        lon = data.LLA.y
        utm_cord = utm.from_latlon(lat, lon)
        self.warthog_ppo.set_pose([utm_cord[0], utm_cord[1]])

def main():
    rospy.init_node('warthog_ppo_node')
    warthog_ppo_node = HuskyPPONode()
    rate = rospy.Rate(20)
    do_sim = rospy.get_param("~do_sim", True)
    if do_sim:
        warthog_ppo_node.warthog_ppo.set_pose([132.180, -78.957, 160*math.pi/180.])
        #warthog_ppo_node.warthog_ppo.set_pose([7.54831069e+05, 3.39048552e+06, 5.53962977e+00])
        #warthog_ppo_node.warthog_ppo.set_pose([0. , 0., 5.53962977e+00])
        warthog_ppo_node.warthog_ppo.set_twist([0., 0.])
        x_pose = []
        y_pose = []
        for i in range(0, 2000):
            obs = warthog_ppo_node.warthog_ppo.get_observation()
            twist = warthog_ppo_node.warthog_ppo.get_control(np.array(obs).reshape(1,42))
            v = np.clip(twist[0][0], 0, 1) * 4.0
            w = np.clip(twist[0][1], -1, 1) * 2.5
            current_pose = simulate_warthog(warthog_ppo_node.warthog_ppo.get_pose(), v, w, 0.05)
            warthog_ppo_node.warthog_ppo.set_pose(current_pose)
            warthog_ppo_node.warthog_ppo.set_twist([v, w])
            x_pose.append(current_pose[0])
            y_pose.append(current_pose[1])
        plt.plot(warthog_ppo_node.cx, warthog_ppo_node.cy, '+b')
        plt.plot(x_pose, y_pose, '+g')
        plt.show()
    else:
        while not rospy.is_shutdown():
            obs = warthog_ppo_node.warthog_ppo.get_observation()
            twist_cmd = warthog_ppo_node.warthog_ppo.get_control(np.array(obs).reshape(1,42))
            twist_msg = Twist()
            twist_msg.linear.x = twist_cmd[0]
            twist_msg.angular.z = twist_cmd[1]
            warthog_ppo_node.twist_pub.publish(twist_msg)
            rate.sleep()
if __name__=='__main__':
    main()
