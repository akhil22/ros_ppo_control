#!/usr/bin/env python
import rospy
from ppo_control import PPOControl
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
#from ilqr.msg import ins
import tensorflow as tf
import utm
import matplotlib.pyplot as plt
import numpy as np
import math
import rospkg
def zero_to_2pi(theta):
    if theta < 0:
        theta = 2*math.pi + theta
    elif theta > 2*math.pi:
        theta = theta - 2*math.pi
    return theta
def simulate_warthog(X, v, w, dt):
    xcurr = X[0] + v*math.cos(X[2])*dt
    ycurr = X[1] + v*math.sin(X[2])*dt
    thcurr = zero_to_2pi(X[2]+w*dt)
    #print(v,w)
    return [xcurr, ycurr, thcurr]
class HuskyPPONode:
    def __init__(self):
        self.warthog_ppo = PPOControl()
        vel_topic = rospy.get_param('~vel_topic', 'warthog_velocity_controller/cmd_vel')
        twist_odom_topic = rospy.get_param('~odom_topic', 'warthog_velocity_controller/odom')
        pose_odom_topic = rospy.get_param('~odom_topic2', 'odometry/filtered2')
        ins_topic = rospy.get_param('~ins_topic', 'vectronav/fix')
        path_topic = rospy.get_param('~path_topic', '/local_planning/path/final_trajectory')
        #frozen_graph_path = rospy.get_param('~frozen_graph_path', "/home/sai/hdd1/ml-master/ml-agents/config/ppo/results/wlong_path54/3DBall/frozen_graph_def.pb")
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('ros_ppo_control')
        frozen_graph_path = rospy.get_param('~frozen_graph_path', pkg_path + "/policies/wlong_path54/3DBall/frozen_graph_def.pb")
        #waypoint_file_path = rospy.get_param('~waypoint_file_path', "unity_waypoints_bkp.txt")
        waypoint_file_path = rospy.get_param('~waypoint_file_path', pkg_path + "/scripts/waypoints.txt")
        self.warthog_ppo.read_tf_frozen_graph(frozen_graph_path)
        #self.warthog_ppo.read_waypoint_file(waypoint_file_path)
        self.twist_pub = rospy.Publisher(vel_topic, Twist, queue_size = 10)
        rospy.Subscriber(twist_odom_topic, Odometry, self.twist_odom_cb)
        rospy.Subscriber(pose_odom_topic, Odometry, self.pose_odom_cb)
        rospy.Subscriber(path_topic, Path, self.path_cb, queue_size = 1)
        self.got_twist = False
        self.got_odom = False
        self.got_path = False
        #rospy.Subscriber(ins_topic, ins, self.ins_cb)
        show_path = False
        if show_path:
            self.cx = [i[0] for i in self.warthog_ppo.waypoints_list]
            self.cy = [i[1] for i in self.warthog_ppo.waypoints_list]
            print(self.warthog_ppo.waypoints_list)
            plt.plot(self.cx, self.cy, '+b')
            plt.show()
    def twist_odom_cb(self, data):
        #return
        v = data.twist.twist.linear.x
        w = data.twist.twist.angular.z
        self.warthog_ppo.set_twist([v, w])
        if not self.got_twist:
            self.got_twist = True
    def path_cb(self, path):
        x_list = []
        y_list = []
        th_list = []
        v_list = []
        for pose in path.poses:
            x_list.append(pose.pose.position.x)
            y_list.append(pose.pose.position.y)
            th_list.append(pose.pose.position.x)
            v_list.append(pose.pose.position.z)
        self.warthog_ppo.set_waypoints_from_list(x_list, y_list, th_list, v_list)
        if not self.got_path:
            self.got_path = True
    def pose_odom_cb(self, data):
        #return
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        th = data.pose.covariance[1]
        self.warthog_ppo.set_pose([x, y, th])
        if not self.got_odom:
            self.got_odom = True
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
    do_sim = False
    #warthog_ppo_node.got_path = True
    if do_sim:
        r = rospy.Rate(1)
        while(not rospy.is_shutdown() and (not warthog_ppo_node.got_path)):
            rospy.logwarn("not getting path will try again")
            r.sleep()
        start_idx = 10
        xinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][0]
        yinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][1]
        thinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][2]
        warthog_ppo_node.warthog_ppo.set_pose([xinit, yinit, thinit])
       # warthog_ppo_node.warthog_ppo.set_pose([132.180, -78.957, 160*math.pi/180.])
        #warthog_ppo_node.warthog_ppo.set_pose([7.54831069e+05, 3.39048552e+06, 5.53962977e+00])
        #warthog_ppo_node.warthog_ppo.set_pose([0. , 0., 5.53962977e+00])
        warthog_ppo_node.warthog_ppo.set_twist([0., 0.])
        x_pose = []
        y_pose = []
        for i in range(0, 300):
            tstart = rospy.get_rostime()
            obs = warthog_ppo_node.warthog_ppo.get_observation()
            twist = warthog_ppo_node.warthog_ppo.get_control(np.array(obs).reshape(1,42))
            v = np.clip(twist[0][0], 0, 1) * 2.0
            w = np.clip(twist[0][1], -1, 1) * 2.5
            current_pose = simulate_warthog(warthog_ppo_node.warthog_ppo.get_pose(), v, w, 0.05)
            warthog_ppo_node.warthog_ppo.set_pose(current_pose)
            warthog_ppo_node.warthog_ppo.set_twist([v, w])
            x_pose.append(current_pose[0])
            y_pose.append(current_pose[1])
            #print(v,w)
            delta = (rospy.get_rostime() - tstart).to_sec()
            logstring = "getting v= " + str(v) +" getting w= "+ str(w) + " time delta = " + str(delta)
            rospy.loginfo(logstring)
        warthog_ppo_node.warthog_ppo.path_lock.acquire()
        cx = [i[0] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        cy = [i[1] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        warthog_ppo_node.warthog_ppo.path_lock.release()
        plt.plot(cx, cy, '+b')
        plt.plot(x_pose, y_pose, '+g')
        plt.show()
    else:
        while not rospy.is_shutdown():
            tstart = rospy.get_rostime()
            if not (warthog_ppo_node.got_odom and warthog_ppo_node.got_path and warthog_ppo_node.got_twist):
                if not warthog_ppo_node.got_odom:
                    rospy.logwarn("Not Receiving Odometry")
                if not warthog_ppo_node.got_twist:
                    rospy.logwarn("Not Receiving Twist")
                if not warthog_ppo_node.got_odom:
                    rospy.logwarn("Not Receiving Path")
                continue
            obs = warthog_ppo_node.warthog_ppo.get_observation()
            twist = warthog_ppo_node.warthog_ppo.get_control(np.array(obs).reshape(1,42))
            v = np.clip(twist[0][0], 0, 1) * 2.0
            closest_idx = warthog_ppo_node.warthog_ppo.closest_idx
            v = warthog_ppo_node.warthog_ppo.waypoints_list[closest_idx][3]
            w = np.clip(twist[0][1], -1, 1) * 2.5
            twist_msg = Twist()
            twist_msg.linear.x =v
            twist_msg.angular.z =w
            warthog_ppo_node.twist_pub.publish(twist_msg)
            delta = (rospy.get_rostime() - tstart).to_sec()
            logstring = "getting v= " + str(v) +" getting w= "+ str(w) + " time delta = " + str(delta)
            rate.sleep()
if __name__=='__main__':
    main()
