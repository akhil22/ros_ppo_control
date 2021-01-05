#!/usr/bin/env python
import rospy
from ppo_control_tf import PPOControl
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
#from ilqr.msg import ins
#import tensorflow as tf 
from pyquaternion import Quaternion as qut
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
'''def zero_to_2pi(theta):
    if theta > math.pi:
        theta = theta - 2*math.pi
    elif theta > -math.pi:
        theta = theta + 2*math.pi
    return theta'''
def simulate_warthog(X, v, w, dt):
    xcurr = X[0] + v*math.cos(X[2])*dt
    ycurr = X[1] + v*math.sin(X[2])*dt
    thcurr = zero_to_2pi(X[2]+w*dt)
    #thcurr = X[2]+w*dt
    #print(v,w)
    return [xcurr, ycurr, thcurr]
class HuskyPPONode:
    def __init__(self):
        self.warthog_ppo = PPOControl()
        vel_topic = rospy.get_param('~vel_topic', 'warthog_velocity_controller/cmd_vel')
        twist_odom_topic = rospy.get_param('~odom_topic', '/warthog_velocity_controller/odom')
        pose_odom_topic = rospy.get_param('~odom_topic2', '/odometry/filtered')
        ins_topic = rospy.get_param('~ins_topic', 'vectronav/fix')
        path_topic = rospy.get_param('~path_topic', '/local_planning/path/final_trajectory')
        #frozen_graph_path = rospy.get_param('~frozen_graph_path', "/home/sai/hdd1/ml-master/ml-agents/config/ppo/results/wlong_path54/3DBall/frozen_graph_def.pb")
        rospack = rospkg.RosPack()
        #pkg_path = rospack.get_path('ros_ppo_control')
        pkg_path = "./"
        frozen_graph_path = rospy.get_param('~frozen_graph_path', pkg_path + "/policies/wlong_path54/3DBall/frozen_graph_def.pb")
        #waypoint_file_path = rospy.get_param('~waypoint_file_path', "unity_waypoints_bkp.txt")
        waypoint_file_path = rospy.get_param('~waypoint_file_path', pkg_path + "/scripts/waypoints.txt")
        #self.warthog_ppo.read_tf_frozen_graph(frozen_graph_path)
        #self.warthog_ppo.read_ppo_policy('/home/sai/warthog_rl_alien/policy/vel_weight4_d9')
        self.warthog_ppo.read_ppo_policy('/home/sai/warthog_rl_alien/policy/vel_weight7_stable9')
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
        self.got_twist = True
        return
        #if self.got_twist:
            #return
        v = data.twist.twist.linear.x
        rospy.logwarn("getting twist")
        w = data.twist.twist.angular.z
        self.warthog_ppo.set_twist([v, w])
        if not self.got_twist:
            self.got_twist = True
    def path_cb(self, path):
        if self.got_path:
            return
        x_list = []
        y_list = []
        th_list = []
        v_list = []
        i = 0
        for pose in path.poses:
            x_list.append(pose.pose.position.x)
            y_list.append(pose.pose.position.y)
            th_list.append(pose.pose.position.x)
            #v_list.append(pose.pose.position.z)
            v_list.append(2.5)
            i = i+1
        x_list = x_list[:-1]
        y_list = y_list[:-1]
        th_list = th_list[:-1]
        v_list = v_list[:-1]
        for j in range(0,30):
            x_list.append(x_list[i-2 + j] + x_list[i-3] - x_list[i-4])
            y_list.append(y_list[i-2 + j] + y_list[i-3] - y_list[i-4])
            th_list.append(0)
            v_list.append(2.5)
        self.warthog_ppo.set_waypoints_from_list(x_list, y_list, th_list, v_list)
        if not self.got_path:
            self.got_path = True
    def pose_odom_cb(self, data):
        self.got_odom = True
        return
        #if self.got_odom:
            #return
        x = data.pose.pose.position.x 
        y = data.pose.pose.position.y
        temp_y = data.pose.pose.orientation.z
        temp_x = data.pose.pose.orientation.w
        quat = (temp_x, 0, 0, temp_y)
        myqut = qut(quat)
        th = myqut.radians
        th = 2*math.atan2(temp_y, temp_x)*180/math.pi
        #th = data.pose.covariance[1]
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
    #do_sim = True
    do_sim = False
    #warthog_ppo_node.got_path = True
    if do_sim:
        r = rospy.Rate(1)
        while(not rospy.is_shutdown() and (not warthog_ppo_node.got_path)):
            rospy.logwarn("not getting path will try again")
            r.sleep()
        start_idx = 0
        xinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][0] + 0.1
        yinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][1] + 0.1
        thinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][2]
        warthog_ppo_node.warthog_ppo.set_pose([xinit, yinit, thinit])
       # warthog_ppo_node.warthog_ppo.set_pose([132.180, -78.957, 160*math.pi/180.])
        #warthog_ppo_node.warthog_ppo.set_pose([7.54831069e+05, 3.39048552e+06, 5.53962977e+00])
        #warthog_ppo_node.warthog_ppo.set_pose([0. , 0., 5.53962977e+00])
        warthog_ppo_node.warthog_ppo.set_twist([0., 0.])
        x_pose = []
        y_pose = []
        for i in range(0, 100):
            tstart = rospy.get_rostime()
            obs = warthog_ppo_node.warthog_ppo.get_observation()
            #twist = warthog_ppo_node.warthog_ppo.get_control(np.array(obs).reshape(1,42))
            twist = warthog_ppo_node.warthog_ppo.get_ppo_control(np.array(obs).reshape(1,42))[0]
            #v = np.clip(twist[0][0], 0, 1) * 2.0
            #w = np.clip(twist[0][1], -1, 1) * 2.5
            v = np.clip(twist[0][0], 0, 1)*4
            w = np.clip(twist[0][1], -1, 1)*2.5
            current_pose = simulate_warthog(warthog_ppo_node.warthog_ppo.get_pose(), v, w, 0.05)
            warthog_ppo_node.warthog_ppo.set_pose(current_pose)
            warthog_ppo_node.warthog_ppo.set_twist([v, w])
            x_pose.append(current_pose[0])
            y_pose.append(current_pose[1])
            #print(v,w)
            print(current_pose)
            delta = (rospy.get_rostime() - tstart).to_sec()
            logstring = "getting v= " + str(v) +" getting w= "+ str(w) + " time delta = " + str(delta)
            rospy.loginfo(logstring)
        warthog_ppo_node.warthog_ppo.path_lock.acquire()
        cx = [i[0] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        cy = [i[1] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        warthog_ppo_node.warthog_ppo.path_lock.release()
        f1 = plt.figure(1)
        plt.plot(cx, cy, '+b')
        plt.plot(x_pose, y_pose, '+g')
        fig1 = plt.figure(2)
        plt.plot(v_rec)
        fig2 = plt.figure(3)
        plt.plot(w_rec)
        plt.show()
    else:
        x_pose = []
        y_pose = []
        start_idx = -1
        #while not rospy.is_shutdown():
        counter = 0
        v_rec = []
        w_rec = []
        while 1:
            tstart = rospy.get_rostime()
            if not (warthog_ppo_node.got_odom and warthog_ppo_node.got_path and warthog_ppo_node.got_twist):
                if not warthog_ppo_node.got_odom:
                    rospy.logwarn("Not Receiving Odometry")
                if not warthog_ppo_node.got_twist:
                    rospy.logwarn("Not Receiving Twist")
                if not warthog_ppo_node.got_path:
                    rospy.logwarn("Not Receiving Path")
                continue
            if start_idx == -1:
                sim_run = True
                if sim_run == True:
                    start_idx = 1
                    xinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][0] + 0.05
                    yinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][1] + 0.05
                    thinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][2]
                    warthog_ppo_node.warthog_ppo.set_pose([xinit, yinit, thinit])
                    warthog_ppo_node.warthog_ppo.set_twist([0.,0.])
            obs = warthog_ppo_node.warthog_ppo.get_observation()
            twist = warthog_ppo_node.warthog_ppo.get_ppo_control(np.array(obs).reshape(1,42))
            #v = np.clip(twist[0][0], 0, 1) * 2.0
            closest_idx = warthog_ppo_node.warthog_ppo.closest_idx
            #v = warthog_ppo_node.warthog_ppo.waypoints_list[closest_idx][3]
            #w = np.clip(twist[0][1], -1, 1) * 2.5
            twist = twist[0]
            v = np.clip(twist[0][0], 0, 1)*4.0
            w = np.clip(twist[0][1], -1, 1)*2.5
            v_rec.append(v)
            w_rec.append(w)
            if sim_run:
                current_pose = simulate_warthog(warthog_ppo_node.warthog_ppo.get_pose(), v, w, 0.05)
                warthog_ppo_node.warthog_ppo.set_pose(current_pose)
                warthog_ppo_node.warthog_ppo.set_twist([v, w])
            twist_msg = Twist()
            twist_msg.linear.x =v
            twist_msg.angular.z =w
            warthog_ppo_node.twist_pub.publish(twist_msg)
            delta = (rospy.get_rostime() - tstart).to_sec()
            logstring = "getting v= " + str(v) +" getting w= "+ str(w) + " time delta = " + str(delta) + "theta = " + str(0.1)
            rospy.logwarn(logstring)
            if sim_run:
                x_pose.append(current_pose[0])
                y_pose.append(current_pose[1])
            if counter == 300:
                break
            counter = counter + 1
        warthog_ppo_node.warthog_ppo.path_lock.acquire()
        cx = [i[0] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        cy = [i[1] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        warthog_ppo_node.warthog_ppo.path_lock.release()
        f1 = plt.figure(1)
        plt.plot(cx, cy, '+b')
        plt.plot(x_pose, y_pose, '+g')
        fig1 = plt.figure(2)
        plt.plot(v_rec)
        fig2 = plt.figure(3)
        plt.plot(w_rec)
        plt.show()
        #plt.plot(cx, cy, '+b')
        #plt.plot(x_pose, y_pose, '+g')
        #plt.show()
if __name__=='__main__':
    main()
