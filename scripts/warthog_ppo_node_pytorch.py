#!/usr/bin/env python
from tkinter import W
import rospy
import torch
from ppo_control_pytorch import PPOControl
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
import sys
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
        #pose_odom_topic = rospy.get_param('~odom_topic2', '/odometry/filtered')
        #pose_odom_topic = rospy.get_param('~odom_topic2', '/aft_mapped_to_init_high_frec')
        pose_odom_topic = rospy.get_param('~odom_topic2', '/aft_mapped_to_init_high_frec')
        #pose_odom_topic = rospy.get_param('~odom_topic2', '/odometry/filtered2')
        path_topic = rospy.get_param('~path_topic', '/local_planning/path/final_trajectory')
        #path_topic = rospy.get_param('~path_topic', '/local_planning/path/optimized_a_star_path')
        #frozen_graph_path = rospy.get_param('~frozen_graph_path', "/home/sai/hdd1/ml-master/ml-agents/config/ppo/results/wlong_path54/3DBall/frozen_graph_def.pb")
        rospack = rospkg.RosPack()
        #pkg_path = rospack.get_path('ros_ppo_control')
        pkg_path = "./"
        #waypoint_file_path = rospy.get_param('~waypoint_file_path', "unity_waypoints_bkp.txt")
        waypoint_file_path = rospy.get_param('~waypoint_file_path', pkg_path + "/scripts/waypoints2.txt")
        #self.warthog_ppo.read_tf_frozen_graph(frozen_graph_path)
        #self.warthog_ppo.read_ppo_policy('/home/administrator/warthog_rl_alien/policy/vel_weight9_stable_delayed9')
        #self.warthog_ppo.read_ppo_policy('/home/administrator/Downloads/warthog_rl/policy/vel_weight9_stable_delayed9')
        #self.warthog_ppo.read_ppo_policy('/home/administrator/Downloads/warthog_rl/policy/vel_weight8_stable9')
        #self.warthog_ppo.read_ppo_policy('/home/administrator/Downloads/warthog/warthog_rl/policy/combine_trained')
        self.warthog_ppo.read_ppo_policy('/home/administrator/Downloads/warthog/warthog_rl/policy/combine_trained_may8_300')
        #self.warthog_ppo.read_ppo_policy('/home/administrator/warthog_rl_alien/policy/vel_weight9_stable9')
        #self.warthog_ppo.read_ppo_policy('./model2')
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
        #if self.got_twist:
            #return
        v = data.twist.twist.linear.x
        rospy.logwarn("getting twist")
        w = data.twist.twist.angular.z
        self.warthog_ppo.set_twist([v, w])
        if not self.got_twist:
            self.got_twist = True
    def path_cb(self, path):
        #if self.got_path:
            #return
        x_list = []
        y_list = []
        th_list = []
        v_list = []
        i = 0
        for pose in path.poses:
            x_list.append(pose.pose.position.x)
            y_list.append(pose.pose.position.y)
            th_list.append(pose.pose.position.x)
            v_list.append(pose.pose.position.z)
            #v_list.append(2.5)
            i = i+1
        x_list = x_list[:-1]
        y_list = y_list[:-1]
        th_list = th_list[:-1]
        v_list = v_list[:-1]
        '''for j in range(0,30):
            x_list.append(x_list[i-2 + j] + x_list[i-3] - x_list[i-4])
            y_list.append(y_list[i-2 + j] + y_list[i-3] - y_list[i-4])
            th_list.append(0)
            v_list.append(0)'''
        self.warthog_ppo.set_waypoints_from_list(x_list, y_list, th_list, v_list)
        if not self.got_path:
            self.got_path = True
    def pose_odom_cb(self, data):
        #self.got_odom = True
        #return
        #if self.got_odom:
            #return
        x = data.pose.pose.position.x 
        y = data.pose.pose.position.y
        temp_y = data.pose.pose.orientation.z
        temp_x = data.pose.pose.orientation.w
        print("z: ", temp_y)
        print("w: ", temp_x)
        quat = (temp_x, 0, 0, temp_y)
        myqut = qut(quat)
        print("without sign: ",myqut.radians*180/math.pi)
        th = myqut.radians*np.sign(temp_y)
        print("with sign: ",th*180/math.pi)
        #th = 2*math.atan2(temp_y, temp_x)*180/math.pi
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
    traj_file_name = sys.argv[1]
    traj_file = open(traj_file_name, 'w')
    traj_file.writelines("x,y,th,vel,w,v_cmd,w_cmd\n")
    warthog_ppo_node = HuskyPPONode()
    rate = rospy.Rate(30)
    do_sim = rospy.get_param("~do_sim", True)
    #do_sim = True
    do_sim = False
    poses = []
    max_num_poses = 200
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
        #cx = [i[0] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        #cy = [i[1] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        warthog_ppo_node.warthog_ppo.path_lock.release()
        #plt.plot(cx, cy, '+b')
        plt.plot(x_pose, y_pose, '+g')
        plt.show()
    else:
        x_pose = []
        y_pose = []
        start_idx = -1
        v_rec = []
        w_rec = []
        v_act = []
        w_act = []
        cross_err = []
        while not rospy.is_shutdown():
            temp_pose=[0,0,0]
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
                sim_run = False
                if sim_run == True:
                    start_idx = 1
                    xinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][0] + 0.05
                    yinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][1] + 0.05
                    thinit = warthog_ppo_node.warthog_ppo.waypoints_list[start_idx][2]
                    warthog_ppo_node.warthog_ppo.set_pose([xinit, yinit, thinit])
            obs = warthog_ppo_node.warthog_ppo.get_observation()
            curr_pose = warthog_ppo_node.warthog_ppo.get_pose()
            curr_twist = warthog_ppo_node.warthog_ppo.get_twist()
            x = curr_pose[0]
            y = curr_pose[1]
            th = curr_pose[2]
            curr_v = curr_twist[0]
            curr_w = curr_twist[1]
            #twist = warthog_ppo_node.warthog_ppo.get_pytorch_ppo_control(np.array(obs).reshape(1,42))
            twist = warthog_ppo_node.warthog_ppo.get_pytorch_ppo_control(obs)
            poses.append(np.array([x, y, th, curr_v, curr_w, twist[0], twist[1]]))
            traj_file.writelines(f"{x},{y},{th},{curr_v},{curr_w},{twist[0]},{twist[1]}\n")
            #v = np.clip(twist[0][0], 0, 1) * 2.0
            closest_idx = warthog_ppo_node.warthog_ppo.closest_idx
            #v = warthog_ppo_node.warthog_ppo.waypoints_list[closest_idx][3]
            #w = np.clip(twist[0][1], -1, 1) * 2.5
            twist = twist[0]
            v = np.clip(twist[0][0], 0, 1)*4.0
            w = np.clip(twist[0][1], -1, 1)*2.5
            v_rec.append(v)
            w_rec.append(w)
            cross_err.append(warthog_ppo_node.warthog_ppo.current_cross_track)
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
            rate.sleep()
            temp_pose = warthog_ppo_node.warthog_ppo.get_pose()
            temp_twist = warthog_ppo_node.warthog_ppo.get_twist()
            x_pose.append(temp_pose[0])
            y_pose.append(temp_pose[1])
            v_act.append(temp_twist[0])
            w_act.append(temp_twist[1])
        warthog_ppo_node.warthog_ppo.path_lock.acquire()
        cx = [i[0] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        cy = [i[1] for i in warthog_ppo_node.warthog_ppo.waypoints_list]
        warthog_ppo_node.warthog_ppo.path_lock.release()
        plt.figure(1)
        plt.plot(v_rec, 'g', label='controller commanded v')
        plt.plot(v_act, 'r', label='warthog actual v')
        plt.legend()
        plt.title("linear velocity plots")
        plt.figure(2)
        plt.plot(w_rec, 'g', label='controller commanded w')
        plt.plot(w_act, 'r', label='warthog actual w')
        plt.legend()
        plt.title("angular velocity plots")
        plt.figure(3)
        #plt.show()
        plt.plot(cx, cy, '+b', label='waypoints')
        x_ = temp_pose[0]
        y_ = temp_pose[1]
        th_ = temp_pose[2]
        plt.arrow(x_,y_, 2*math.cos(th_), 2*math.sin(th_), head_length=.5, head_width=.2)
        print(cx)
        print("diff")
        print(cy)
        plt.plot(x_pose, y_pose, '+g', label='controller trajectory')
        plt.xlim(x_-30, x_ +30)
        plt.ylim(y_-30, y_ +30)
        plt.title("Ground truth vs Controller response")
        plt.legend()
        plt.figure(4)
        plt.plot(cross_err, '+r', label='crosstrack error')
        plt.legend()
        plt.title('Crosstrack error with Time')
        plt.show()
if __name__=='__main__':
    main()
