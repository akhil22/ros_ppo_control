from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rospy
import tensorflow as tf
import threading
import numpy as np
import utm
import csv
import math
class HuskyPPO:
    def __init__(self):
        self.twist_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.waypoints_list = []
        self.pose = None
        self.twist = None
        self.graph = None
        self.graph_def = None
        self.observation = None
        self.closest_idx = 0
        self.closest_dist = math.inf
        self.num_waypoints = 0
        self.horizon = 10
    def zero_to_2pi(self, theta):
        if theta < 0:
            theta = 2*math.pi + theta
        elif theta > 2*math.pi:
            theta = theta - 2*math.pi
        return theta
    def get_theta(self, xdiff, ydiff):
        theta = math.atan2(ydiff, xdiff)
        return self.zero_to_2pi(theta)
    def get_dist(self, waypoint, pose):
        xdiff = pose[0] - waypoint[0]
        ydiff = pose[1] - waypoint[1]
        return math.sqrt(xdiff*xdiff, ydiff*ydiff)
    def read_waypoint_file(self, filename):
        with open(filename) as csv_file:
            pos = csv.reader(csv_file, delimiter=',')
            for row in pos:
                #utm_cord = utm.from_latlon(float(row[0]), float(row[1]))
                utm_cord = [float(row[0]), float(row[1])]
                self.waypoints_list.append(np.array([utm_cord[0], utm_cord[1], float(row[2]),float(row[3])]))
            for i in range(0, len(self.waypoints_list) - 1):
                xdiff = self.waypoints_list[i+1][0] - self.waypoints_list[i][0]
                ydiff = self.waypoints_list[i+1][1] - self.waypoints_list[i][1]
                self.waypoints_list[i][2] = self.zero_to_2pi(self.get_theta(xdiff, ydiff))
            self.waypoints_list[i+1][2] = self.waypoints_list[i][2]
            self.num_waypoints = i+2
        pass
    def update_closest_idx(self, pose):
        idx = closest_idx
        for i in range(self.closest_idx, self.num_waypoints):
            dist = self.get_dist(self.waypoints_list[i], pose)
            if(dist <= self.closest_dist):
                self.closest_dist = dist
                idx = i
            else:
                break
        self.closest_idx = idx
    def get_observation(self):
        obs = [0]*(self.horizon*4 + 2)
        pose = self.get_pose()
        self.update_closest_idx(pose)
        j = 0
        for i in closest_idx
            r = get_dist(i, pose)
            xdiff = i[0] - pose[0]
            ydiff = i[1] - pose[1]
            th = get_theta(xdiff, ydiff)
            yaw_error = self.zero_to_2p(i[2] - pose[2])
            vel = i[3]
            obs[j] = r
            obs[j+1] = th
            obs[j+2] = yaw_error
            obs[j+3] = vel
    def get_pose(self):
        self.pose_lock.acquire()
        current_pos = self.pose
        self.pose_lock.release()
        return current_pos
    def get_twist(self):
        self.twist_lock.acquire()
        current_twist = self.twist
        self.twist_lock.release()
        return current_twist
    def set_pose(self, pose):
        self.pose_lock.acquire()
        self.pose = pose
        self.pose_lock.release()
    def set_twist(self, twist):
        self.twist_lock.acquire()
        self.twist = twist
        self.twist_lock.release()
    def odom_callback(self, data):
        pass
    def gps_callback(self, data):
        pass
    def read_tf_frozen_graph(self, filepath):
        with tf.gfile.GFile(filepath, "rb") as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as self.graph:
            tf.import_graph_def(self.graph_def, name="")
        self.observation = self.graph.get_tensor_by_name("vector_observation:0")
    def get_control(self, observation):
        feed_dict = {self.observation:observation}
        sess = tf.Session(graph = self.graph)
        op = self.graph.get_tensor_by_name("action:0")
        print(sess.run(op, feed_dict))
husky_ppo = HuskyPPO()
husky_ppo.read_waypoint_file("waypoint.txt")
print(husky_ppo.waypoints_list)
#print(husky_ppo.zero_to_2pi(-0.1))
#print(husky_ppo.zero_to_2pi(2*math.pi + 0.5))
'''husky_ppo = HuskyPPO()
husky_ppo.read_tf_frozen_graph("/home/sai/hdd1/ml-master/ml-agents/config/ppo/results/wlong_path19/3DBall/dogs-cats-model.pb")
observation = np.random.rand(1,42)
husky_ppo.get_control(observation)
husky_ppo.get_control(observation)
husky_ppo.get_control(observation)'''
