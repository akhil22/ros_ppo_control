import rospy
#import torch
from stable_baselines import PPO2
import threading
import numpy as np
import utm
import csv
import math
import gym
class PPOControl:
    def __init__(self):
        self.twist_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.path_lock = threading.Lock()
        self.waypoints_list = []
        self.pose = None
        self.twist = None
        self.model = None
        self.graph_def = None
        self.observation = None
        self.closest_idx = 0
        self.closest_dist = math.inf
        self.num_waypoints = 0
        self.horizon = 10
    def set_waypoints_from_list(self, x_list, y_list, th_list, v_list):
        #self.path_lock.acquire()
        self.waypoint_list = []
        for i in range(0, len(x_list)):
            self.waypoints_list.append(np.array([x_list[i], y_list[i], th_list[i], v_list[i]]))
            if i > 0:
                xdiff = self.waypoints_list[i][0] - self.waypoints_list[i-1][0]
                ydiff = self.waypoints_list[i][1] - self.waypoints_list[i-1][1]
                self.waypoints_list[i-1][2] = self.zero_to_2pi(self.get_theta(xdiff, ydiff))
        self.waypoints_list[i-1][2] = self.waypoints_list[i-2][2]
        self.num_waypoints = i
        '''for i in range(0, len(self.waypoints_list) - 1):
            xdiff = self.waypoints_list[i+1][0] - self.waypoints_list[i][0]
            ydiff = self.waypoints_list[i+1][1] - self.waypoints_list[i][1]
            self.waypoints_list[i][2] = self.zero_to_2pi(self.get_theta(xdiff, ydiff))
        self.waypoints_list[i+1][2] = self.waypoints_list[i][2]
        self.num_waypoints = i+2'''
        #self.path_lock.release()
    def zero_to_2pi(self, theta):
        if theta < 0:
            theta = 2*math.pi + theta
        elif theta > 2*math.pi:
            theta = theta - 2*math.pi
        return theta
    def pi_to_pi(self, theta):
        if theta < -math.pi:
            theta = theta + 2*math.pi
        elif theta > math.pi:
            theta = theta - 2*math.pi
        return theta
    def get_theta(self, xdiff, ydiff):
        theta = math.atan2(ydiff, xdiff)
        return self.zero_to_2pi(theta)
    def get_dist(self, waypoint, pose):
        xdiff = pose[0] - waypoint[0]
        ydiff = pose[1] - waypoint[1]
        return math.sqrt(xdiff*xdiff + ydiff*ydiff)
    def read_waypoint_file(self, filename):
        with open(filename) as csv_file:
            pos = csv.reader(csv_file, delimiter=',')
            for row in pos:
                #utm_cord = utm.from_latlon(float(row[0]), float(row[1]))
                utm_cord = [float(row[0]), float(row[1])]
                #phi = math.pi/4
                phi = 0.
                xcoord = utm_cord[0]*math.cos(phi) + utm_cord[1]*math.sin(phi)
                ycoord = -utm_cord[0]*math.sin(phi) + utm_cord[1]*math.cos(phi)
             #   self.waypoints_list.append(np.array([xcoord, ycoord, float(row[2]),float(row[3])]))
                #self.waypoints_list.append(np.array([xcoord, ycoord, float(row[2]),2.5]))
                self.waypoints_list.append(np.array([utm_cord[0], utm_cord[1], float(row[2]),float(row[3])]))
               # self.waypoints_list.append(np.array([utm_cord[0], utm_cord[1], float(row[2]), 1.5]))
            for i in range(0, len(self.waypoints_list) - 1):
                xdiff = self.waypoints_list[i+1][0] - self.waypoints_list[i][0]
                ydiff = self.waypoints_list[i+1][1] - self.waypoints_list[i][1]
                self.waypoints_list[i][2] = self.zero_to_2pi(self.get_theta(xdiff, ydiff))
            self.waypoints_list[i+1][2] = self.waypoints_list[i][2]
            self.num_waypoints = i+2
        pass
    def update_closest_idx(self, pose):
        idx = self.closest_idx
        self.closest_dist = math.inf
        for i in range(self.closest_idx, self.num_waypoints):
            dist = self.get_dist(self.waypoints_list[i], pose)
            if(dist <= self.closest_dist):
                self.closest_dist = dist
                idx = i
            else:
                break
        self.closest_idx = idx
    def get_observation(self):
        self.path_lock.acquire()
        obs = [0]*(self.horizon*4 + 2)
        pose = self.get_pose()
        twist = self.get_twist()
        self.update_closest_idx(pose)
        j = 0
        for i in range(0, self.horizon):
            k = i + self.closest_idx
            if k < self.num_waypoints:
                r = self.get_dist(self.waypoints_list[k], pose)
                xdiff = self.waypoints_list[k][0] - pose[0]
                ydiff = self.waypoints_list[k][1] - pose[1]
                th = self.get_theta(xdiff, ydiff)
                vehicle_th = self.zero_to_2pi(pose[2])
                #vehicle_th = -vehicle_th
                #vehicle_th = 2*math.pi - vehicle_th
                yaw_error = self.pi_to_pi(self.waypoints_list[k][2] - vehicle_th)
                vel = self.waypoints_list[k][3]
                obs[j] = r
                obs[j+1] = self.pi_to_pi(th - vehicle_th)
                obs[j+2] = yaw_error
                obs[j+3] = vel - twist[0]
            else:
                obs[j] = 0.
                obs[j+1] = 0.
                obs[j+2] = 0.
                obs[j+3] = 0.
            j = j+4
        self.path_lock.release()
        obs[j] = twist[0]
        obs[j+1] = twist[1]
        return obs
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
    #def odom_callback(self, data):
    #    pass
    #def gps_callback(self, data):
    #    pass
    def read_ppo_policy(self, filepath):
        env = gym.make("CartPole-v1")
        self.model = PPO2('MlpPolicy', env, verbose=1);
        self.model = self.model.load(filepath);
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
        return sess.run(op, feed_dict)
    def get_ppo_control(self, observation):
        action, _states = self.model.predict(observation, deterministic=True)
        return [action]
#print(husky_ppo.zero_to_2pi(-0.1))
#print(husky_ppo.zero_to_2pi(2*math.pi + 0.5))
'''husky_ppo = HuskyPPO()
husky_ppo.read_tf_frozen_graph("/home/sai/hdd1/ml-master/ml-agents/config/ppo/results/wlong_path19/3DBall/dogs-cats-model.pb")
observation = np.random.rand(1,42)
husky_ppo.get_control(observation)
husky_ppo.get_control(observation)
husky_ppo.get_control(observation)'''
