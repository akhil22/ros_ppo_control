from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rospy
import tensorflow as tf
import threading
import numpy as np
class HuskyPPO:
    def __init__(self):
        self.twist_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.pose = []
        self.twist = []
        self.graph = None
        self.graph_def = None
        self.observation = None
    def read_waypoint_file(self, filename):
        pass
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
'''husky_ppo = HuskyPPO()
husky_ppo.read_tf_frozen_graph("/home/sai/hdd1/ml-master/ml-agents/config/ppo/results/wlong_path19/3DBall/dogs-cats-model.pb")
observation = np.random.rand(1,42)
husky_ppo.get_control(observation)
husky_ppo.get_control(observation)
husky_ppo.get_control(observation)'''
