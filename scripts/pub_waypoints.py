import rospy
from nav_msgs.msg import Path
import utm
import csv
from geometry_msgs.msg import PoseStamped
rospy.init_node("waypoints_publisher")
filename = "waypoints.txt"
pub = rospy.Publisher("/waypoints_topic", Path, queue_size = 100, latch=True)
path_msg = Path()
path_msg.header.stamp = rospy.get_rostime()
path_msg.header.frame_id = "map"
with open(filename) as csv_file:
    pos = csv.reader(csv_file, delimiter=',')
    for row in pos:
        utm_cord = utm.from_latlon(float(row[0]), float(row[1]))
        waypoint_pose = PoseStamped()
        waypoint_pose.pose.position.x = utm_cord[0]
        waypoint_pose.pose.position.y = utm_cord[1]
        path_msg.poses.append(waypoint_pose)
        #utm_cord = [float(row[0]), float(row[1])]
r = rospy.Rate(1)
while(not rospy.is_shutdown()):
    pub.publish(path_msg)
