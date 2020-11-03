#!/usr/bin/env python
import rospy
from nav_msgs.msg import Path
got_data = False
f = open("planner_way.txt", 'w')
def path_cb(data):
    global f, got_data
    if got_data:
        return
    for path_point in data.poses:
        x = path_point.pose.position.x
        y = path_point.pose.position.y
        z = path_point.pose.position.z
        f.write(str(x) + ',' + str(y) + ',0,' + str(z) + ',2.2\n')
    got_data = True
    f.close()
rospy.init_node("planner_point_record")
rospy.Subscriber("local_planning/path/final_trajectory", Path, path_cb)
rospy.spin()

