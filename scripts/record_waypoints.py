#!/usr/bin/env python
import rospy
from nav_msgs.msg import Path
got_data = False
f = open("planner_way.txt", 'w')
def path_cb(data):
    global f, got_data
    x = []
    y = []
    z = []
    if got_data:
        return
    i = 0
    for path_point in data.poses:
        x.append(path_point.pose.position.x)
        y.append(path_point.pose.position.y)
        z.append(path_point.pose.position.z)
        i = i+1
        #f.write(str(x) + ',' + str(y) + ',0,' + str(z) + ',2.2\n')
    x = x[:-1]
    y = y[:-1]
    z = z[:-1]
    for j in range(0,100):
        x.append(x[i-2 + j] + x[i-3] - x[i-4])
        y.append(y[i-2 + j] + y[i-3] - y[i-4])
        z.append(2.5)
    for k in range(0, len(x)):
        f.write(str(x[k]) + ',' + str(y[k]) + ',0,' + str(z[k]) + ', -2.97\n')
    got_data = True
    f.close()
rospy.init_node("planner_point_record")
rospy.Subscriber("local_planning/path/final_trajectory", Path, path_cb)
rospy.spin()

