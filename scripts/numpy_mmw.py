import math
import numpy as np

def mmw_to_xylwp_array(mmw_msg):
    info = []
    num = len(mmw_msg.markers)
    for i in range(num):
        m = mmw_msg.markers[i]
        
        x = m.pose.position.x
        y = m.pose.position.y
        l = m.scale.x
        w = m.scale.y
        
        alpha = m.pose.orientation.z / m.pose.orientation.w
        p = 2 * math.atan(alpha)
        
        info.append([x, y, l, w, p])
    return np.array(info)
