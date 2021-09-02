import numpy as np

def mmw_to_xyzlwhp_array(mmw_msg):
    num = len(mmw_msg.markers)
    if num == 0:
        raise ValueError('len(mmw_msg.markers) must be greater than 0.')
        
    info = []
    for i in range(num):
        m = mmw_msg.markers[i]
        
        x = m.pose.position.x
        y = m.pose.position.y
        z = m.pose.position.z
        l = m.scale.x
        w = m.scale.y
        h = m.scale.z
        
        alpha = m.pose.orientation.z / m.pose.orientation.w
        p = 2 * math.atan(alpha)
        
        info.append([x, y, z, l, w, h, p])
    return np.array(info)
