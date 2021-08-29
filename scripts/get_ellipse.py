# -*- coding: UTF-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

def get_ellipse(x0, y0, a, b, phi):
    # 功能：计算椭圆上的点集
    # 输入：x0 <class 'float'> 圆心X坐标
    #      y0 <class 'float'> 圆心Y坐标
    #      a <class 'float'> 长轴长
    #      b <class 'float'> 短轴长
    #      phi <class 'float'> 旋转角度，范围[0, pi)
    # 输出：xs <class 'numpy.ndarray'> (n,)
    #      ys <class 'numpy.ndarray'> (n,)
    
    angles = np.arange(0, 2 * np.pi, 0.01)
    xs = []
    ys = []
    
    for angle in angles:
        x = a * math.cos(angle)
        y = b * math.sin(angle)
        l = math.sqrt(x ** 2 + y ** 2)
        theta = math.atan2(y, x)
        
        new_theta = theta + phi
        new_x = x0 + l * math.cos(new_theta)
        new_y = y0 + l * math.sin(new_theta)
        xs.append(new_x)
        ys.append(new_y)
        
    return xs, ys

if __name__ == '__main__':
    xs, ys = get_ellipse(0, 0, 10, 4, 0)
    
    fig, ax = plt.subplots()
    ax.plot(xs, ys, linewidth=1, linestyle='-')
    plt.show()
    
    
