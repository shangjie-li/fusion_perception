# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import math

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

PI = 3.14159

def find_min_enclosing_circle_using_minEnclosingCircle(xs, ys):
    # 功能：提取平面点云最小包围圆
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    # 输出：x <class 'float'> 圆心横坐标
    #      y <class 'float'> 圆心纵坐标
    #      radius <class 'float'> 半径
    
    xs = xs * 100
    ys = ys * 100
    xs = xs.astype(np.int)
    ys = ys.astype(np.int)
    
    pts = np.array((xs, ys)).T
    
    # cv2.minEnclosingCircle只处理整数数据
    # pts <class 'numpy.ndarray'> (n, 2)
    (x, y), radius = cv2.minEnclosingCircle(pts)
    
    x = x / 100.0
    y = y / 100.0
    radius = radius / 100.0
    return x, y, radius
    
def find_point_clouds_boundary(xs, ys, resolution=10):
    # 功能：提取平面点云外部轮廓
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    #      resolution <class 'int'> 代表分辨率，单位度
    # 输出：hull <class 'numpy.ndarray'> (m, 1, 2) m为轮廓点数
    
    resolution = int(resolution)
    n = 360 // resolution
    hull = np.zeros((n, 1, 2))
    hull_dd = np.zeros((n))
    
    pts = np.array((xs, ys)).T
    num = pts.shape[0]
    
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    for i in range(num):
        dx = pts[i, 0] - center_x
        dy = pts[i, 1] - center_y
        dd = dx ** 2 + dy ** 2
        # math.atan2(y, x)返回值范围(-pi, pi]
        angle = math.atan2(dy, dx)
        # angle范围[0, 2pi)
        if angle < 0:
            angle += 2 * PI
        # angle范围[0, 360)
        angle = int(angle * 180 / PI)
        
        idx = angle // resolution
        if dd > hull_dd[idx]:
            hull[idx, 0, 0] = pts[i, 0]
            hull[idx, 0, 1] = pts[i, 1]
            hull_dd[idx] = dd
            
    # 平面轮廓 <class 'numpy.ndarray'> (m, 1, 2) m为轮廓点数
    hull = hull[np.logical_or(hull[:, 0, 0], hull[:, 0, 1])]
    return hull
    
def find_point_clouds_boundary_using_convexHull(xs, ys):
    # 功能：提取平面点云外部轮廓
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    # 输出：hull <class 'numpy.ndarray'> (m, 1, 2) m为轮廓点数
    
    xs = xs * 100
    ys = ys * 100
    xs = xs.astype(np.int)
    ys = ys.astype(np.int)
    
    pts = np.array((xs, ys)).T
    
    # cv2.convexHull只处理整数数据
    # pts <class 'numpy.ndarray'> (n, 2)
    hull = cv2.convexHull(pts)
    
    # 平面轮廓 <class 'numpy.ndarray'> (m, 1, 2) m为轮廓点数
    hull = hull / 100.0
    return hull
    
def sort_polygon(polygon):
    # 功能：将多边形各顶点按逆时针方向排序
    # 输入：polygon <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    # 输出：polygon <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    
    num = polygon.shape[0]
    center_x = np.mean(polygon[:, 0, 0])
    center_y = np.mean(polygon[:, 0, 1])
    angles = []
    for i in range(num):
        dx = polygon[i, 0, 0] - center_x
        dy = polygon[i, 0, 1] - center_y
        # math.atan2(y, x)返回值范围(-pi, pi]
        angle = math.atan2(dy, dx)
        # angle范围[0, 2pi)
        if angle < 0:
            angle += 2 * PI
        angles.append(angle)
    idxs = list(np.argsort(angles))
    
    return polygon[idxs]
    
def test_point_in_polygon_using_pointPolygonTest(x, y, polygon):
    # 功能：测试点是否在多边形内
    # 输入：x <class 'float'> 代表点的横坐标
    #      y <class 'float'> 代表点的纵坐标
    #      polygon <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    # 输出：flag <class 'int'> +1代表在内部，-1代表在外部，0代表在轮廓上
    
    x = x * 100
    y = y * 100
    point = (int(x), int(y))
    
    polygon = polygon * 100
    polygon = polygon.astype(np.int)
    
    # cv2.pointPolygonTest只处理整数数据
    # point <class 'tuple'> 待测试点坐标
    try:
        flag = cv2.pointPolygonTest(polygon, point, False)
    except:
        flag = False
    
    return flag
    
def find_boundary_close_to_origin_in_polygon(polygon):
    # 功能：保留多边形各顶点中靠近坐标原点的部分
    # 输入：polygon <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    # 输出：polygon_remained <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    
    # 将多边形中的点按逆时针方向排序
    polygon_sorted = sort_polygon(polygon)
    
    num = polygon_sorted.shape[0]
    
    # 判断X负半轴是否穿过多边形
    xaxis_in_polygon = False
    for j in range(num):
        x, y = polygon_sorted[j, 0, 0], 0
        if x < 0:
            flag = test_point_in_polygon_using_pointPolygonTest(x, y, polygon_sorted)
            if flag > 0:
                xaxis_in_polygon = True
    
    # 计算多边形各顶点相对坐标原点的极角，范围[0, 2pi)
    angles = []
    for j in range(num):
        # math.atan2(y, x)返回值范围(-pi, pi]
        angle = math.atan2(polygon_sorted[j, 0, 1], polygon_sorted[j, 0, 0])
        
        # 如果X负半轴穿过多边形，将angle范围调整为[0, 2pi)
        if xaxis_in_polygon and angle < 0:
            angle += 2 * PI
        angles.append(angle)
    
    angles_array = np.array(angles)
    # 极角最小点索引
    min_idx = np.where(angles == angles_array.min())[0][0]
    # 极角最大点索引
    max_idx = np.where(angles == angles_array.max())[0][0]
    
    # 以极角最小点和极角最大点为界限，保留多边形各顶点中靠近坐标原点的部分
    if max_idx < min_idx:
        remain_idxs = list(range(max_idx, min_idx + 1))
    else:
        remain_idxs = list(range(max_idx, num)) + list(range(0, min_idx + 1))
    polygon_remained = polygon_sorted[remain_idxs, :, :]
    
    return polygon_remained
    
def approximate_polygon_using_approxPolyDP(polygon, epsilon=0.10, closed=False):
    # 功能：逼近多边形
    #      应用Douglas-Peucker算法
    # 输入：polygon <class 'numpy.ndarray'> (n, 1, 2) n为多边形顶点数
    #      epsilon <class 'double'> 代表逼近准确度，亦为距离误差
    #      closed <class 'bool'> 代表多边形闭合标志
    # 输出：polygon_new <class 'numpy.ndarray'> (m, 1, 2) m为多边形顶点数
    
    epsilon = epsilon * 100
    polygon = polygon * 100
    polygon = polygon.astype(np.int)
    
    # cv2.approxPolyDP只处理整数数据
    polygon_new = cv2.approxPolyDP(polygon, epsilon, closed)
    
    # 多边形 <class 'numpy.ndarray'> (m, 1, 2) m为多边形顶点数
    polygon_new = polygon_new / 100.0
    return polygon_new
    
def project_point_clouds_on_line(xs, ys, x0, y0, phi):
    # 功能：将平面点云投影到直线上
    #      设某点坐标(x, y)，其投影点坐标(xp, yp)，直线经过点(x0, y0)，与直线同方向的单位向量(vx, vy)
    #      由(xp-x, yp-y)与(vx, vy)垂直以及(xp-x0, yp-y0)与(vx, vy)平行，可求得(xp, yp)
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    #      x0 <class 'float'> 为直线经过的点的横坐标
    #      y0 <class 'float'> 为直线经过的点的纵坐标
    #      phi <class 'float'> 为直线与横轴的夹角，范围[0, pi)
    # 输出：xsp <class 'numpy.ndarray'> (n,) 代表投影点横坐标
    #      ysp <class 'numpy.ndarray'> (n,) 代表投影点纵坐标
    #      xp_max <class 'float'> 为投影后的端点1横坐标
    #      yp_max <class 'float'> 为投影后的端点1纵坐标
    #      xp_min <class 'float'> 为投影后的端点2横坐标
    #      yp_min <class 'float'> 为投影后的端点2纵坐标
    
    if phi < 0 or phi >= PI:
        raise Exception('phi is not in [0, pi).')
        
    # 代表直线方向的单位向量
    vx = math.cos(phi)
    vy = math.sin(phi)
    
    # 点云投影到直线后的横纵坐标
    num = xs.shape[0]
    if abs(vy) > 0.0001:
        ysp = ys * vy ** 2 + xs * vx * vy + y0 * vx ** 2 - x0 * vx * vy
        xsp = ((ysp - y0) * vx + x0 * vy) / vy
        
        # 按纵坐标寻找两端点
        max_idx = np.where(ysp == ysp.max())[0][0]
        xp_max = xsp[max_idx]
        yp_max = ysp[max_idx]
        
        min_idx = np.where(ysp == ysp.min())[0][0]
        xp_min = xsp[min_idx]
        yp_min = ysp[min_idx]
        
    else:
        ysp = y0 * np.ones((num))
        xsp = xs
        
        # 按横坐标寻找两端点
        max_idx = np.where(xsp == xsp.max())[0][0]
        xp_max = xsp[max_idx]
        yp_max = ysp[max_idx]
        
        min_idx = np.where(xsp == xsp.min())[0][0]
        xp_min = xsp[min_idx]
        yp_min = ysp[min_idx]
        
    return xsp, ysp, xp_max, yp_max, xp_min, yp_min
    
def compute_distance_between_point_and_line(x, y, x0, y0, phi):
    # 功能：计算点到直线距离
    #      设某点坐标(x, y)，其投影点坐标(xp, yp)，直线经过点(x0, y0)，与直线同方向的单位向量(vx, vy)
    #      由(xp-x, yp-y)与(vx, vy)垂直以及(xp-x0, yp-y0)与(vx, vy)平行，可求得(xp, yp)
    # 输入：x <class 'float'> 代表点的横坐标
    #      y <class 'float'> 代表点的纵坐标
    #      x0 <class 'float'> 为直线经过的点的横坐标
    #      y0 <class 'float'> 为直线经过的点的纵坐标
    #      phi <class 'float'> 为直线与横轴的夹角，范围[0, pi)
    # 输出：distance <class 'float'> 代表距离
    
    if phi < 0 or phi >= PI:
        raise Exception('phi is not in [0, pi).')
        
    # 代表直线方向的单位向量
    vx = math.cos(phi)
    vy = math.sin(phi)
    
    # 点云投影到直线后的横纵坐标
    if abs(vy) > 0.0001:
        yp = y * vy ** 2 + x * vx * vy + y0 * vx ** 2 - x0 * vx * vy
        xp = ((yp - y0) * vx + x0 * vy) / vy
        
    else:
        yp = y0
        xp = x
        
    dd = (xp - x) ** 2 + (yp - y) ** 2
    distance = math.sqrt(dd)
    return distance
    
def compute_angle_of_longest_line_in_polygon(polygon, closed=False, m=1):
    # 功能：计算多边形中最长的m条线段对应的角度
    # 输入：polygon <class 'numpy.ndarray'> (n, 1, 2) n为多边形顶点数
    #      closed <class 'bool'> 代表多边形闭合标志
    #      m <class 'int'> 代表计算的次数，m小于n
    # 输出：angles_numpy <class 'numpy.ndarray'> (m,) 代表m个角度，范围[0, pi)，按对应的线段长度排序
    
    n = polygon.shape[0]
    
    if m > n:
        raise Exception('m > polygon.shape[0].')
        
    if m == n and not closed:
        raise Exception('m = polygon.shape[0] and not closed.')
        
    dds = []
    angles = []
    
    num = n if closed else n - 1
    for j in range(num):
        x1 = polygon[j, 0, 0]
        y1 = polygon[j, 0, 1]
        x2 = polygon[(j + 1) % n, 0, 0]
        y2 = polygon[(j + 1) % n, 0, 1]
        
        # 计算线段长度及对应的方向角
        dd = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            angle = PI / 2
        else:
            # math.atan(x)返回值范围(-pi/2, pi/2)
            angle = math.atan(dy / dx)
            # angle范围[0, pi)
            if angle < 0:
                angle += PI
            
        dds.append(dd)
        angles.append(angle)
        
    # 按线段长度排序
    idxs = np.argsort(dds)
    idxs = np.flipud(idxs)
    idxs = idxs[:m]
    
    angles_numpy = np.array(angles)[idxs]
    return angles_numpy
    
def compute_scd_of_point_clouds_in_bounding_box(xs, ys, p1, p2, p3, p4):
    # 功能：计算包围盒中点云的特征距离和（Sum of Characteristic Distance，SCD）
    #      p1、p2、p3、p4应按序分布
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    #      p1 <class 'numpy.ndarray'> (2,) 包围盒顶点P1坐标
    #      p2 <class 'numpy.ndarray'> (2,) 包围盒顶点P2坐标
    #      p3 <class 'numpy.ndarray'> (2,) 包围盒顶点P3坐标
    #      p4 <class 'numpy.ndarray'> (2,) 包围盒顶点P4坐标
    # 输出：scd <class 'float'> 点云的特征距离和
    
    # 点云到p1p2的距离
    vx = p2[0] - p1[0]
    vz = p2[1] - p1[1]
    if abs(vx) > 0.0001:
        k = vz / vx
        dis_12 = abs(k * xs - ys - k * p1[0] + p1[1]) / np.sqrt(k ** 2 + 1)
    else:
        dis_12 = abs(xs - p1[0])
        
    # 点云到p2p3的距离
    vx = p3[0] - p2[0]
    vz = p3[1] - p2[1]
    if abs(vx) > 0.0001:
        k = vz / vx
        dis_23 = abs(k * xs - ys - k * p2[0] + p2[1]) / np.sqrt(k ** 2 + 1)
    else:
        dis_23 = abs(xs - p2[0])
        
    # 点云到p3p4的距离
    vx = p4[0] - p3[0]
    vz = p4[1] - p3[1]
    if abs(vx) > 0.0001:
        k = vz / vx
        dis_34 = abs(k * xs - ys - k * p3[0] + p3[1]) / np.sqrt(k ** 2 + 1)
    else:
        dis_34 = abs(xs - p3[0])
        
    # 点云到p4p1的距离
    vx = p1[0] - p4[0]
    vz = p1[1] - p4[1]
    if abs(vx) > 0.0001:
        k = vz / vx
        dis_41 = abs(k * xs - ys - k * p4[0] + p4[1]) / np.sqrt(k ** 2 + 1)
    else:
        dis_41 = abs(xs - p4[0])
        
    # 点云到包围盒各边界的距离
    dis = np.array([dis_12, dis_23, dis_34, dis_41]).T
    
    # 点云到包围盒的特征距离
    dis_min = dis.min(axis=1)
    
    # 点云到包围盒的特征距离的和
    scd = dis_min.sum(axis=0)
    
    return scd
    
def fit_2d_bounding_box(xs, ys, phi):
    # 功能：根据方向角计算2D带方向包围盒
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    #      phi <class 'float'> 为直线与横轴的夹角，范围[0, pi)
    # 输出：pnn <class 'numpy.ndarray'> (2,) 包围盒顶点Pnn坐标
    #      pfn <class 'numpy.ndarray'> (2,) 包围盒顶点Pfn坐标
    #      pff <class 'numpy.ndarray'> (2,) 包围盒顶点Pff坐标
    #      pnf <class 'numpy.ndarray'> (2,) 包围盒顶点Pnf坐标
    #      x0 <class 'float'> 代表2D包围盒中心点横坐标
    #      y0 <class 'float'> 代表2D包围盒中心点纵坐标
    #      l <class 'float'> 代表2D包围盒的长
    #      w <class 'float'> 代表2D包围盒的宽
    #      phi <class 'float'> 代表方向角，亦为长边与横轴的夹角，范围[0, pi)
    
    if phi < 0 or phi >= PI:
        raise Exception('phi is not in [0, pi).')
        
    angle = phi
    angle_second = angle + PI / 2 if angle < PI / 2 else angle - PI / 2
    
    # 将点云投影至局部坐标系
    _, _, xp1, yp1, xp2, yp2 = project_point_clouds_on_line(xs, ys, 0, 0, angle)
    _, _, xp3, yp3, xp4, yp4 = project_point_clouds_on_line(xs, ys, 0, 0, angle_second)
    
    # 计算投影点与原点的距离，选出近点和远点
    dd1 = xp1 ** 2 + yp1 ** 2
    dd2 = xp2 ** 2 + yp2 ** 2
    
    if dd1 < dd2:
        xpl_near = xp1
        ypl_near = yp1
        xpl_far = xp2
        ypl_far = yp2
    else:
        xpl_near = xp2
        ypl_near = yp2
        xpl_far = xp1
        ypl_far = yp1
        
    # 计算投影点与原点的距离，选出近点和远点
    dd3 = xp3 ** 2 + yp3 ** 2
    dd4 = xp4 ** 2 + yp4 ** 2
    
    if dd3 < dd4:
        xpw_near = xp3
        ypw_near = yp3
        xpw_far = xp4
        ypw_far = yp4
    else:
        xpw_near = xp4
        ypw_near = yp4
        xpw_far = xp3
        ypw_far = yp3
    
    # 根据近点和远点计算包围盒顶点
    xnn = xpl_near + xpw_near
    ynn = ypl_near + ypw_near
    
    xfn = xpl_far + xpw_near
    yfn = ypl_far + ypw_near
    
    xff = xpl_far + xpw_far
    yff = ypl_far + ypw_far
    
    xnf = xpl_near + xpw_far
    ynf = ypl_near + ypw_far
    
    # 包络矩形的顶点
    pnn = np.array([xnn, ynn])
    pfn = np.array([xfn, yfn])
    pff = np.array([xff, yff])
    pnf = np.array([xnf, ynf])
    
    # 计算包围盒的长宽
    l = math.sqrt((xp1 - xp2) ** 2 + (yp1 - yp2) ** 2)
    w = math.sqrt((xp3 - xp4) ** 2 + (yp3 - yp4) ** 2)
    
    if l > w:
        phi = angle
    else:
        phi = angle_second
        temp = l
        l = w
        w = temp
    
    # 计算包围盒的中心点坐标
    x0 = (xp1 + xp2 + xp3 + xp4) / 2
    y0 = (yp1 + yp2 + yp3 + yp4) / 2
    
    return pnn, pnf, pff, pfn, x0, y0, l, w, phi
    
def fit_2d_obb(xs, ys):
    # 功能：拟合2D带方向包围盒
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    # 输出：x0 <class 'float'> 代表2D包围盒中心点横坐标
    #      y0 <class 'float'> 代表2D包围盒中心点纵坐标
    #      l <class 'float'> 代表2D包围盒的长
    #      w <class 'float'> 代表2D包围盒的宽
    #      phi <class 'float'> 代表方向角，亦为长边与横轴的夹角，范围[0, pi)
    #      polygon <class 'numpy.ndarray'> (n, 1, 2) 轮廓多边形，n为多边形顶点数
    #      polygon_approximated <class 'numpy.ndarray'> (n, 1, 2) 近似多边形，n为多边形顶点数
    #      polygon_approximated_is_closed <class 'bool'> 近似多边形闭合标志
    
    # STEP1
    # 提取点云的轮廓多边形（闭合）
    polygon = find_point_clouds_boundary_using_convexHull(xs, ys)
    
    # 保留多边形各顶点中靠近坐标原点的部分
    polygon_remained = find_boundary_close_to_origin_in_polygon(polygon)
    
    # STEP2
    # 对点云的轮廓多边形进行逼近（非闭合）
    polygon_approximated = approximate_polygon_using_approxPolyDP(polygon_remained)
    polygon_approximated_is_closed = False
    
    # STEP3
    # 计算最长线段对应的方向角，亦为包围盒长边方向角
    angles = compute_angle_of_longest_line_in_polygon(polygon_approximated)
    angle = angles[0]
    
    # STEP4
    # 根据方向角计算2D带方向包围盒
    _, _, _, _, x0, y0, l, w, phi = fit_2d_bounding_box(xs, ys, angle)
    
    return x0, y0, l, w, phi, polygon, polygon_approximated, polygon_approximated_is_closed
    
def fit_2d_obb_using_iterative_procedure(xs, ys):
    # 功能：拟合2D带方向包围盒
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    # 输出：x0 <class 'float'> 代表2D包围盒中心点横坐标
    #      y0 <class 'float'> 代表2D包围盒中心点纵坐标
    #      l <class 'float'> 代表2D包围盒的长
    #      w <class 'float'> 代表2D包围盒的宽
    #      phi <class 'float'> 代表方向角，亦为长边与横轴的夹角，范围[0, pi)
    #      polygon <class 'numpy.ndarray'> (n, 1, 2) 轮廓多边形，n为多边形顶点数
    #      polygon_approximated <class 'numpy.ndarray'> (n, 1, 2) 近似多边形，n为多边形顶点数
    #      polygon_approximated_is_closed <class 'bool'> 近似多边形闭合标志
    
    # STEP1
    # 提取点云的轮廓多边形（闭合）
    polygon = find_point_clouds_boundary_using_convexHull(xs, ys)
    
    # STEP2
    # 对点云的轮廓多边形进行逼近（闭合）
    polygon_approximated = approximate_polygon_using_approxPolyDP(polygon, epsilon=0.10, closed=True)
    polygon_approximated_is_closed = True
    
    # STEP3
    # 计算轮廓中最长的m条线段对应的方向角
    m = min(polygon_approximated.shape[0], 4)
    angles = compute_angle_of_longest_line_in_polygon(polygon_approximated, closed=True, m=m)
    
    # STEP4
    # 对每个方向角拟合2D带方向包围盒，并根据特征距离和（SCD）选取最佳包围盒
    scd_min = float('inf')
    for i in range(len(angles)):
        angle = angles[i]
        
        # 根据方向角计算2D带方向包围盒
        p1, p2, p3, p4, x0_t, y0_t, l_t, w_t, phi_t = fit_2d_bounding_box(xs, ys, angle)
        
        # 计算包围盒中点云的特征距离和
        scd = compute_scd_of_point_clouds_in_bounding_box(xs, ys, p1, p2, p3, p4)
        
        if scd < scd_min:
            x0, y0, l, w, phi = x0_t, y0_t, l_t, w_t, phi_t
            scd_min = scd
            
    return x0, y0, l, w, phi, polygon, polygon_approximated, polygon_approximated_is_closed
    
def transform_2d_point_clouds(xs, ys, phi, x0, y0):
    # 功能：对平面点云进行旋转、平移
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    #      phi <class 'float'> 为旋转角度，单位为rad
    #      x0 <class 'float'> 为横轴平移量
    #      y0 <class 'float'> 为纵轴平移量
    # 输出：xst <class 'numpy.ndarray'> (n,) 代表旋转、平移后横坐标
    #      yst <class 'numpy.ndarray'> (n,) 代表旋转、平移后纵坐标
    
    xst = xs * math.cos(phi) - ys * math.sin(phi)
    yst = xs * math.sin(phi) + ys * math.cos(phi)
    
    xst += x0
    yst += y0
    
    return xst, yst
    
def compute_obb_vertex_coordinates(x0, y0, l, w, phi):
    # 功能：计算包围盒顶点坐标
    # 输入：x0 <class 'float'> 代表2D包围盒中心点横坐标
    #      y0 <class 'float'> 代表2D包围盒中心点纵坐标
    #      l <class 'float'> 代表2D包围盒的长
    #      w <class 'float'> 代表2D包围盒的宽
    #      phi <class 'float'> 代表方向角，亦为长边与横轴的夹角，范围[0, pi)
    # 输出：xst <class 'numpy.ndarray'> (4,) 代表横坐标
    #      yst <class 'numpy.ndarray'> (4,) 代表纵坐标
    
    if phi < 0 or phi >= PI:
        raise Exception('phi is not in [0, pi).')
        
    # 计算顶点坐标
    x1, y1 = l / 2, w / 2
    x2, y2 = - l / 2, w / 2
    x3, y3 = - l / 2, - w / 2
    x4, y4 = l / 2, - w / 2
    
    # 按方向角旋转、按中心点平移
    xl = np.array([x1, x2, x3, x4])
    yl = np.array([y1, y2, y3, y4])
    xst, yst = transform_2d_point_clouds(xl, yl, phi, x0, y0)
    
    return xst, yst
    
def fit_3d_model(xs, ys, zs, radius_threshold=0.6):
    # 功能：拟合3D模型
    #      采用盒子模型描述大尺寸目标，采用圆点模型描述小尺寸目标
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表X坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表Y坐标
    #      zs <class 'numpy.ndarray'> (n,) 代表Z坐标
    #      radius_threshold <class 'float'> 代表目标最小包围圆的半径阈值，低于阈值为小尺寸目标，否则为大尺寸目标
    # 输出：x0 <class 'float'> 代表3D盒子模型或3D圆点模型的中心点X坐标
    #      y0 <class 'float'> 代表3D盒子模型或3D圆点模型的中心点Y坐标
    #      z0 <class 'float'> 代表3D盒子模型或3D圆点模型的中心点Z坐标
    #      l <class 'float'> 3D盒子模型中代表3D包围盒的长，3D圆点模型中代表直径
    #      w <class 'float'> 3D盒子模型中代表3D包围盒的宽，3D圆点模型中代表直径
    #      h <class 'float'> 3D盒子模型中代表3D包围盒的高，3D圆点模型中代表高度
    #      phi <class 'float'> 3D盒子模型中代表方向角，亦为XY平面内长边与X轴的夹角，范围[0, pi)，3D圆点模型中为0
    #      has_orientation <class 'bool'> 是否包含方向
    
    # 提取最小包围圆
    xmec, ymec, radius = find_min_enclosing_circle_using_minEnclosingCircle(xs, ys)
    
    if radius < radius_threshold:
        # 圆点模型
        has_orientation = False
        x0, y0, l, w, phi = xmec, ymec, 2 * radius, 2 * radius, 0
    else:
        # 盒子模型
        has_orientation = True
        x0, y0, l, w, phi, _, _, _ = fit_2d_obb_using_iterative_procedure(xs, ys)
    
    z_max = zs.max()
    z_min = zs.min()
    z0 = (z_max + z_min) / 2
    h = abs(z_max - z_min)
    
    return x0, y0, z0, l, w, h, phi, has_orientation
    
def fit_3d_model_of_cylinder(xs, ys, zs):
    # 功能：拟合3D圆点模型
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表X坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表Y坐标
    #      zs <class 'numpy.ndarray'> (n,) 代表Z坐标
    # 输出：x0 <class 'float'> 代表3D圆点模型的中心点X坐标
    #      y0 <class 'float'> 代表3D圆点模型的中心点Y坐标
    #      z0 <class 'float'> 代表3D圆点模型的中心点Z坐标
    #      l <class 'float'> 3D圆点模型中代表直径
    #      w <class 'float'> 3D圆点模型中代表直径
    #      h <class 'float'> 3D圆点模型中代表高度
    #      phi <class 'float'> 3D圆点模型中为0
    #      has_orientation <class 'bool'> 是否包含方向
    
    # 提取最小包围圆
    xmec, ymec, radius = find_min_enclosing_circle_using_minEnclosingCircle(xs, ys)
    
    has_orientation = False
    x0, y0, l, w, phi = xmec, ymec, 2 * radius, 2 * radius, 0
    
    z_max = zs.max()
    z_min = zs.min()
    z0 = (z_max + z_min) / 2
    h = abs(z_max - z_min)
    
    return x0, y0, z0, l, w, h, phi, has_orientation
    
def fit_3d_model_of_cube(xs, ys, zs):
    # 功能：拟合3D盒子模型
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表X坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表Y坐标
    #      zs <class 'numpy.ndarray'> (n,) 代表Z坐标
    # 输出：x0 <class 'float'> 代表3D盒子模型的中心点X坐标
    #      y0 <class 'float'> 代表3D盒子模型的中心点Y坐标
    #      z0 <class 'float'> 代表3D盒子模型的中心点Z坐标
    #      l <class 'float'> 3D盒子模型中代表3D包围盒的长
    #      w <class 'float'> 3D盒子模型中代表3D包围盒的宽
    #      h <class 'float'> 3D盒子模型中代表3D包围盒的高
    #      phi <class 'float'> 3D盒子模型中代表方向角，亦为XY平面内长边与X轴的夹角，范围[0, pi)
    #      has_orientation <class 'bool'> 是否包含方向
    
    has_orientation = True
    x0, y0, l, w, phi, _, _, _ = fit_2d_obb_using_iterative_procedure(xs, ys)
    
    z_max = zs.max()
    z_min = zs.min()
    z0 = (z_max + z_min) / 2
    h = abs(z_max - z_min)
    
    return x0, y0, z0, l, w, h, phi, has_orientation
    
def find_nearest_point(xs, ys):
    # 功能：提取平面点云中的最近点
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    # 输出：xref <class 'float'> 代表最近点横坐标
    #      yref <class 'float'> 代表最近点纵坐标
    
    pts_dis = np.sqrt(xs ** 2 + ys ** 2)
    idx = np.where(pts_dis == pts_dis.min())[0]
    
    xref = xs[idx]
    yref = ys[idx]
    
    return xref, yref
    
if __name__ == "__main__":
    # 测试数据1
    # 场景：目标为行人，点云数量为73
    xs1 = [1.17, 1.27, 1.15, 1.15, 1.27, 1.14, 1.21, 1.1, 1.07, 1.25, 1.31, 1.16, 1.08, 1.03, 1.32, 1.26, 1.3, 1.15, 1.06, 1.02, 1.33, 1.26, 1.17, 1.09, 1.03, 1.34, 1.34, 1.38, 1.25, 1.3, 1.17,
     1.08, 1.03, 1.01, 1.44, 1.4, 1.36, 1.27, 1.16, 1.07, 1.01, 0.97, 1.44, 1.43, 1.36, 1.27, 1.17, 1.18, 1.11, 1.05, 0.99, 0.96, 1.36, 1.28, 1.23, 1.16, 1.06, 1.0, 1.36, 1.33, 1.23, 1.28, 1.19, 1.13,
      1.06, 1.27, 1.28, 1.17, 1.11, 1.09, 1.27, 1.21, 1.18]
    ys1 = [9.49, 9.75, 9.46, 9.49, 9.74, 9.57, 9.64, 9.58, 9.62, 9.58, 9.61, 9.54, 9.55, 9.6, 9.57, 9.56, 9.6, 9.53, 9.56, 9.62, 9.57, 9.55, 9.53, 9.56, 9.64, 9.59, 9.64, 9.7, 9.56, 9.6, 9.55,
     9.56, 9.62, 9.57, 9.53, 9.48, 9.54, 9.57, 9.54, 9.58, 9.59, 9.62, 9.58, 9.62, 9.57, 9.58, 9.58, 9.6, 9.58, 9.62, 9.67, 9.73, 9.58, 9.6, 9.61, 9.61, 9.64, 9.71, 9.65, 9.72, 9.58, 9.64, 9.59, 9.64,
      9.69, 9.65, 9.74, 9.63, 9.67, 9.7, 9.65, 9.6, 9.63]
    
    # 测试数据2
    # 场景：目标为骑车人，点云数量为40
    xs2 = [-2.1, -2.14, -1.85, -1.73, -1.83, -1.84, -2.21, -1.75, -1.7, -1.74, -1.72, -1.83, -1.73, -1.74, -1.72, -1.83, -1.8, -1.85, -1.85, -1.75, -1.73, -1.83, -1.94, -1.9, -2.04, -2.13, -2.2, -1.82,
     -1.96, -1.92, -1.91, -2.0, -2.04, -2.12, -1.92, -2.01, -1.94, -2.03, -1.94, -2.03]
    ys2 = [8.9, 8.97, 10.07, 9.51, 9.54, 9.62, 9.21, 9.17, 9.22, 9.28, 9.43, 9.13, 9.22, 9.55, 9.61, 9.14, 9.25, 9.57, 9.59, 9.55, 9.6, 9.56, 9.56, 9.57, 9.58, 9.6, 9.61, 9.62, 9.51, 9.6, 9.69, 9.48,
     9.64, 9.66, 9.5, 9.49, 9.51, 9.53, 9.52, 9.52]
    
    # 测试数据3
    # 场景：目标为车辆，点云数量为208，有遮挡
    xs3 = [-4.09, -4.08, -4.16, -4.12, -4.13, -4.17, -4.18, -4.17, -4.1, -4.06, -4.11, -4.12, -4.1, -4.1, -4.11, -4.07, -4.05, -4.05, -4.06, -4.08, -4.07, -4.09, -4.09, -4.16,
     -4.12, -4.13, -4.12, -4.12, -4.09, -4.09, -4.08, -4.07, -4.08, -4.08, -4.06, -4.08, -4.09, -4.07, -4.07, -4.07, -4.09, -4.09, -4.11, -4.09, -4.1, -4.09, -4.1, -4.1, -4.09,
      -4.11, -4.09, -4.1, -4.08, -4.08, -4.04, -4.08, -4.06, -4.06, -4.06, -4.11, -4.12, -4.12, -4.12, -4.11, -4.12, -4.11, -4.12, -4.11, -4.11, -4.13, -4.08, -4.09, -4.08,
       -4.07, -4.06, -4.06, -4.06, -4.07, -4.09, -4.06, -4.07, -4.07, -4.09, -4.08, -4.16, -4.15, -4.14, -4.14, -4.14, -4.12, -4.12, -4.12, -4.13, -4.1, -4.09, -4.1, -4.1, -4.09,
        -4.09, -4.09, -4.1, -4.09, -4.1, -4.09, -4.09, -4.09, -4.1, -4.17, -4.15, -4.13, -4.12, -4.12, -4.14, -4.12, -4.12, -4.12, -4.11, -4.13, -4.18, -4.1, -4.11, -4.11, -4.11,
         -4.12, -4.17, -4.14, -4.13, -4.12, -4.15, -4.15, -4.13, -4.13, -4.14, -4.13, -4.15, -4.12, -4.13, -4.12, -4.12, -4.12, -4.12, -4.21, -4.04, -4.11, -4.15, -4.17, -4.18,
          -4.16, -4.15, -4.15, -4.13, -4.13, -4.13, -4.16, -4.19, -4.2, -4.2, -4.21, -4.25, -4.25, -4.24, -4.57, -4.04, -4.14, -4.1, -4.2, -4.27, -4.3, -4.27, -4.23, -4.25, -4.26,
           -4.24, -4.25, -4.35, -4.57, -4.53, -4.27, -4.28, -4.27, -4.23, -4.26, -4.33, -4.38, -4.54, -4.29, -4.31, -4.31, -4.31, -4.35, -4.32, -4.38, -4.36, -4.37, -4.34, -4.32,
            -4.34, -4.32, -4.45, -4.4, -4.39, -4.38, -4.38, -4.38, -4.37, -4.5, -4.46, -4.42]
    ys3 = [13.26, 13.48, 13.24, 12.35, 12.53, 12.8, 13.01, 13.1, 12.01, 13.76, 11.51, 11.67, 11.76, 11.89, 12.18, 12.8, 12.91, 13.07, 13.26, 13.5, 13.64, 13.69, 13.87, 11.03,
     11.07, 11.2, 11.3, 11.43, 11.73, 11.85, 11.97, 12.09, 12.23, 12.39, 12.49, 12.6, 12.71, 12.82, 12.97, 13.12, 13.4, 13.61, 13.84, 13.97, 12.02, 12.13, 12.23, 12.3, 12.35, 12.55,
      12.66, 12.82, 12.93, 13.08, 13.19, 13.25, 13.44, 13.76, 13.97, 14.12, 11.06, 11.17, 11.28, 11.41, 11.47, 11.59, 11.66, 11.78, 11.9, 13.83, 11.97, 12.09, 12.22, 12.36, 12.48,
       12.65, 12.79, 12.89, 13.01, 13.09, 13.31, 13.45, 13.69, 13.85, 11.02, 11.11, 11.3, 11.42, 11.54, 11.69, 11.87, 12.0, 14.11, 11.75, 11.86, 12.15, 12.31, 12.43, 12.57, 12.7,
        12.82, 12.97, 13.15, 13.28, 13.45, 13.62, 13.82, 11.06, 11.19, 11.33, 11.43, 11.54, 11.62, 11.68, 11.81, 11.95, 12.05, 13.93, 14.26, 13.16, 13.36, 13.59, 13.77, 13.99, 11.3,
         11.4, 11.47, 11.61, 11.84, 11.96, 12.06, 12.18, 12.28, 12.41, 12.54, 12.6, 12.76, 12.9, 13.03, 13.54, 13.72, 11.31, 11.4, 13.44, 11.31, 12.14, 12.38, 12.5, 12.61, 12.73,
          12.84, 13.0, 13.14, 13.31, 13.44, 13.65, 13.8, 14.03, 11.29, 13.17, 14.28, 12.26, 11.39, 11.34, 11.38, 13.15, 11.34, 11.49, 12.3, 12.48, 13.15, 13.41, 13.6, 13.82, 12.27,
           12.34, 12.36, 12.33, 12.41, 12.49, 13.17, 13.41, 11.57, 12.31, 12.27, 12.29, 12.73, 13.19, 13.35, 11.63, 11.74, 11.9, 12.1, 12.39, 12.52, 12.61, 12.95, 13.05, 11.83,
            12.41, 12.52, 12.65, 12.78, 12.92, 13.07, 12.15, 12.22, 12.34]
    
    # 测试数据4
    # 场景：目标为车辆，点云数量为224，有遮挡
    xs4 = [-4.07, -3.78, -3.87, -3.85, -3.73, -3.69, -3.83, -3.82, -3.75, -3.92, -3.87, -3.85, -3.84, -4.0, -3.99, -4.0, -3.99, -3.95, -3.95, -3.95, -3.94, -3.75, -3.73, -3.82, -3.81, -3.79, -3.77,
     -3.77, -3.77, -3.76, -3.92, -3.9, -3.89, -4.0, -4.02, -4.02, -4.0, -3.99, -3.99, -3.96, -3.94, -3.94, -4.09, -4.04, -4.17, -4.14, -4.25, -4.36, -4.32, -4.47, -4.57, -4.64, -4.76, -4.76, -4.86,
      -4.98, -3.75, -3.82, -3.81, -3.79, -3.78, -3.77, -3.76, -3.92, -3.9, -3.89, -3.88, -3.86, -3.85, -3.85, -4.01, -4.01, -4.0, -3.97, -4.16, -4.23, -4.29, -4.37, -4.47, -4.45, -4.56, -4.58, -4.64,
       -4.67, -4.67, -4.78, -4.8, -4.78, -4.85, -4.96, -5.04, -5.06, -3.75, -3.78, -3.8, -3.91, -3.89, -3.99, -3.96, -3.98, -3.96, -4.01, -3.94, -3.93, -3.93, -4.17, -4.12, -4.29, -4.23, -4.4, -4.35,
        -4.47, -4.56, -4.65, -4.72, -4.76, -4.87, -4.83, -5.0, -4.95, -5.1, -3.74, -4.01, -3.99, -3.94, -4.19, -4.13, -4.28, -4.41, -4.37, -4.46, -4.54, -4.65, -4.77, -4.89, -4.96, -5.06, -3.79, -3.79,
         -3.87, -3.86, -4.92, -3.82, -3.91, -3.85, -3.89, -3.88, -3.86, -3.87, -3.95, -4.06, -4.2, -4.29, -4.4, -4.37, -4.51, -4.46, -4.58, -4.69, -4.77, -4.86, -4.97, -4.97, -5.07, -5.1, -5.15, -5.2,
          -5.28, -5.27, -5.41, -5.48, -5.46, -5.53, -5.63, -3.93, -3.91, -3.93, -3.92, -3.94, -3.95, -4.1, -4.13, -4.1, -4.37, -5.08, -5.04, -5.42, -5.52, -5.48, -5.46, -5.46, -5.57, -3.93, -4.02,
           -3.99, -4.16, -4.36, -5.0, -5.08, -5.41, -5.37, -5.5, -4.18, -4.14, -5.06, -5.16, -5.24, -5.29, -5.35, -4.09, -4.08, -4.16, -4.24, -5.21, -5.28, -4.24, -4.31, -4.29, -4.49, -4.63, -4.58,
            -4.89, -4.99, -5.09]
    ys4 = [13.82, 17.08, 15.97, 16.15, 17.02, 17.39, 16.2, 16.68, 16.82, 15.02, 15.64, 15.8, 16.01, 13.78, 13.91, 14.13, 14.28, 14.34, 14.54, 14.72, 14.9, 17.31, 17.53, 15.81, 16.0, 16.18, 16.39,
     16.59, 16.88, 17.1, 15.18, 15.32, 15.5, 13.58, 13.82, 13.98, 14.11, 14.29, 14.47, 14.7, 14.84, 15.01, 13.54, 13.72, 13.5, 13.56, 13.47, 13.4, 13.45, 13.34, 13.34, 13.32, 13.29, 13.34, 13.27,
      13.31, 17.5, 16.03, 16.49, 16.66, 16.78, 17.03, 17.26, 14.44, 14.77, 14.93, 15.11, 15.24, 15.46, 15.88, 13.9, 14.08, 14.23, 14.41, 13.46, 13.43, 13.45, 13.39, 13.3, 13.38, 13.31, 13.38, 13.28,
       13.35, 13.49, 13.24, 13.4, 13.48, 13.4, 13.46, 13.34, 13.53, 17.48, 17.07, 17.41, 14.98, 15.14, 13.91, 14.0, 14.16, 14.34, 14.36, 14.47, 14.66, 14.85, 13.64, 13.66, 13.52, 13.59, 13.4, 13.46,
        13.3, 13.38, 13.51, 13.55, 13.52, 13.38, 13.5, 13.43, 13.46, 13.56, 17.31, 13.97, 14.61, 14.82, 13.61, 13.68, 13.55, 13.4, 13.51, 13.32, 13.33, 13.39, 13.36, 13.42, 13.38, 13.45, 17.42,
         17.73, 15.52, 15.72, 13.72, 16.68, 15.08, 15.15, 17.01, 17.23, 17.44, 17.74, 15.07, 14.54, 14.15, 14.12, 14.02, 14.1, 14.02, 14.05, 14.01, 14.0, 14.01, 14.0, 14.02, 15.89, 14.05, 15.9,
          14.07, 15.94, 14.1, 15.94, 16.01, 15.07, 15.95, 14.95, 15.08, 16.66, 16.82, 17.02, 17.57, 16.2, 16.45, 14.8, 14.9, 14.98, 15.54, 15.82, 16.0, 16.01, 15.09, 15.15, 15.24, 15.95, 15.06,
           16.77, 15.89, 17.2, 15.09, 15.63, 15.87, 15.8, 15.27, 15.96, 15.25, 15.31, 15.38, 15.51, 15.63, 15.65, 15.55, 15.45, 16.56, 16.75, 15.67, 15.49, 15.56, 15.62, 16.16, 15.98, 16.1, 15.87,
            15.83, 15.86, 15.88, 15.92, 15.95]
    
    # 测试数据5
    # 场景：目标为车辆，点云数量为194，有遮挡
    xs5 = [-3.68, -3.56, -3.58, -3.56, -3.63, -3.65, -3.62, -3.59, -3.6, -3.83, -3.8, -3.91, -3.63, -3.63, -3.61, -3.6, -3.6, -3.59, -3.58, -3.74, -3.7, -3.66, -3.8, -3.76,
     -3.89, -3.85, -3.56, -3.54, -3.66, -3.63, -3.63, -3.61, -3.6, -3.74, -3.71, -3.67, -3.82, -3.9, -4.01, -3.56, -3.57, -3.56, -3.54, -3.6, -3.6, -3.58, -3.73, -3.71, -3.67,
      -3.68, -3.8, -3.89, -4.01, -3.56, -3.65, -3.66, -3.6, -3.58, -3.57, -3.57, -3.75, -3.72, -3.69, -3.79, -3.87, -3.99, -4.12, -3.58, -3.63, -3.63, -3.62, -3.62, -3.61, -3.59,
       -3.6, -3.58, -3.75, -3.71, -3.68, -3.66, -3.83, -3.78, -3.91, -3.99, -4.09, -3.57, -3.57, -3.57, -3.65, -3.63, -3.58, -3.74, -3.7, -3.68, -3.67, -3.79, -3.76, -3.92, -4.0,
        -3.96, -4.09, -4.19, -4.28, -3.65, -3.62, -3.61, -3.6, -3.58, -3.66, -3.75, -3.72, -3.69, -3.66, -3.74, -3.71, -3.82, -3.77, -3.89, -4.01, -4.09, -4.19, -4.31, -4.41,
         -4.49, -3.75, -3.74, -3.71, -3.7, -3.82, -3.79, -3.77, -3.9, -3.85, -3.99, -4.14, -4.07, -4.2, -4.29, -4.42, -4.52, -4.61, -3.75, -3.86, -3.83, -3.81, -3.8, -3.78, -3.9,
          -4.04, -4.12, -4.22, -4.3, -4.44, -4.54, -4.48, -4.61, -4.73, -4.67, -4.79, -4.86, -3.85, -3.83, -3.84, -3.82, -3.94, -3.91, -3.88, -3.87, -4.21, -4.3, -4.4, -4.49, -4.6,
           -4.71, -4.78, -5.04, -4.99, -3.96, -3.94, -3.92, -3.91, -4.05, -3.99, -4.13, -4.22, -4.32, -4.42, -4.49, -4.56, -4.63, -4.71, -4.79, -4.87, -4.96]
    ys5 = [17.86, 20.36, 20.27, 20.62, 17.81, 18.2, 18.77, 19.55, 19.96, 17.25, 17.41, 17.17, 18.32, 18.61, 18.88, 19.21, 19.96, 20.34, 20.73, 17.35, 17.49, 17.97, 17.07, 17.21,
     16.92, 17.03, 20.37, 20.68, 18.43, 18.66, 18.98, 19.26, 19.58, 17.25, 17.4, 17.84, 17.06, 16.99, 16.92, 19.59, 19.98, 20.6, 20.88, 18.68, 18.97, 19.27, 17.25, 17.4, 17.87,
      18.21, 17.09, 16.96, 16.91, 20.17, 17.93, 18.24, 18.85, 19.14, 19.49, 19.84, 17.15, 17.27, 17.44, 17.06, 16.99, 16.95, 16.97, 20.71, 17.97, 18.32, 18.57, 18.95, 19.23, 19.57,
       19.94, 20.27, 17.31, 17.45, 17.63, 17.77, 17.1, 17.2, 17.01, 17.0, 16.97, 19.62, 20.0, 20.39, 18.39, 18.64, 19.3, 17.54, 17.7, 17.89, 18.17, 17.22, 17.36, 17.18, 17.01,
        17.11, 17.01, 17.01, 16.99, 18.2, 18.43, 18.71, 18.98, 19.9, 19.96, 17.41, 17.61, 17.78, 17.98, 19.59, 19.85, 17.18, 17.27, 17.09, 17.03, 17.01, 17.0, 16.99, 16.98, 16.97,
         18.26, 18.53, 18.76, 18.99, 17.65, 17.83, 18.02, 17.44, 17.53, 17.4, 17.34, 17.35, 17.32, 17.3, 17.3, 17.32, 17.31, 18.93, 17.82, 17.99, 18.23, 18.51, 18.71, 17.7, 17.71,
          17.67, 17.67, 17.62, 17.76, 17.65, 17.68, 17.66, 17.61, 17.65, 17.58, 17.61, 18.73, 18.94, 19.36, 19.65, 17.89, 18.05, 18.2, 18.51, 17.79, 17.76, 17.73, 17.74, 17.74,
           17.82, 17.79, 18.02, 18.08, 18.45, 18.67, 18.93, 19.21, 18.24, 18.28, 18.16, 18.12, 18.11, 18.11, 18.12, 18.12, 18.16, 18.17, 18.23, 18.28, 18.36]
    
    # 测试数据6
    # 场景：目标为车辆，点云数量为107，有遮挡
    xs6 = [2.3, 2.6, 2.51, 2.45, 2.39, 2.34, 2.33, 2.27, 2.24, 2.26, 2.23, 2.59, 2.5, 2.44, 2.38, 2.33, 2.27, 2.27, 2.3, 2.22, 2.67, 2.59, 2.5, 2.45, 2.39, 2.32, 2.31, 2.26, 2.27, 2.29, 2.68, 2.59, 2.5,
     2.41, 2.36, 2.3, 2.25, 2.27, 2.3, 2.7, 2.6, 2.54, 2.49, 2.42, 2.37, 2.35, 2.38, 2.32, 2.27, 2.27, 2.79, 2.78, 2.73, 2.67, 2.58, 2.48, 2.44, 2.4, 2.36, 2.34, 2.94, 2.87, 2.8, 2.76, 2.7, 2.56, 2.44,
      2.5, 2.33, 2.42, 2.26, 2.93, 2.86, 2.79, 2.71, 2.49, 2.46, 3.16, 3.09, 3.01, 2.85, 2.77, 2.59, 2.54, 2.47, 2.48, 3.29, 3.19, 3.08, 2.98, 2.88, 2.78, 2.67, 2.61, 2.55, 2.58, 3.46, 3.38, 3.29, 3.23,
       3.16, 3.06, 2.96, 2.88, 2.78, 2.71, 2.69]
    ys6 = [20.38, 17.63, 17.67, 17.7, 17.73, 20.39, 17.78, 17.83, 18.6, 19.4, 18.03, 17.64, 17.66, 17.69, 17.76, 17.79, 17.83, 18.68, 19.46, 17.95, 17.73, 17.85, 17.93, 18.01, 18.03, 20.53, 17.87, 18.01,
     18.86, 19.55, 17.76, 17.79, 17.85, 17.89, 17.95, 18.03, 18.19, 19.49, 20.39, 17.91, 17.96, 17.95, 18.03, 18.05, 18.13, 19.84, 20.8, 18.2, 18.4, 18.9, 18.14, 20.15, 18.18, 18.22, 18.25, 18.28, 18.41,
      18.59, 18.85, 19.28, 18.79, 18.81, 18.83, 18.97, 19.0, 18.99, 19.14, 20.77, 19.36, 19.58, 19.36, 20.21, 20.2, 20.22, 20.2, 19.3, 19.89, 19.4, 19.43, 19.39, 20.17, 20.11, 19.36, 19.43, 19.45, 20.07,
       19.56, 19.55, 19.54, 19.56, 19.61, 19.59, 19.62, 19.66, 19.74, 20.53, 19.9, 19.86, 19.81, 19.83, 19.82, 19.87, 19.96, 20.08, 20.14, 20.18, 20.52]
    
    # 测试数据7
    # 场景：目标为车辆，点云数量为80，有遮挡
    xs7 = [5.06, 4.99, 4.98, 4.94, 4.89, 4.9, 4.88, 5.29, 5.14, 4.97, 4.97, 4.95, 4.96, 5.44, 5.35, 4.98, 4.97, 4.98, 4.94, 5.6, 5.51, 5.42, 5.02, 4.98, 4.96, 4.94, 4.95, 4.94,
     4.98, 4.98, 5.01, 4.96, 5.75, 5.66, 5.53, 5.4, 5.05, 5.01, 4.99, 5.92, 5.84, 5.74, 5.65, 5.57, 5.43, 5.09, 5.05, 5.03, 5.04, 5.02, 5.06, 5.13, 6.23, 6.14, 6.04, 5.94, 5.84,
      5.76, 5.66, 5.53, 5.4, 5.32, 5.22, 5.14, 5.14, 6.44, 6.31, 6.21, 6.11, 6.01, 5.85, 5.8, 5.71, 5.61, 5.54, 5.45, 5.37, 5.31, 5.26, 5.22]
    ys7 = [24.49, 24.54, 27.77, 24.76, 24.99, 25.46, 25.9, 24.35, 24.42, 27.21, 25.7, 26.03, 26.62, 24.41, 24.43, 27.4, 27.97, 24.78, 25.03, 24.45, 24.45, 24.45, 24.69, 24.95,
     25.26, 25.68, 26.24, 26.71, 25.88, 26.44, 27.62, 26.84, 24.37, 24.39, 24.41, 24.43, 24.85, 25.12, 25.51, 24.74, 24.74, 24.72, 24.75, 24.76, 24.79, 25.08, 25.34, 25.76, 26.25,
      26.67, 27.4, 26.39, 25.4, 25.42, 25.35, 25.36, 25.35, 25.37, 25.36, 25.37, 25.42, 25.5, 25.45, 25.54, 26.02, 26.05, 25.95, 25.94, 25.9, 25.9, 25.64, 25.8, 25.8, 25.83, 25.88,
       25.93, 26.03, 26.13, 26.39, 26.73]
    
    # 测试数据8
    # 场景：目标为车辆，点云数量为621，无遮挡，L型
    xs8 = [0.97, 0.94, 0.84, 0.89, 0.73, 0.79, 0.67, 2.86, 2.91, 2.79, 2.86, 2.72, 2.73, 2.65, 2.55, 2.39, 2.46, 2.33, 2.21, 2.27, 1.11, 1.16, 1.02, 1.06, 0.95, 0.84, 0.88, 0.73,
     0.79, 0.66, 0.02, -0.04, -0.1, -0.16, -0.24, -0.31, -0.37, -0.45, -0.54, -0.61, -0.65, 2.89, 2.97, 2.82, 2.7, 2.77, 2.65, 2.62, 2.52, 2.39, 2.47, 2.34, 2.21, 2.27, 2.12, 2.16,
      2.04, 1.95, 1.82, 1.86, 1.7, 1.76, 1.61, 1.66, 1.51, 1.55, 1.46, 0.62, 0.56, 0.46, 0.36, 0.26, 0.14, 0.06, 2.92, 2.78, 2.85, 2.72, 2.62, 2.54, 2.42, 2.34, 2.23, 2.14, 2.05,
       1.92, 1.97, 1.84, 1.74, 1.63, 1.53, 1.4, 1.45, 1.3, 1.32, 1.35, 1.2, 1.25, 1.12, 1.17, 1.02, 1.06, 1.08, 0.95, 0.83, 0.88, 0.7, 0.79, 0.76, -0.04, -0.14, -0.24, -0.32, -0.38,
        -0.45, -0.54, -0.6, -0.65, -0.7, 2.97, 2.91, 2.78, 2.72, 2.63, 2.51, 2.51, 2.42, 2.42, 2.34, 2.21, 2.27, 2.28, 2.13, 2.03, 1.92, 1.98, 1.82, 1.83, 1.7, 1.75, 1.65, 1.64, 1.69,
         1.56, 1.46, 1.33, 1.38, 1.21, 1.28, 1.16, 1.12, 1.02, 1.07, 1.02, 0.94, 0.95, 0.86, 0.84, 0.89, 0.77, 0.63, 0.66, 0.56, 0.46, 0.36, 0.26, 0.15, 0.05, -0.04, -0.1, -0.16,
          -0.23, -0.26, -0.31, -0.37, -0.47, -0.54, -0.56, -0.61, -0.64, -0.69, 3.01, 3.04, 2.94, 2.84, 2.77, 2.49, 2.55, 2.43, 2.34, 2.2, 2.23, 2.12, 2.16, 2.04, 1.93, 1.99, 1.84,
           1.71, 1.76, 1.6, 1.65, 1.51, 1.55, 1.39, 1.42, 1.45, 1.3, 1.34, 1.35, 1.25, 1.13, 1.19, 1.02, 1.06, 0.92, 0.97, 0.85, 0.73, 0.79, 0.64, 0.64, 0.56, 0.55, 0.46, 0.45, 0.36,
            0.34, 0.26, 0.21, 0.26, 0.12, 0.14, 0.06, 0.02, -0.02, -0.05, -0.15, -0.16, -0.25, -0.33, -0.34, -0.39, -0.41, -0.43, -0.48, -0.53, -0.57, 3.07, 2.9, 2.93, 2.83, 2.82,
             2.68, 2.76, 2.61, 2.49, 2.54, 2.43, 2.34, 2.19, 2.25, 2.13, 2.01, 2.07, 1.9, 1.94, 1.8, 1.84, 1.74, 1.63, 1.68, 1.55, 1.43, 1.48, 1.32, 1.38, 1.2, 1.23, 1.27, 1.1, 1.15,
              1.19, 1.01, 1.05, 1.09, 0.92, 0.97, 0.83, 0.88, 0.7, 0.77, 0.64, 0.54, 0.42, 0.34, 0.22, 0.26, 0.15, 0.06, 0.02, -0.04, -0.14, -0.23, -0.27, -0.31, -0.37, -0.35, -0.44,
               -0.48, -0.44, -0.55, -0.51, -0.55, 2.89, 2.97, 2.83, 2.72, 2.77, 2.64, 2.63, 2.53, 2.57, 2.43, 2.34, 2.25, 2.1, 2.16, 2.04, 1.93, 1.99, 1.79, 1.82, 1.87, 1.74, 1.65,
                1.56, 1.46, 1.36, 1.26, 1.17, 1.01, 1.07, 0.92, 0.95, 0.84, 0.84, 0.73, 0.76, 0.64, 0.67, 0.52, 0.55, 0.45, 0.47, 0.41, 0.37, 0.33, 0.35, 0.28, 0.23, 0.27, 0.21, 0.15,
                 0.14, 0.04, 0.04, 0.08, -0.04, -0.06, -0.14, -0.15, -0.2, -0.27, -0.21, -0.26, -0.3, -0.25, -0.29, -0.34, -0.39, -0.33, -0.45, -0.52, 3.07, 2.92, 2.78, 2.85, 2.71,
                  2.64, 2.51, 2.58, 2.39, 2.45, 2.32, 2.33, 2.21, 2.25, 2.14, 2.05, 1.91, 1.97, 1.81, 1.86, 1.7, 1.75, 1.59, 1.65, 1.5, 1.53, 1.4, 1.44, 1.49, 1.31, 1.36, 1.22, 1.27,
                   1.14, 1.13, 1.05, 1.05, 0.96, 0.99, 0.92, 0.85, 0.84, 0.78, 0.74, 0.76, 0.64, 0.69, 0.63, 0.63, 0.54, 0.54, 0.48, 0.42, 0.46, 0.4, 0.33, 0.33, 0.25, 0.25, 0.15,
                    0.18, 0.14, 0.09, 0.06, 0.02, -0.0, -0.02, -0.07, -0.13, -0.2, 3.02, 2.88, 2.96, 2.82, 2.72, 2.61, 2.49, 2.56, 2.42, 2.3, 2.37, 2.21, 2.1, 2.15, 2.17, 2.03, 1.92,
                     1.94, 1.82, 1.85, 1.73, 1.52, 1.44, 1.33, 1.23, 1.18, 1.16, 1.13, 1.11, 1.02, 1.09, 1.04, 1.0, 0.95, 0.82, 0.8, 0.86, 0.77, 0.76, 0.64, 0.66, 0.41, 0.34, 0.27,
                      0.21, 0.26, 0.16, 0.12, 0.05, 0.03, 2.89, 2.93, 2.83, 2.76, 2.48, 2.55, 2.43, 2.36, 1.89, 1.84, 1.61, 1.55, 1.57, 1.44, 1.33, 1.36, 1.23, 1.27, 1.25, 1.19, 1.17,
                       1.03, 0.83, 0.81, 0.74, 0.35, 0.24, 2.84, 2.79, 2.74, 2.72, 2.61, 2.65, 2.49, 2.43, 2.32, 2.38, 1.89, 1.84, 1.32, 1.37, 1.27, 1.16, 0.76, 0.42, 0.47, 0.34,
                        2.72, 2.58, 2.65, 2.51, 2.39, 2.38, 2.45, 2.31, 2.19, 2.25, 2.13, 2.04, 1.9, 1.95, 1.82, 1.42, 1.35, 1.28, 0.64, 0.6, 0.64, 0.59, 0.54, 0.55, 0.47, 0.42,
                         2.52, 2.43, 2.29, 2.36, 2.23, 2.1, 2.17, 2.02, 1.93, 1.81, 1.86, 1.72, 1.77, 1.63, 1.51, 1.57, 1.55, 1.51, 1.44, 1.47, 1.41, 1.35, 1.34, 1.3, 1.28, 1.22,
                          1.26, 1.21, 1.13, 1.18, 1.14, 1.1, 1.04, 1.06, 1.02, 0.98, 0.91, 0.94, 0.83, 0.82, 0.87, 0.74, 0.68, 1.89]
    ys8 = [11.72, 11.83, 11.58, 11.67, 11.47, 11.54, 11.47, 14.0, 14.0, 13.93, 14.01, 13.79, 13.86, 13.72, 13.57, 13.38, 13.47, 13.28, 13.14, 13.21, 11.88, 11.94, 11.77, 11.84,
     11.73, 11.58, 11.63, 11.45, 11.51, 11.46, 11.58, 11.61, 11.65, 11.71, 11.75, 11.81, 11.88, 11.82, 11.89, 11.94, 12.02, 14.01, 14.14, 13.94, 13.82, 13.92, 13.77, 13.82, 13.57,
      13.4, 13.53, 13.35, 13.19, 13.26, 13.05, 13.12, 12.96, 12.83, 12.67, 12.72, 12.55, 12.61, 12.45, 12.49, 12.36, 12.4, 12.32, 11.35, 11.32, 11.31, 11.32, 11.32, 11.34, 11.37,
       14.02, 13.87, 13.94, 13.78, 13.69, 13.62, 13.41, 13.35, 13.23, 13.14, 13.04, 12.84, 12.91, 12.77, 12.65, 12.55, 12.43, 12.25, 12.33, 12.08, 12.14, 12.22, 11.97, 12.05, 11.99,
        12.01, 11.77, 11.8, 12.05, 11.71, 11.55, 11.62, 11.43, 11.57, 11.58, 11.45, 11.5, 11.57, 11.65, 11.71, 11.76, 11.83, 11.91, 12.01, 12.16, 14.09, 14.13, 13.9, 13.82, 13.75,
         13.47, 13.51, 13.38, 13.43, 13.34, 13.18, 13.24, 13.3, 13.05, 12.93, 12.79, 12.87, 12.68, 12.72, 12.54, 12.63, 12.48, 12.52, 12.6, 12.44, 12.32, 12.16, 12.21, 12.0, 12.1,
          11.95, 12.01, 11.88, 11.96, 12.02, 11.72, 12.03, 11.64, 11.89, 12.05, 11.63, 11.36, 11.4, 11.33, 11.33, 11.33, 11.33, 11.33, 11.36, 11.44, 11.46, 11.53, 11.57, 11.59, 11.64,
           11.71, 11.79, 11.87, 11.92, 11.94, 12.01, 12.19, 14.12, 14.26, 14.07, 14.12, 14.16, 13.49, 13.53, 13.43, 13.35, 13.18, 13.23, 13.08, 13.14, 12.99, 12.85, 12.93, 12.76,
            12.59, 12.64, 12.44, 12.53, 12.33, 12.43, 12.18, 12.24, 12.3, 12.05, 12.14, 12.2, 12.01, 11.87, 11.91, 11.78, 11.81, 11.65, 11.7, 11.55, 11.44, 11.51, 11.38, 11.41,
             11.38, 11.41, 11.36, 11.51, 11.36, 11.52, 11.37, 11.44, 11.56, 11.38, 11.43, 11.44, 11.52, 11.44, 11.52, 11.54, 11.6, 11.64, 11.68, 11.72, 11.82, 11.75, 11.84, 11.92,
              11.97, 12.02, 14.27, 13.96, 14.06, 13.89, 13.96, 13.76, 13.86, 13.65, 13.5, 13.56, 13.46, 13.36, 13.19, 13.27, 13.12, 12.95, 13.04, 12.8, 12.87, 12.69, 12.75, 12.62,
               12.5, 12.56, 12.42, 12.28, 12.34, 12.16, 12.23, 11.99, 12.06, 12.12, 11.88, 11.94, 12.02, 11.78, 11.82, 11.9, 11.68, 11.73, 11.59, 11.62, 11.49, 11.53, 11.46, 11.52,
                11.54, 11.53, 11.45, 11.5, 11.43, 11.48, 11.49, 11.53, 11.61, 11.64, 11.68, 11.74, 11.8, 11.95, 11.88, 11.92, 12.03, 12.03, 12.12, 12.2, 14.03, 14.16, 13.96, 13.77,
                 13.89, 13.66, 13.73, 13.57, 13.65, 13.45, 13.35, 13.26, 13.08, 13.16, 13.01, 12.89, 12.94, 12.69, 12.77, 12.83, 12.66, 12.56, 12.46, 12.36, 12.25, 12.15, 12.04,
                  11.84, 11.94, 11.75, 11.83, 11.67, 11.74, 11.57, 11.65, 11.54, 11.61, 11.47, 11.54, 11.48, 11.58, 11.62, 11.48, 11.49, 11.65, 11.48, 11.54, 11.68, 11.72, 11.57,
                   11.75, 11.63, 11.78, 11.81, 11.72, 11.83, 11.72, 11.9, 11.76, 11.83, 11.95, 12.04, 12.1, 12.55, 12.65, 12.13, 12.21, 12.64, 12.32, 12.33, 14.4, 14.19, 14.0, 14.08,
                    13.9, 13.79, 13.63, 13.74, 13.48, 13.55, 13.39, 13.45, 13.25, 13.34, 13.16, 13.05, 12.9, 12.97, 12.8, 12.83, 12.68, 12.73, 12.53, 12.62, 12.39, 12.45, 12.28,
                     12.34, 12.4, 12.18, 12.26, 12.15, 12.23, 12.03, 12.15, 11.96, 12.18, 11.88, 12.18, 12.2, 11.83, 12.22, 13.98, 11.8, 12.25, 11.83, 12.27, 12.31, 13.86, 11.85,
                      12.29, 11.87, 11.91, 12.34, 12.39, 11.97, 12.44, 11.98, 12.53, 12.02, 12.65, 12.77, 12.9, 12.08, 12.13, 13.17, 12.18, 12.22, 12.35, 12.46, 14.32, 14.14, 14.24,
                       14.08, 13.95, 13.81, 13.66, 13.75, 13.58, 13.45, 13.52, 13.34, 13.16, 13.25, 14.17, 13.12, 12.99, 13.02, 12.88, 12.93, 12.78, 12.21, 12.18, 12.24, 12.24, 13.73,
                        12.18, 12.39, 13.75, 12.39, 12.4, 13.79, 13.87, 12.32, 12.93, 13.96, 14.03, 12.89, 13.94, 12.96, 13.02, 12.86, 12.86, 12.75, 13.28, 13.31, 13.11, 13.41, 13.25,
                         13.43, 14.27, 14.38, 14.2, 14.13, 13.75, 13.83, 13.72, 13.62, 13.06, 12.99, 12.25, 12.22, 13.11, 12.24, 12.35, 12.44, 12.29, 12.31, 12.41, 12.31, 13.78,
                          13.88, 12.93, 13.93, 13.96, 13.36, 13.35, 14.31, 14.32, 14.18, 14.24, 14.05, 14.17, 13.83, 13.76, 13.69, 13.72, 13.12, 13.07, 12.48, 12.51, 12.48, 13.79,
                           13.94, 13.34, 13.47, 13.37, 14.35, 14.14, 14.22, 14.05, 13.79, 13.88, 13.96, 13.79, 13.61, 13.69, 13.55, 13.44, 13.28, 13.34, 13.06, 12.64, 12.57, 12.61,
                            13.18, 13.3, 13.45, 13.56, 13.34, 13.6, 13.39, 13.42, 14.15, 14.03, 13.81, 13.93, 13.73, 13.58, 13.65, 13.47, 13.35, 13.21, 13.25, 13.1, 13.15, 12.98,
                             12.8, 12.86, 12.93, 13.04, 12.77, 13.06, 13.14, 12.78, 13.19, 13.24, 12.8, 12.82, 13.28, 13.33, 12.87, 13.41, 13.55, 13.61, 12.93, 13.66, 13.78, 12.99,
                              13.03, 13.88, 13.14, 13.8, 13.84, 13.22, 13.33, 13.44]
    
    # 测试数据9
    # 场景：目标为车辆，点云数量为374，无遮挡，L型/O型
    xs9 = [2.39, 2.34, 2.28, 2.32, 2.3, 2.33, 2.33, 2.32, 2.32, 2.37, 2.3, 2.3, 2.3, 2.3, 2.31, 2.31, 2.36, 3.56, 3.45, 3.36, 3.33, 3.23, 3.29, 3.15, 3.15, 3.04, 2.94, 2.98, 2.85, 2.76, 2.73, 2.65, 2.55,
     2.5, 2.44, 2.38, 2.34, 2.32, 2.3, 2.31, 2.33, 2.33, 2.29, 2.28, 2.28, 2.29, 2.27, 2.27, 2.27, 3.67, 3.54, 3.45, 3.43, 3.35, 3.34, 3.25, 3.12, 3.06, 3.04, 2.95, 2.81, 2.86, 2.76, 2.66, 2.63, 2.57, 2.54,
      2.56, 2.5, 2.46, 2.45, 2.38, 2.35, 2.32, 2.31, 2.3, 2.3, 2.31, 2.3, 2.36, 2.28, 2.26, 3.66, 3.52, 3.58, 3.46, 3.35, 3.23, 3.15, 3.05, 2.96, 2.92, 2.86, 2.76, 2.68, 2.66, 2.64, 2.57, 2.53, 2.44, 2.37,
       2.34, 2.31, 2.28, 2.3, 2.3, 2.28, 2.28, 2.27, 2.28, 2.28, 2.28, 3.65, 3.65, 3.57, 3.55, 3.44, 3.43, 3.48, 3.33, 3.38, 3.35, 3.24, 3.25, 3.26, 3.16, 3.12, 3.17, 3.05, 3.06, 2.95, 2.95, 2.87, 2.85, 2.76,
        2.75, 2.65, 2.67, 2.63, 2.54, 2.56, 2.47, 2.43, 2.44, 2.38, 2.34, 2.35, 2.35, 2.33, 2.31, 2.3, 2.29, 2.28, 2.28, 2.32, 2.34, 2.28, 2.28, 2.27, 2.28, 3.85, 3.76, 3.73, 3.79, 3.65, 3.65, 3.55, 3.56,
         3.49, 3.45, 3.44, 3.39, 3.37, 3.38, 3.33, 3.33, 3.3, 3.25, 3.14, 3.05, 2.95, 2.87, 2.83, 2.74, 2.76, 2.64, 2.56, 2.51, 2.47, 2.43, 2.43, 2.39, 2.36, 2.34, 2.35, 2.34, 2.36, 2.34, 2.33, 2.32, 2.32,
          2.31, 2.34, 2.34, 4.01, 4.07, 3.95, 3.85, 3.86, 3.82, 3.8, 3.7, 3.76, 3.64, 3.63, 3.57, 3.55, 3.52, 3.51, 3.46, 3.4, 3.34, 3.39, 3.32, 3.28, 3.27, 3.2, 3.23, 3.14, 3.05, 3.04, 2.95, 2.92, 2.84,
           2.82, 2.84, 2.79, 2.78, 2.74, 2.73, 2.73, 2.65, 2.66, 2.63, 2.57, 2.52, 2.45, 2.43, 2.45, 2.38, 2.32, 2.36, 2.35, 2.25, 2.21, 3.9, 3.85, 3.83, 3.83, 3.78, 3.76, 3.77, 3.74, 3.72, 3.65, 3.6, 3.57,
            3.55, 3.51, 3.39, 3.4, 3.45, 3.42, 3.32, 3.31, 3.3, 3.21, 3.22, 3.02, 2.95, 2.6, 2.51, 2.54, 2.49, 2.45, 2.45, 2.47, 2.46, 2.47, 2.46, 2.46, 2.46, 2.33, 2.24, 3.79, 3.81, 3.74, 3.76, 3.62, 3.52,
             3.44, 3.34, 3.18, 2.93, 2.86, 2.81, 2.78, 2.64, 2.52, 2.53, 2.51, 2.5, 2.52, 2.5, 2.53, 2.49, 3.77, 3.71, 3.71, 3.72, 3.64, 3.58, 3.51, 3.33, 3.22, 3.12, 3.08, 2.9, 2.84, 2.85, 2.77, 2.74, 2.63,
              2.56, 2.54, 2.53, 2.53, 2.53, 2.56, 2.53, 2.55, 3.71, 3.6, 3.65, 3.63, 3.51, 3.51, 3.56, 3.42, 3.42, 3.32, 3.31, 3.22, 3.23, 3.12, 3.17, 3.14, 3.03, 3.02, 3.03, 2.93, 2.92, 2.83, 2.85, 2.81,
               2.74, 2.74, 2.63, 2.68, 2.64, 2.63, 2.56, 2.56, 2.56]
    ys9 = [14.49, 14.54, 12.04, 11.75, 11.88, 14.04, 14.49, 14.75, 14.89, 14.74, 12.35, 13.1, 13.41, 13.71, 13.83, 14.52, 14.99, 11.32, 11.29, 11.3, 11.33, 11.29, 11.32, 11.3, 11.31, 11.29, 11.3, 11.31, 11.3,
     11.3, 11.33, 11.33, 11.37, 11.41, 11.44, 11.55, 11.68, 11.76, 12.12, 13.53, 13.85, 14.88, 11.86, 12.03, 12.23, 12.56, 12.74, 12.8, 13.04, 11.59, 11.36, 11.36, 11.46, 11.34, 11.44, 11.45, 11.35, 11.32,
      11.47, 11.46, 11.33, 11.46, 11.36, 11.39, 11.62, 11.39, 11.42, 11.67, 11.72, 11.44, 11.75, 11.53, 11.65, 11.75, 12.33, 12.54, 12.87, 13.15, 13.72, 15.12, 11.9, 12.49, 11.56, 11.4, 11.45, 11.4, 11.39,
       11.39, 11.38, 11.39, 11.4, 11.41, 11.41, 11.44, 11.46, 11.56, 11.69, 11.64, 11.75, 11.73, 11.8, 11.87, 11.95, 13.44, 13.78, 14.49, 12.05, 12.25, 12.48, 12.8, 13.0, 13.15, 11.55, 11.67, 11.49, 11.63,
        11.45, 11.61, 11.63, 11.44, 11.61, 11.61, 11.44, 11.6, 11.62, 11.45, 11.61, 11.62, 11.45, 11.61, 11.46, 11.62, 11.47, 11.65, 11.49, 11.68, 11.54, 11.69, 11.72, 11.63, 11.77, 11.68, 11.71, 11.85,
         11.77, 11.89, 11.96, 12.05, 12.16, 12.29, 12.47, 12.68, 13.75, 14.01, 14.13, 14.55, 12.87, 12.91, 13.09, 13.43, 12.17, 11.91, 12.11, 12.16, 11.84, 12.08, 11.8, 12.06, 13.73, 11.78, 13.72, 14.4,
          11.78, 13.71, 13.72, 14.36, 14.58, 11.76, 11.77, 11.77, 11.79, 11.81, 11.82, 11.83, 13.74, 11.87, 11.9, 11.98, 11.98, 12.01, 12.33, 12.04, 12.12, 12.24, 12.36, 12.55, 12.72, 12.86, 13.06, 13.23,
           13.52, 13.76, 14.06, 14.37, 12.89, 12.94, 12.89, 12.68, 13.62, 13.67, 13.75, 12.69, 12.72, 12.64, 14.2, 12.64, 14.18, 14.29, 14.56, 12.03, 14.48, 12.01, 12.03, 14.44, 14.63, 12.01, 12.02, 14.64,
            12.02, 12.04, 14.01, 12.06, 13.79, 12.08, 12.76, 13.79, 13.81, 12.1, 12.12, 12.68, 13.77, 12.14, 12.66, 12.75, 12.18, 12.23, 12.27, 12.7, 12.76, 12.9, 12.93, 14.25, 14.48, 12.97, 13.01, 13.67,
             12.78, 12.86, 13.7, 12.82, 12.89, 13.76, 14.21, 14.35, 14.39, 14.58, 14.64, 14.6, 14.65, 13.0, 14.49, 14.64, 14.85, 14.46, 14.8, 14.85, 14.81, 14.82, 14.69, 14.73, 13.73, 12.87, 12.91, 13.4,
              12.79, 12.87, 13.56, 13.82, 14.12, 14.39, 14.66, 14.97, 12.9, 12.95, 13.0, 13.68, 13.07, 13.77, 13.74, 13.75, 13.75, 14.91, 13.07, 13.76, 13.6, 13.65, 13.73, 13.25, 13.05, 13.47, 13.6, 13.83,
               14.18, 14.41, 14.58, 14.69, 13.1, 13.15, 13.42, 13.67, 13.55, 13.69, 13.74, 13.3, 13.07, 13.07, 13.09, 13.07, 13.11, 13.39, 13.21, 13.31, 13.15, 13.15, 13.26, 13.53, 13.74, 14.03, 14.19,
                14.32, 14.43, 13.26, 13.21, 13.23, 13.79, 13.19, 13.7, 13.73, 13.19, 13.68, 13.19, 13.63, 13.2, 13.62, 13.2, 13.61, 13.64, 13.2, 13.61, 13.64, 13.19, 13.69, 13.19, 13.7, 13.76, 13.23,
                 13.78, 13.25, 13.85, 13.93, 14.12, 13.41, 13.66, 13.89]
    
    # 测试数据10
    # 场景：目标为车辆，点云数量为383，无遮挡，L型
    xs10 = [-2.38, -3.46, -3.53, -3.59, -4.51, -5.02, -5.34, -2.11, -2.22, -2.18, -2.31, -2.28, -2.39, -2.48, -2.61, -2.56, -2.71, -2.66, -2.78, -2.93, -3.0, -3.09, -3.05, -3.17,
     -3.24, -3.3, -3.36, -3.44, -3.51, -3.58, -3.77, -3.86, -4.04, -4.12, -4.22, -4.41, -4.49, -4.58, -4.67, -5.0, -5.33, -2.03, -1.99, -2.13, -2.08, -2.23, -2.18, -2.3, -2.4, -2.53,
      -2.47, -2.6, -2.69, -2.81, -2.76, -2.89, -2.99, -3.1, -3.19, -3.24, -3.31, -3.39, -3.46, -3.52, -3.6, -3.76, -3.84, -4.01, -4.11, -4.21, -4.29, -4.39, -4.49, -4.57, -4.79,
       -4.88, -4.98, -5.1, -5.21, -1.92, -2.0, -1.96, -2.1, -2.05, -2.2, -2.15, -2.27, -2.37, -2.46, -2.59, -2.54, -2.66, -2.79, -2.73, -2.86, -2.97, -3.08, -3.2, -3.29, -3.37,
        -3.44, -3.51, -3.58, -3.67, -3.75, -3.84, -3.98, -4.11, -4.2, -4.29, -4.38, -4.6, -5.02, -5.14, -5.25, -1.99, -2.0, -2.11, -2.06, -2.05, -2.2, -2.15, -2.3, -2.26, -2.38,
         -2.47, -2.57, -2.67, -2.77, -2.87, -2.98, -3.09, -3.17, -3.23, -3.29, -3.37, -3.43, -3.51, -3.58, -3.7, -3.88, -4.0, -4.14, -4.23, -4.31, -4.41, -4.51, -4.59, -4.7, -4.81,
          -4.91, -5.02, -5.13, -5.24, -2.01, -2.1, -2.05, -2.16, -2.24, -2.38, -2.47, -2.6, -2.55, -2.69, -2.64, -2.76, -2.88, -2.99, -3.09, -3.04, -3.17, -3.29, -3.4, -3.48, -3.54,
           -3.62, -3.69, -3.77, -3.86, -3.94, -4.08, -4.22, -4.32, -4.41, -4.51, -4.59, -5.0, -5.12, -5.23, -5.33, -2.01, -2.13, -2.08, -2.2, -2.31, -2.27, -2.37, -2.47, -2.57,
            -2.69, -2.64, -2.77, -2.88, -2.85, -2.99, -2.94, -3.07, -3.18, -3.29, -3.4, -3.47, -3.53, -3.55, -3.61, -3.69, -3.77, -3.84, -3.95, -4.02, -4.12, -4.2, -4.3, -4.39,
             -4.49, -4.54, -4.64, -5.07, -5.18, -2.1, -2.05, -2.2, -2.14, -2.29, -2.24, -2.41, -2.36, -2.48, -2.48, -2.58, -2.72, -2.67, -2.81, -2.77, -2.87, -2.92, -3.0, -3.1,
              -3.16, -3.28, -3.27, -3.38, -3.48, -3.58, -3.67, -3.74, -3.82, -3.91, -4.0, -4.08, -4.17, -4.27, -4.35, -4.45, -4.59, -4.74, -5.01, -2.3, -2.24, -2.36, -2.53, -2.47,
               -2.59, -2.58, -2.69, -2.64, -2.68, -2.64, -2.76, -2.77, -2.89, -2.84, -2.9, -2.84, -2.97, -2.97, -3.07, -3.08, -3.16, -3.15, -3.22, -3.31, -3.4, -3.47, -3.58, -3.7,
                -3.78, -3.85, -3.99, -4.12, -4.22, -4.31, -4.39, -4.48, -4.92, -4.96, -2.43, -2.39, -2.49, -2.62, -2.6, -2.7, -2.8, -2.74, -2.86, -3.03, -3.3, -3.4, -3.37, -3.35,
                 -3.48, -3.55, -3.61, -3.72, -3.79, -4.28, -2.61, -2.56, -2.66, -2.73, -2.77, -2.89, -2.85, -2.97, -3.1, -3.05, -3.15, -3.32, -3.4, -3.45, -3.57, -3.68, -3.76, -3.84,
                  -4.21, -4.28, -2.7, -2.81, -2.76, -2.9, -2.85, -2.97, -3.11, -3.05, -3.18, -3.31, -3.27, -3.4, -3.5, -3.59, -3.67, -3.78, -3.85, -3.96, -4.05, -4.15, -4.22, -4.31,
                   -3.0, -3.12, -3.06, -3.19, -3.29, -3.4, -3.51, -3.59, -3.68, -3.8, -3.9, -3.98, -4.07, -4.16, -4.25, -4.34]
    ys10 = [16.31, 15.31, 15.4, 15.41, 16.57, 17.24, 17.77, 15.7, 15.59, 15.66, 15.45, 15.53, 15.39, 15.29, 15.22, 15.25, 15.14, 15.18, 15.09, 15.17, 15.25, 15.0, 15.2, 15.0, 15.04,
     15.08, 15.11, 15.2, 15.28, 15.34, 15.6, 15.68, 15.94, 16.03, 16.15, 16.42, 16.59, 16.7, 16.82, 17.29, 17.72, 15.66, 15.78, 15.55, 15.6, 15.45, 15.52, 15.39, 15.31, 15.24, 15.26,
      15.17, 15.1, 15.02, 15.05, 15.0, 14.96, 14.96, 14.96, 15.0, 15.05, 15.12, 15.17, 15.24, 15.34, 15.64, 15.75, 15.95, 16.12, 16.23, 16.34, 16.47, 16.58, 16.67, 16.96, 17.1, 17.22,
       17.39, 17.53, 15.85, 15.64, 15.73, 15.53, 15.58, 15.45, 15.49, 15.38, 15.28, 15.21, 15.13, 15.16, 15.09, 15.03, 15.05, 14.99, 14.97, 14.98, 15.01, 15.07, 15.16, 15.22, 15.31,
        15.36, 15.48, 15.58, 15.7, 15.9, 16.06, 16.18, 16.32, 16.42, 16.69, 17.27, 17.44, 17.54, 15.72, 15.84, 15.53, 15.6, 15.77, 15.44, 15.5, 15.35, 15.42, 15.3, 15.23, 15.17, 15.1,
         15.06, 15.03, 15.0, 15.0, 14.99, 15.02, 15.05, 15.16, 15.18, 15.27, 15.35, 15.47, 15.91, 15.9, 16.08, 16.21, 16.31, 16.45, 16.57, 16.65, 16.79, 16.93, 17.09, 17.22, 17.37,
          17.53, 15.91, 15.7, 15.78, 15.63, 15.54, 15.52, 15.42, 15.34, 15.38, 15.22, 15.3, 15.2, 15.1, 15.07, 15.05, 15.06, 15.03, 15.1, 15.2, 15.26, 15.34, 15.41, 15.47, 15.57,
           15.68, 15.77, 15.99, 16.17, 16.31, 16.42, 16.56, 16.73, 17.28, 17.44, 17.57, 17.71, 15.89, 15.66, 15.79, 15.62, 15.53, 15.58, 15.47, 15.38, 15.32, 15.24, 15.27, 15.19,
            15.07, 15.19, 15.05, 15.11, 15.06, 15.04, 15.08, 15.18, 15.29, 15.4, 15.33, 15.48, 15.56, 15.66, 15.72, 15.9, 15.98, 16.1, 16.2, 16.36, 16.46, 16.59, 16.68, 16.84, 17.42,
             17.56, 15.75, 15.83, 15.65, 15.69, 15.54, 15.57, 15.44, 15.48, 15.36, 15.39, 15.34, 15.25, 15.26, 15.14, 15.23, 15.15, 15.17, 15.12, 15.18, 15.1, 15.13, 15.23, 15.29,
              15.41, 15.52, 15.61, 15.68, 15.77, 15.88, 16.01, 16.11, 16.22, 16.35, 16.46, 16.58, 16.83, 17.17, 17.43, 15.92, 15.96, 15.82, 15.64, 15.71, 15.29, 15.58, 15.23, 15.28,
               15.51, 15.57, 15.2, 15.49, 15.14, 15.17, 15.45, 15.48, 15.12, 15.41, 15.11, 15.39, 15.13, 15.33, 15.38, 15.39, 15.45, 15.5, 15.6, 15.72, 15.84, 15.9, 16.09, 16.24,
                16.38, 16.48, 16.58, 16.69, 17.44, 17.39, 16.23, 16.29, 16.22, 15.99, 16.4, 16.11, 15.82, 16.04, 15.75, 15.71, 15.91, 15.58, 15.67, 15.83, 15.64, 15.68, 15.75, 15.95,
                 15.98, 16.63, 16.44, 16.52, 16.44, 16.46, 16.22, 16.06, 16.16, 16.02, 15.95, 15.98, 15.92, 15.89, 15.96, 15.95, 15.82, 15.92, 16.0, 16.09, 16.64, 16.67, 16.67,
                  16.55, 16.63, 16.43, 16.47, 16.34, 16.24, 16.29, 16.19, 16.11, 16.17, 16.09, 16.04, 16.03, 16.02, 16.1, 16.17, 16.33, 16.44, 16.61, 16.67, 16.74, 16.82, 16.62,
                   16.67, 16.53, 16.43, 16.39, 16.32, 16.3, 16.28, 16.3, 16.36, 16.44, 16.54, 16.61, 16.76, 16.86]
    
    # 测试数据11
    # 场景：目标为车辆，点云数量为201，无遮挡，L型/O型
    xs11 = [2.78, 2.67, 2.49, 2.38, 2.3, 2.24, 1.91, 1.85, 1.79, 1.74, 1.59, 1.38, 1.23, 1.65, 1.21, 1.2, 2.8, 2.68, 2.58, 2.51, 2.41, 2.28, 2.19, 2.09, 2.0, 1.9, 1.8, 1.71, 1.58,
     1.49, 1.43, 1.37, 1.3, 1.25, 1.2, 1.18, 2.72, 2.6, 2.51, 2.37, 2.44, 2.31, 2.21, 2.08, 1.98, 1.89, 1.79, 1.7, 1.58, 1.48, 1.42, 1.36, 1.3, 1.24, 1.22, 2.82, 2.7, 2.59, 2.5,
      2.4, 2.3, 2.21, 2.11, 1.98, 1.89, 1.79, 1.7, 1.6, 1.48, 1.38, 1.33, 1.27, 1.23, 1.22, 2.82, 2.69, 2.59, 2.48, 2.39, 2.29, 2.2, 2.1, 1.97, 1.87, 1.75, 1.82, 1.68, 1.62, 1.56,
       1.49, 1.4, 1.31, 1.26, 1.22, 1.2, 2.7, 2.58, 2.49, 2.39, 2.29, 2.19, 2.13, 2.06, 2.0, 1.94, 1.87, 1.78, 1.68, 1.58, 1.49, 1.42, 1.37, 1.31, 1.25, 1.26, 1.22, 2.75, 2.68, 2.58,
        2.48, 2.41, 2.31, 2.18, 2.09, 1.99, 1.9, 1.8, 1.71, 1.65, 1.58, 1.52, 1.46, 1.37, 1.3, 1.27, 2.78, 2.68, 2.61, 2.5, 2.4, 2.3, 2.21, 2.1, 1.98, 1.88, 1.79, 1.69, 1.6, 1.51,
         1.46, 1.4, 1.36, 1.34, 2.75, 2.66, 2.6, 2.52, 2.45, 2.39, 2.28, 2.19, 2.09, 2.0, 1.94, 1.87, 1.81, 1.66, 1.73, 1.6, 1.53, 1.47, 1.42, 1.38, 1.36, 2.59, 2.51, 2.41, 2.17,
          2.07, 2.0, 1.94, 1.58, 1.49, 1.42, 1.4, 2.58, 2.47, 2.36, 2.3, 2.19, 2.09, 1.98, 1.89, 1.78, 1.69, 1.59, 1.5, 1.48, 2.24, 2.17, 2.1]
    ys11 = [18.65, 18.59, 18.51, 18.41, 18.31, 18.31, 18.29, 18.32, 18.41, 18.47, 18.4, 18.52, 18.75, 18.27, 18.71, 19.69, 18.25, 18.16, 18.1, 18.1, 18.07, 18.06, 18.07, 18.06, 18.08,
     18.07, 18.05, 18.08, 18.1, 18.12, 18.17, 18.21, 18.27, 18.4, 18.64, 19.44, 18.21, 18.12, 18.11, 18.08, 18.1, 18.08, 18.07, 18.07, 18.06, 18.07, 18.08, 18.09, 18.11, 18.13, 18.17,
      18.22, 18.26, 18.4, 19.53, 18.31, 18.15, 18.09, 18.09, 18.08, 18.08, 18.08, 18.07, 18.05, 18.07, 18.08, 18.08, 18.12, 18.16, 18.19, 18.28, 18.42, 18.73, 20.0, 18.4, 18.23, 18.19,
       18.15, 18.17, 18.17, 18.18, 18.18, 18.18, 18.19, 18.19, 18.21, 18.17, 18.18, 18.2, 18.22, 18.26, 18.38, 18.58, 18.9, 19.82, 18.34, 18.23, 18.21, 18.21, 18.18, 18.2, 18.2, 18.2,
        18.21, 18.2, 18.22, 18.22, 18.24, 18.23, 18.24, 18.26, 18.31, 18.39, 18.6, 20.31, 18.96, 18.33, 18.24, 18.24, 18.21, 18.16, 18.17, 18.18, 18.18, 18.17, 18.18, 18.19, 18.19,
         18.3, 18.24, 18.3, 18.37, 18.45, 18.84, 19.37, 18.58, 18.43, 18.33, 18.27, 18.25, 18.24, 18.23, 18.22, 18.22, 18.24, 18.24, 18.26, 18.33, 18.4, 18.53, 18.68, 19.08, 19.79,
          18.9, 18.81, 18.8, 18.69, 18.73, 18.7, 18.67, 18.66, 18.73, 18.73, 18.84, 18.85, 18.87, 18.75, 18.81, 18.78, 18.78, 18.87, 18.94, 19.44, 20.19, 19.04, 19.17, 18.95, 18.87,
           18.88, 18.87, 18.91, 19.02, 19.11, 19.18, 19.73, 19.37, 19.33, 19.29, 19.28, 19.26, 19.27, 19.28, 19.28, 19.3, 19.32, 19.38, 19.56, 20.22, 19.74, 19.75, 19.73]
    
    # 测试数据12
    # 场景：目标为车辆，点云数量为128，无遮挡，L型
    xs12 = [4.73, 4.72, 5.23, 5.1, 5.0, 4.88, 4.82, 4.76, 4.74, 4.68, 4.67, 4.69, 4.69, 4.7, 4.71, 4.68, 4.7, 5.21, 5.06, 4.96, 4.89, 4.78, 4.75, 4.73, 4.71, 4.68, 4.68, 4.65, 4.64,
     4.67, 4.67, 4.67, 4.72, 5.2, 5.1, 5.04, 4.99, 4.93, 4.73, 4.74, 4.7, 4.69, 4.67, 4.66, 4.65, 4.69, 4.7, 4.69, 4.7, 4.71, 5.21, 5.11, 5.0, 4.91, 4.71, 4.67, 4.69, 4.7, 4.71, 4.72,
      5.19, 5.1, 5.01, 4.95, 4.91, 4.85, 4.68, 4.71, 4.69, 4.68, 4.67, 4.7, 4.71, 5.19, 5.11, 5.01, 4.94, 4.89, 4.78, 4.75, 4.76, 4.67, 5.12, 5.01, 4.9, 4.88, 4.84, 4.84, 4.74, 4.74,
       4.73, 4.72, 5.41, 5.26, 5.35, 5.19, 5.11, 5.05, 5.0, 4.93, 4.91, 4.88, 4.84, 4.85, 4.88, 4.85, 4.84, 4.83, 5.49, 5.39, 5.29, 5.21, 5.22, 5.17, 5.06, 4.95, 4.88, 4.89, 4.86,
        4.9, 4.85, 5.39, 5.19, 5.09, 4.99, 4.93, 4.9, 4.87]
    ys12 = [19.94, 19.61, 16.57, 16.59, 16.66, 16.7, 16.79, 16.84, 16.97, 17.46, 18.01, 18.37, 18.66, 18.95, 19.26, 19.82, 20.17, 16.59, 16.62, 16.74, 16.78, 16.87, 20.24, 17.02,
     17.18, 17.54, 17.81, 18.24, 18.49, 18.84, 19.13, 19.46, 19.82, 16.57, 16.58, 16.57, 16.63, 16.65, 20.29, 16.83, 16.93, 17.13, 17.54, 17.77, 17.96, 18.25, 18.56, 18.82, 19.14,
      19.47, 16.6, 16.62, 16.65, 16.71, 17.95, 18.58, 18.92, 19.24, 19.58, 19.92, 16.64, 16.65, 16.66, 16.7, 16.74, 16.77, 17.81, 18.15, 18.35, 18.88, 19.15, 19.88, 20.21, 16.64,
       16.69, 16.68, 16.67, 16.8, 16.88, 19.07, 19.36, 18.03, 16.66, 16.7, 16.77, 19.69, 19.81, 16.87, 18.74, 18.2, 18.41, 19.63, 17.11, 17.08, 17.14, 17.05, 17.02, 17.02, 17.05,
        17.06, 17.21, 17.31, 17.91, 18.21, 18.6, 18.71, 17.7, 18.94, 17.29, 17.29, 17.29, 17.36, 17.61, 17.65, 17.52, 17.39, 17.59, 17.87, 17.99, 18.55, 18.75, 17.52, 17.65, 17.64,
         17.54, 18.53, 18.7, 18.85]
    
    # 测试数据13
    # 场景：目标为车辆，点云数量为344，无遮挡，L型
    xs13 = [-5.36, -7.66, -7.65, -4.88, -4.92, -5.08, -5.03, -5.27, -5.44, -7.04, -7.24, -7.38, -7.48, -7.63, -7.79, -4.19, -4.29, -4.24, -4.34, -4.44, -4.56, -4.64, -4.72, -4.89,
     -4.88, -4.96, -5.08, -5.2, -5.37, -5.33, -5.44, -5.57, -6.33, -6.45, -6.57, -6.66, -6.74, -6.82, -6.99, -6.91, -7.07, -7.16, -7.24, -7.32, -7.42, -7.57, -7.67, -7.77, -7.84,
      -8.18, -8.26, -4.29, -4.24, -4.24, -4.94, -5.49, -5.58, -6.45, -6.58, -6.67, -6.74, -6.82, -6.98, -6.91, -7.07, -7.15, -7.23, -7.32, -7.41, -7.59, -7.66, -7.74, -7.83, -7.91,
       -8.04, -8.25, -8.25, -4.34, -4.48, -4.41, -4.57, -4.64, -4.73, -4.81, -4.96, -5.04, -5.12, -5.28, -5.2, -5.35, -5.64, -5.76, -5.85, -5.92, -6.04, -6.17, -6.25, -6.33, -4.2,
        -4.24, -4.22, -4.33, -4.45, -4.53, -4.64, -4.76, -4.85, -4.97, -5.06, -5.13, -5.22, -5.34, -5.48, -5.44, -5.57, -5.64, -5.72, -5.88, -5.81, -5.97, -6.05, -6.13, -6.2, -6.33,
         -6.45, -6.54, -6.62, -6.74, -6.87, -6.95, -7.03, -7.12, -7.28, -7.2, -7.36, -7.46, -7.66, -7.73, -7.71, -7.81, -7.93, -8.07, -8.14, -8.13, -8.22, -8.32, -4.29, -4.24, -4.35,
          -4.44, -4.56, -4.65, -4.72, -4.84, -4.95, -5.03, -5.19, -5.12, -5.27, -5.35, -5.43, -5.51, -5.64, -5.75, -5.84, -5.92, -6.04, -6.17, -6.24, -6.33, -6.45, -6.57, -6.66, -6.74,
           -6.82, -6.91, -7.07, -7.0, -7.16, -7.24, -7.32, -7.45, -7.57, -7.65, -7.75, -7.84, -7.95, -8.25, -7.58, -7.68, -7.73, -7.95, -8.0, -8.13, -4.35, -4.32, -4.43, -4.54, -4.64,
            -4.72, -4.84, -4.96, -5.04, -5.12, -5.24, -5.35, -5.43, -5.52, -5.64, -5.76, -5.84, -5.92, -6.05, -6.17, -6.26, -6.34, -6.41, -6.54, -6.67, -6.74, -6.82, -6.95, -7.07,
             -7.16, -7.23, -7.33, -7.45, -4.36, -4.31, -4.44, -4.57, -4.52, -4.65, -4.73, -4.84, -4.97, -5.05, -5.13, -5.21, -5.44, -5.53, -5.64, -5.78, -5.85, -5.94, -6.02, -6.14,
              -6.27, -6.34, -6.42, -6.51, -6.59, -6.68, -6.76, -6.85, -6.92, -7.01, -7.13, -7.24, -7.35, -7.43, -4.49, -4.45, -4.45, -4.55, -4.64, -4.79, -4.73, -4.87, -4.95, -5.01,
               -5.15, -5.27, -5.36, -5.43, -5.53, -5.67, -5.61, -5.77, -5.93, -6.01, -6.15, -6.11, -6.24, -6.33, -6.48, -6.42, -6.75, -6.91, -7.15, -7.23, -4.48, -4.54, -4.51, -4.64,
                -4.75, -4.84, -5.08, -5.16, -5.23, -5.35, -5.64, -5.88, -5.82, -5.99, -6.13, -6.21, -6.3, -6.88, -6.81, -6.96, -7.06, -4.7, -4.65, -4.63, -4.78, -4.84, -4.93, -5.05,
                 -5.17, -5.25, -5.33, -5.41, -5.63, -5.83, -6.06, -6.14, -6.23, -6.47, -6.56, -6.65, -6.72, -4.98, -5.18, -5.24, -5.31, -5.47, -5.4, -5.56, -5.64, -5.73, -5.85, -5.98,
                  -6.05]
    ys13 = [22.31, 22.07, 22.29, 22.39, 22.37, 22.35, 22.44, 22.43, 22.29, 22.15, 22.12, 22.12, 22.19, 22.09, 22.06, 23.06, 22.73, 22.93, 22.54, 22.46, 22.42, 22.38, 22.39, 22.38,
     22.75, 22.38, 22.54, 22.33, 22.32, 22.55, 22.31, 22.5, 22.19, 22.17, 22.16, 22.15, 22.15, 22.13, 22.1, 22.13, 22.13, 22.13, 22.1, 22.11, 22.14, 22.07, 22.12, 22.13, 22.08, 22.08,
      22.05, 22.68, 22.86, 23.35, 22.64, 22.51, 22.51, 22.17, 22.15, 22.16, 22.14, 22.14, 22.1, 22.12, 22.11, 22.09, 22.09, 22.08, 22.08, 22.12, 22.08, 22.07, 22.06, 22.04, 22.02,
       22.0, 22.26, 22.59, 22.43, 22.47, 22.44, 22.39, 22.41, 22.39, 22.34, 22.34, 22.35, 22.31, 22.34, 22.3, 22.26, 22.25, 22.25, 22.23, 22.21, 22.22, 22.21, 22.16, 23.28, 22.63,
        22.97, 22.47, 22.45, 22.4, 22.4, 22.35, 22.42, 22.59, 22.62, 22.58, 22.57, 22.57, 22.28, 22.45, 22.27, 22.27, 22.24, 22.22, 22.24, 22.23, 22.21, 22.23, 22.18, 22.19, 22.18,
         22.17, 22.15, 22.14, 22.13, 22.13, 22.12, 22.12, 22.1, 22.12, 22.07, 22.13, 22.45, 22.13, 22.34, 22.12, 22.21, 22.36, 22.06, 22.3, 22.03, 22.08, 22.73, 22.94, 22.58, 22.46,
          22.41, 22.43, 22.4, 22.38, 22.36, 22.33, 22.31, 22.35, 22.31, 22.27, 22.29, 22.26, 22.27, 22.24, 22.25, 22.22, 22.22, 22.22, 22.2, 22.2, 22.18, 22.16, 22.16, 22.17, 22.14,
           22.16, 22.12, 22.14, 22.13, 22.1, 22.11, 22.1, 22.08, 22.06, 22.07, 22.08, 22.03, 22.14, 22.11, 22.12, 22.06, 22.15, 22.04, 22.18, 22.6, 22.88, 22.61, 22.48, 22.43, 22.38,
            22.38, 22.37, 22.35, 22.34, 22.32, 22.3, 22.28, 22.27, 22.26, 22.26, 22.23, 22.23, 22.23, 22.24, 22.24, 22.2, 22.19, 22.18, 22.18, 22.16, 22.15, 22.13, 22.1, 22.12, 22.09,
             22.14, 22.09, 22.66, 23.24, 22.63, 22.48, 22.59, 22.45, 22.43, 22.41, 22.4, 22.37, 22.39, 22.38, 22.29, 22.3, 22.28, 22.3, 22.28, 22.3, 22.26, 22.28, 22.27, 22.24, 22.22,
              22.22, 22.22, 22.25, 22.2, 22.21, 22.21, 22.21, 22.19, 22.12, 22.17, 22.18, 22.68, 22.94, 23.32, 22.55, 22.57, 22.53, 22.65, 22.5, 22.53, 22.43, 22.47, 22.47, 22.48,
               22.45, 22.51, 22.38, 22.48, 22.47, 22.43, 22.42, 22.31, 22.47, 22.36, 22.36, 22.31, 22.36, 22.35, 22.59, 22.25, 22.23, 23.28, 22.74, 23.0, 22.62, 22.54, 22.56, 22.53,
                22.56, 22.47, 22.47, 22.44, 22.37, 22.47, 22.53, 22.38, 22.38, 22.38, 22.33, 22.39, 22.31, 22.36, 22.68, 22.88, 23.16, 22.66, 22.63, 22.6, 22.58, 22.58, 22.55, 22.53,
                 22.54, 22.43, 22.5, 22.43, 22.45, 22.45, 22.42, 22.4, 22.41, 22.37, 23.46, 22.78, 22.72, 22.66, 22.64, 22.66, 22.64, 22.63, 22.64, 22.62, 22.61, 22.57]
    
    # 测试数据14
    # 场景：目标为车辆，点云数量为178，无遮挡，L型
    xs14 = [-6.62, -6.65, -6.65, -6.58, -6.62, -6.63, -6.78, -6.96, -6.95, -7.27, -7.34, -7.44, -7.53, -7.94, -8.04, -8.25, -6.59, -6.58, -6.6, -6.69, -6.65, -6.63, -6.63, -6.74, -7.81, -6.65, -6.6,
     -6.62, -6.66, -6.65, -6.63, -6.75, -6.83, -7.06, -7.13, -7.22, -7.34, -7.46, -7.56, -7.74, -7.84, -7.93, -8.06, -8.14, -6.54, -6.63, -6.53, -6.67, -6.7, -6.64, -6.65, -6.78, -6.73, -6.71, -6.83,
      -7.22, -7.35, -7.48, -7.84, -8.03, -8.12, -6.59, -6.6, -6.57, -6.59, -6.57, -6.56, -6.55, -6.53, -8.1, -6.6, -6.58, -6.56, -6.59, -6.58, -6.55, -6.55, -6.68, -6.63, -6.74, -6.85, -7.2, -7.34,
       -7.47, -7.56, -7.65, -7.81, -8.01, -6.68, -6.64, -6.62, -6.62, -6.62, -6.66, -6.67, -6.64, -6.73, -6.86, -6.81, -6.94, -7.05, -7.27, -7.36, -7.44, -7.53, -7.62, -7.74, -7.87, -8.08, -8.21,
        -6.68, -6.66, -6.64, -6.62, -6.64, -6.66, -6.77, -6.74, -6.71, -6.87, -6.81, -6.94, -7.01, -7.1, -7.23, -7.32, -7.41, -7.58, -7.51, -7.68, -7.86, -8.08, -8.18, -6.64, -6.78, -6.78, -6.86,
         -6.83, -6.8, -6.91, -7.29, -7.23, -7.46, -7.4, -7.54, -7.66, -7.79, -7.91, -6.88, -6.87, -6.86, -6.83, -6.95, -6.92, -7.09, -7.01, -7.16, -7.23, -7.32, -7.47, -7.59, -7.68, -7.78, -7.87,
          -7.99, -6.89, -6.87, -6.85, -6.86, -6.97, -6.93, -7.04, -7.37, -7.46, -7.55, -7.19, -7.12, -7.28]
    ys14 = [26.0, 23.03, 23.34, 25.62, 24.7, 25.5, 22.86, 22.6, 23.05, 22.76, 22.74, 22.75, 22.74, 22.89, 22.97, 22.79, 25.15, 25.48, 26.29, 22.75, 22.9, 23.73, 24.09, 22.63, 22.42, 23.35, 25.2,
     26.01, 22.8, 23.05, 24.3, 22.65, 22.5, 22.37, 22.34, 22.35, 22.33, 22.31, 22.33, 22.34, 22.36, 22.39, 22.49, 22.48, 25.5, 26.21, 26.26, 22.82, 23.56, 23.62, 23.99, 22.61, 22.72, 23.26, 22.5,
      22.38, 22.35, 22.34, 22.38, 22.39, 22.44, 23.01, 23.35, 23.54, 23.96, 24.22, 24.51, 24.8, 25.13, 22.5, 24.15, 24.44, 24.73, 25.13, 25.48, 25.78, 26.12, 22.73, 22.84, 22.63, 22.55, 22.45,
       22.46, 22.46, 22.45, 22.45, 22.43, 22.47, 22.87, 23.03, 23.27, 23.6, 23.89, 25.44, 25.8, 26.09, 22.74, 22.62, 22.69, 22.59, 22.49, 22.5, 22.51, 22.5, 22.5, 22.5, 22.48, 22.46, 22.56, 22.68,
        23.51, 23.72, 23.97, 24.25, 24.62, 25.08, 22.86, 23.06, 23.28, 22.63, 22.73, 22.56, 22.54, 22.53, 22.42, 22.42, 22.41, 22.41, 22.42, 22.43, 22.46, 22.55, 22.62, 25.01, 25.2, 23.52, 22.9,
         23.09, 23.28, 22.74, 22.6, 22.7, 22.6, 22.65, 22.56, 22.51, 22.51, 22.61, 23.42, 24.32, 24.99, 25.18, 23.06, 23.22, 22.92, 22.96, 22.88, 22.83, 22.81, 22.87, 22.84, 22.84, 22.88, 22.87,
          22.96, 23.59, 23.84, 24.12, 24.43, 23.28, 23.42, 23.21, 23.1, 23.14, 23.11, 23.12, 23.17, 23.12]
    
    # 测试数据15
    # 场景：目标为车辆，点云数量为129，无遮挡，L型
    xs15 = [5.81, 5.7, 5.6, 5.49, 4.31, 4.27, 5.78, 5.68, 5.57, 5.47, 5.38, 5.27, 5.17, 5.08, 4.98, 4.88, 4.79, 4.69, 4.59, 4.51, 4.41, 4.32, 4.26, 4.25, 4.24, 5.88, 5.76, 5.61,
     5.46, 5.31, 5.16, 5.01, 4.87, 4.76, 4.67, 4.58, 4.47, 4.38, 4.3, 4.24, 4.23, 4.22, 4.21, 4.2, 5.87, 5.77, 5.66, 5.55, 5.46, 5.35, 5.25, 5.16, 5.05, 4.95, 4.86, 4.76, 4.66,
      4.57, 4.47, 4.37, 4.3, 4.23, 4.21, 4.21, 4.2, 4.18, 4.19, 5.77, 5.66, 5.56, 5.44, 5.36, 5.26, 5.15, 5.06, 4.96, 4.81, 4.66, 4.52, 4.39, 4.29, 4.24, 5.64, 5.54, 5.43, 5.33,
       5.24, 5.13, 5.02, 4.94, 4.83, 4.74, 4.65, 4.55, 4.45, 4.37, 4.38, 4.33, 4.32, 4.31, 4.19, 5.71, 5.6, 5.45, 5.35, 5.1, 4.86, 4.75, 4.52, 4.47, 4.45, 4.42, 4.43, 4.48, 4.46,
        4.46, 4.4, 5.63, 5.52, 5.43, 5.32, 5.21, 5.12, 5.02, 4.91, 4.82, 4.72, 4.62, 4.53]
    ys15 = [27.54, 27.52, 27.57, 27.47, 27.53, 28.31, 27.27, 27.23, 27.2, 27.2, 27.18, 27.18, 27.18, 27.17, 27.18, 27.18, 27.2, 27.2, 27.25, 27.27, 27.29, 27.38, 27.59, 29.86, 30.6,
     27.36, 27.29, 27.3, 27.28, 27.27, 27.26, 27.27, 27.27, 27.27, 27.27, 27.31, 27.28, 27.33, 27.42, 27.7, 28.31, 28.95, 29.64, 30.35, 27.4, 27.35, 27.32, 27.31, 27.3, 27.3, 27.28,
      27.28, 27.26, 27.3, 27.29, 27.3, 27.3, 27.32, 27.34, 27.34, 27.42, 27.65, 28.21, 28.26, 28.9, 29.47, 30.32, 27.44, 27.38, 27.37, 27.33, 27.36, 27.37, 27.36, 27.34, 27.35,
       27.36, 27.35, 27.37, 27.47, 27.48, 27.78, 27.48, 27.44, 27.39, 27.41, 27.41, 27.37, 27.37, 27.39, 27.38, 27.43, 27.43, 27.47, 27.48, 27.57, 29.57, 27.93, 28.55, 29.91, 29.85,
        27.61, 27.58, 27.8, 27.85, 27.53, 27.88, 27.85, 27.63, 28.57, 27.86, 28.95, 29.73, 28.05, 28.6, 29.22, 29.53, 27.76, 27.72, 27.7, 27.68, 27.68, 27.7, 27.69, 27.69, 27.68,
         27.71, 27.73, 27.75]
    
    # 测试数据16
    # 场景：目标为车辆，点云数量为168，无遮挡，L型
    xs16 = [-4.97, -5.08, -5.2, -5.28, -4.79, -4.88, -4.98, -5.11, -5.41, -5.48, -4.28, -4.53, -5.04, -5.35, -5.26, -5.58, -6.36, -6.56, -6.67, -6.76, -4.24, -4.29, -4.38, -4.48,
     -4.57, -4.68, -4.77, -4.9, -5.09, -5.19, -5.29, -5.44, -5.53, -5.58, -5.68, -5.79, -5.89, -6.0, -6.19, -6.3, -6.41, -6.51, -6.61, -6.71, -6.82, -7.02, -4.32, -5.1, -5.19, -5.3,
      -5.7, -5.89, -6.0, -6.2, -6.3, -6.41, -6.5, -6.61, -6.72, -6.81, -6.92, -7.03, -7.12, -7.23, -4.38, -4.47, -4.58, -4.68, -4.78, -4.87, -4.98, -5.39, -5.48, -5.58, -5.78, -4.4,
       -4.5, -4.59, -4.69, -4.77, -4.87, -4.99, -5.08, -5.18, -5.29, -5.38, -5.49, -5.6, -5.68, -5.79, -5.9, -6.02, -6.2, -6.31, -6.4, -6.51, -6.62, -6.72, -6.81, -6.93, -7.01,
        -7.13, -7.24, -7.34, -7.45, -4.42, -4.54, -4.46, -4.63, -4.75, -4.84, -5.04, -5.15, -5.24, -5.35, -5.5, -5.66, -5.76, -5.85, -5.97, -6.17, -6.27, -6.38, -6.49, -6.58,
         -6.68, -6.8, -6.89, -7.01, -7.12, -7.17, -7.27, -7.42, -4.51, -4.47, -4.8, -4.9, -5.12, -5.31, -5.42, -6.14, -6.23, -7.06, -7.16, -7.34, -4.73, -4.66, -4.81, -4.91,
          -5.11, -5.22, -5.41, -6.25, -6.67, -6.75, -6.86, -5.25, -5.32, -5.42, -5.52, -5.6, -5.71, -5.81, -5.9, -6.01, -6.12, -6.21, -6.33]
    ys16 = [28.71, 28.73, 28.78, 28.72, 28.79, 28.79, 28.74, 28.84, 28.79, 28.66, 29.23, 28.84, 28.76, 28.71, 28.81, 28.85, 28.56, 28.54, 28.52, 28.52, 29.33, 28.99, 28.9, 28.84,
     28.8, 28.8, 28.74, 28.94, 28.77, 28.73, 28.76, 28.97, 28.88, 28.67, 28.63, 28.63, 28.62, 28.62, 28.59, 28.58, 28.57, 28.54, 28.55, 28.52, 28.5, 28.51, 29.21, 28.77, 28.8,
      28.78, 28.66, 28.63, 28.6, 28.57, 28.57, 28.55, 28.54, 28.53, 28.51, 28.51, 28.5, 28.49, 28.47, 28.46, 28.86, 28.83, 28.82, 28.78, 28.76, 28.73, 28.72, 28.68, 28.65, 28.64,
       28.61, 29.06, 28.99, 28.89, 28.81, 28.78, 28.74, 28.75, 28.74, 28.73, 28.71, 28.71, 28.7, 28.69, 28.64, 28.64, 28.62, 28.67, 28.59, 28.59, 28.57, 28.58, 28.56, 28.57, 28.49,
        28.53, 28.48, 28.52, 28.49, 28.45, 28.51, 29.51, 29.01, 29.07, 28.89, 28.91, 28.93, 28.79, 28.81, 28.8, 28.78, 28.78, 28.77, 28.74, 28.73, 28.74, 28.74, 28.68, 28.7, 28.69,
         28.66, 28.61, 28.68, 28.65, 28.66, 28.66, 28.47, 28.46, 28.59, 29.12, 29.46, 28.89, 28.91, 28.9, 28.85, 28.85, 28.85, 28.72, 28.65, 28.64, 28.93, 29.09, 29.33, 29.02,
          28.94, 28.92, 28.96, 28.87, 28.81, 28.79, 28.73, 28.72, 29.39, 29.26, 29.19, 29.12, 29.04, 29.02, 29.0, 28.99, 28.97, 28.96, 28.94, 28.95]
    
    # 测试数据17
    # 场景：目标为车辆，点云数量为61，无遮挡，O型
    xs17 = [-6.79, -7.75, -7.9, -8.02, -7.0, -7.08, -7.39, -7.5, -6.81, -6.79, -7.33, -7.44, -6.69, -6.75, -6.77, -6.72, -6.93, -7.04, -7.13, -7.2, -7.4, -7.52, -7.63, -7.99,
     -6.75, -6.83, -6.79, -6.88, -6.98, -7.07, -7.17, -7.29, -7.4, -7.5, -7.62, -7.74, -7.85, -7.98, -8.11, -6.71, -6.81, -6.82, -6.92, -7.2, -7.33, -8.15, -6.95, -7.03, -7.32,
      -8.05, -6.95, -7.05, -6.99, -7.14, -7.27, -7.4, -7.5, -7.63, -7.77, -7.88, -8.03]
    ys17 = [30.78, 30.14, 30.29, 30.28, 30.24, 30.13, 30.03, 30.01, 31.37, 31.86, 30.19, 30.22, 32.17, 30.84, 30.69, 31.59, 30.46, 30.41, 30.35, 30.15, 30.12, 30.11, 30.1, 30.15,
     31.69, 30.93, 31.29, 30.73, 30.62, 30.51, 30.5, 30.52, 30.47, 30.48, 30.49, 30.5, 30.52, 30.54, 30.59, 31.48, 31.4, 32.53, 31.33, 31.07, 31.08, 32.6, 32.37, 31.62, 32.44,
      32.5, 32.64, 31.95, 32.22, 31.91, 31.92, 31.93, 31.94, 31.93, 32.0, 32.01, 32.14]
    
    # 测试数据18
    # 场景：目标为车辆，点云数量为83，无遮挡，O型
    xs18 = [-2.46, -3.74, -3.86, -3.95, -2.44, -2.51, -2.61, -2.71, -2.81, -2.91, -3.02, -3.12, -3.23, -3.34, -3.43, -3.54, -3.66, -3.75, -3.87, -3.99, -2.44, -2.63, -2.73, -2.82,
     -2.93, -3.03, -3.13, -3.24, -3.36, -3.45, -3.56, -3.68, -3.78, -3.89, -2.48, -2.63, -2.72, -2.83, -2.93, -3.03, -3.14, -3.25, -3.34, -3.46, -3.57, -3.67, -3.78, -3.89, -2.63,
      -2.72, -2.8, -2.91, -3.02, -3.11, -3.22, -3.34, -3.43, -3.55, -3.66, -3.78, -3.89, -2.62, -2.69, -2.8, -2.9, -3.01, -3.1, -3.22, -3.32, -3.44, -3.55, -3.66, -3.78, -2.75,
       -2.84, -2.95, -3.06, -3.15, -3.26, -3.38, -3.48, -3.61, -3.73]
    ys18 = [30.69, 30.04, 30.08, 29.96, 30.36, 29.99, 29.91, 29.83, 29.83, 29.79, 29.77, 29.76, 29.75, 29.76, 29.76, 29.76, 29.8, 29.78, 29.85, 29.91, 30.52, 30.03, 30.0, 29.92,
     29.88, 29.9, 29.87, 29.86, 29.86, 29.89, 29.86, 29.89, 29.97, 29.98, 30.95, 30.01, 29.98, 29.94, 29.9, 29.92, 29.88, 29.9, 29.87, 29.9, 29.89, 29.95, 29.91, 29.96, 30.81,
      30.49, 30.39, 30.32, 30.3, 30.27, 30.26, 30.26, 30.27, 30.27, 30.31, 30.34, 30.4, 31.25, 30.8, 30.72, 30.7, 30.7, 30.66, 30.67, 30.61, 30.65, 30.71, 30.72, 30.73, 31.44,
       31.27, 31.22, 31.14, 31.14, 31.08, 31.09, 31.09, 31.22, 31.24]
    
    # 测试数据19
    # 场景：目标为车辆，点云数量为35，无遮挡，I型
    xs19 = [9.32, 9.18, 9.04, 11.2, 11.06, 10.91, 10.63, 10.48, 9.35, 9.2, 11.19, 11.06, 10.9, 10.75, 10.62, 10.47, 10.32, 10.19, 10.04, 9.77, 9.62, 9.48, 9.36, 9.21, 9.07, 10.96,
     10.81, 10.68, 10.53, 10.38, 10.25, 10.1, 9.95, 9.81, 9.67]
    ys19 = [35.19, 35.17, 35.16, 35.59, 35.56, 35.52, 35.46, 35.44, 35.18, 35.15, 35.67, 35.66, 35.6, 35.58, 35.55, 35.52, 35.49, 35.48, 35.43, 35.38, 35.36, 35.36, 35.33, 35.31,
     35.3, 35.92, 35.89, 35.87, 35.84, 35.84, 35.8, 35.76, 35.73, 35.69, 35.68]
    
    # 测试数据20
    # 场景：目标为车辆，点云数量为25，无遮挡，O型
    xs20 = [7.34, 7.22, 6.28, 6.02, 6.78, 6.14, 6.01, 7.37, 7.18, 6.78, 6.64, 6.39, 6.14, 6.01, 5.89, 7.19, 7.06, 6.92, 6.77, 6.63, 6.49, 6.37, 6.13, 6.01, 5.9]
    ys20 = [36.38, 36.36, 36.28, 36.41, 36.27, 36.33, 36.39, 36.61, 36.34, 36.31, 36.32, 36.32, 36.37, 36.4, 36.5, 36.6, 36.63, 36.65, 36.46, 36.47, 36.45, 36.46, 36.65, 36.66, 36.84]
    
    # 测试数据21
    # 场景：目标为车辆，点云数量为28，无遮挡，L型
    xs21 = [5.32, 5.51, 5.37, 5.24, 6.72, 6.56, 6.29, 6.14, 6.01, 5.87, 5.73, 5.46, 5.33, 5.23, 6.78, 6.63, 6.34, 6.2, 6.07, 5.93, 5.78, 5.53, 5.41, 5.36, 6.38, 6.23, 5.56, 5.43]
    ys21 = [39.25, 38.81, 38.85, 38.91, 38.93, 38.82, 38.84, 38.84, 38.83, 38.85, 38.84, 38.87, 38.97, 39.17, 38.91, 38.86, 38.79, 38.79, 38.79, 38.82, 38.8, 38.95, 39.1, 39.78, 39.05, 39.05, 39.18, 39.3]
    
    # 测试数据22
    # 场景：目标为车辆，点云数量为27，无遮挡，I型
    xs22 = [3.37, 4.22, 4.07, 3.38, 3.36, 3.37, 3.34, 3.33, 3.31, 3.31, 3.31, 3.31, 3.34, 4.63, 4.52, 4.4, 4.32, 4.23, 4.12, 4.01, 3.91, 3.83, 3.75, 3.67, 3.6, 3.51, 3.37]
    ys22 = [35.01, 35.02, 35.02, 35.11, 35.04, 35.03, 35.2, 35.01, 35.03, 35.03, 35.03, 35.05, 35.03, 35.04, 35.03, 35.02, 35.01, 35.06, 35.1, 35.03, 35.07, 35.08, 35.04, 35.02, 35.09, 35.02, 35.09]
    
    # 测试数据23
    # 场景：目标为车辆，点云数量为240，有遮挡，难度较大
    xs23 = [3.96, 3.91, 3.84, 3.74, 3.68, 2.53, 2.59, 2.51, 2.52, 3.94, 3.86, 3.87, 3.74, 3.6, 3.69, 3.53, 3.44, 3.35, 3.24, 3.14, 3.07, 3.04, 3.03, 2.99, 2.93, 2.84, 2.81, 2.77,
     2.72, 2.64, 2.61, 2.58, 2.54, 2.49, 3.94, 3.85, 3.75, 3.74, 3.6, 3.52, 3.43, 3.35, 3.24, 3.13, 3.05, 2.95, 2.91, 2.86, 2.81, 2.74, 2.73, 2.69, 2.65, 2.65, 2.62, 2.58, 2.52,
      2.47, 2.48, 3.99, 4.05, 3.92, 3.84, 3.7, 3.63, 3.55, 3.52, 3.44, 3.44, 3.33, 3.37, 3.28, 3.23, 3.15, 3.07, 3.02, 2.98, 2.93, 2.84, 2.77, 2.71, 2.69, 2.65, 2.61, 2.57, 2.54,
       2.51, 2.52, 4.01, 3.96, 3.71, 3.6, 3.65, 3.56, 3.53, 3.46, 3.36, 3.25, 3.14, 3.04, 2.96, 2.92, 2.88, 2.83, 2.77, 2.74, 2.71, 2.67, 2.64, 2.65, 2.61, 2.56, 2.56, 3.93, 3.81,
        3.85, 3.72, 3.76, 3.64, 3.55, 3.42, 3.35, 3.21, 3.29, 3.14, 3.08, 3.02, 2.94, 2.91, 2.83, 2.79, 2.77, 2.74, 2.71, 2.66, 2.61, 2.58, 2.56, 3.95, 3.8, 3.86, 3.75, 3.67, 3.56,
         3.54, 3.44, 3.35, 3.38, 3.31, 3.25, 3.23, 3.15, 3.16, 3.1, 3.08, 3.04, 3.08, 3.05, 3.02, 2.98, 2.92, 2.86, 2.81, 2.76, 2.7, 2.67, 2.64, 4.26, 3.99, 4.06, 4.01, 3.92, 3.95,
          3.82, 3.88, 3.83, 3.74, 3.76, 3.73, 3.69, 3.62, 3.63, 3.56, 3.53, 3.53, 3.44, 3.48, 3.42, 3.34, 3.31, 3.35, 3.32, 3.3, 3.32, 3.27, 3.22, 3.15, 3.11, 4.55, 4.33, 4.26, 4.14,
           4.12, 3.97, 3.96, 3.73, 3.62, 3.59, 3.53, 3.52, 3.45, 4.54, 4.32, 4.25, 4.12, 3.78, 3.72, 3.66, 3.57, 4.11, 3.91, 3.8, 4.32, 4.25, 4.21, 4.15, 4.09, 4.05, 4.02, 3.98,
            3.95, 3.94, 3.91, 4.36, 4.31, 4.27, 4.24, 4.24, 4.23, 4.17]
    ys23 = [12.5, 12.51, 12.43, 12.35, 12.35, 13.09, 13.1, 13.24, 13.5, 12.55, 12.49, 12.57, 12.35, 12.27, 12.39, 12.24, 12.28, 12.29, 12.32, 12.34, 12.37, 12.41, 12.54, 12.59, 12.62,
     12.54, 12.63, 12.65, 12.72, 12.83, 13.03, 12.86, 13.03, 13.14, 12.53, 12.46, 12.39, 12.46, 12.27, 12.26, 12.27, 12.29, 12.32, 12.34, 12.37, 12.44, 12.54, 12.56, 12.64, 12.67,
      12.75, 12.82, 12.79, 12.85, 12.91, 12.95, 13.05, 13.17, 13.43, 12.62, 12.85, 12.63, 12.59, 12.29, 12.27, 12.26, 12.38, 12.26, 12.36, 12.28, 12.35, 12.3, 12.31, 12.35, 12.37,
       12.43, 12.46, 12.53, 12.65, 12.75, 12.84, 12.9, 12.96, 13.02, 13.02, 13.14, 13.26, 13.6, 12.86, 12.87, 12.36, 12.3, 12.31, 12.31, 12.38, 12.36, 12.33, 12.36, 12.4, 12.46,
        12.5, 12.53, 12.57, 12.65, 12.7, 12.75, 12.82, 12.86, 12.96, 13.08, 13.12, 13.03, 13.16, 12.5, 12.41, 12.45, 12.37, 12.43, 12.37, 12.44, 12.45, 12.53, 12.46, 12.57, 12.44,
         12.51, 12.55, 12.65, 12.73, 12.74, 12.89, 12.77, 12.87, 12.93, 13.05, 13.15, 13.22, 13.4, 12.58, 12.49, 12.54, 12.47, 12.45, 12.39, 12.42, 12.43, 12.4, 12.49, 12.53, 12.44,
          12.57, 12.49, 12.65, 12.72, 12.52, 12.76, 13.49, 13.6, 13.79, 12.79, 12.86, 12.98, 13.09, 13.18, 13.26, 13.37, 13.47, 14.75, 12.7, 12.91, 12.94, 12.69, 12.96, 12.69, 13.0,
           13.05, 12.76, 13.1, 13.15, 12.8, 12.84, 13.15, 12.88, 12.93, 13.25, 12.97, 13.31, 13.37, 12.98, 13.03, 13.41, 13.5, 13.63, 14.11, 13.06, 13.14, 13.26, 13.4, 14.56, 14.76,
            14.78, 13.12, 13.72, 13.57, 13.9, 13.8, 13.78, 14.06, 14.05, 14.18, 14.35, 14.64, 14.74, 14.79, 13.78, 14.3, 14.24, 14.23, 14.25, 13.77, 14.38, 14.29, 13.68, 13.76, 13.83,
             13.89, 13.94, 14.0, 14.08, 14.13, 14.22, 14.34, 14.44, 13.9, 13.99, 14.12, 14.21, 14.37, 14.53, 14.52]
    
    # 测试数据24
    # 场景：目标为车辆，点云数量为328，有遮挡
    xs24 = [3.55, 3.46, 3.43, 3.33, 3.27, 3.21, 2.07, 3.84, 3.76, 3.71, 3.79, 3.63, 3.55, 3.41, 3.32, 3.23, 3.13, 3.18, 3.06, 3.0, 2.93, 2.84, 2.74, 2.65, 2.62, 2.6, 2.54, 2.44, 2.37,
     2.31, 2.27, 2.22, 2.17, 2.12, 2.08, 2.05, 2.05, 4.04, 3.93, 3.85, 3.7, 3.79, 3.62, 3.52, 3.42, 3.47, 3.32, 3.26, 3.12, 3.18, 3.08, 3.04, 2.99, 2.93, 2.84, 2.75, 2.65, 2.55, 2.49,
      2.44, 2.46, 2.41, 2.34, 2.34, 2.25, 2.19, 2.24, 2.15, 2.14, 2.11, 2.08, 2.04, 2.02, 4.04, 3.94, 3.85, 3.8, 3.74, 3.65, 3.56, 3.47, 3.37, 3.22, 3.24, 3.14, 3.11, 3.04, 3.02, 3.07,
       2.97, 2.92, 2.98, 2.84, 2.73, 2.65, 2.63, 2.58, 2.55, 2.53, 2.47, 2.41, 2.34, 2.27, 2.21, 2.18, 2.14, 2.1, 2.08, 2.08, 4.03, 3.95, 3.84, 3.88, 3.77, 3.72, 3.64, 3.49, 3.57,
        3.33, 3.25, 3.14, 3.02, 2.94, 2.85, 2.84, 2.76, 2.74, 2.64, 2.58, 2.52, 2.47, 2.41, 2.33, 2.27, 2.24, 2.2, 2.17, 2.13, 2.12, 2.07, 4.03, 3.94, 3.8, 3.87, 3.73, 3.74, 3.61,
         3.65, 3.55, 3.46, 3.49, 3.32, 3.37, 3.25, 3.18, 3.13, 3.01, 2.94, 2.86, 2.74, 2.65, 2.58, 2.53, 2.47, 2.43, 2.41, 2.37, 2.33, 2.26, 2.21, 2.18, 2.14, 2.11, 4.03, 3.91, 3.97,
          3.84, 3.74, 3.62, 3.53, 3.45, 3.36, 3.34, 3.25, 3.13, 3.04, 3.0, 2.93, 2.93, 2.85, 2.85, 2.76, 2.7, 2.77, 2.72, 2.64, 2.64, 2.61, 2.59, 2.58, 2.53, 2.53, 2.45, 2.46, 2.41,
           2.36, 2.31, 2.25, 2.2, 4.13, 4.02, 4.06, 3.93, 3.98, 3.89, 3.84, 3.75, 3.64, 3.53, 3.43, 3.35, 3.3, 3.32, 3.22, 3.23, 3.17, 3.11, 3.13, 3.02, 3.03, 2.95, 2.91, 2.93, 2.87,
            2.82, 2.88, 2.86, 2.87, 2.85, 2.74, 2.7, 2.67, 2.64, 4.6, 4.54, 4.12, 4.06, 4.04, 3.92, 3.9, 3.83, 3.83, 3.83, 3.73, 3.75, 3.75, 3.72, 3.63, 3.65, 3.57, 3.53, 3.5, 3.44,
             3.48, 3.36, 3.31, 3.24, 3.16, 3.11, 3.07, 2.99, 2.97, 4.6, 4.53, 4.17, 4.04, 4.04, 3.96, 3.92, 3.82, 3.74, 3.75, 3.3, 3.22, 3.09, 4.53, 4.02, 4.08, 3.98, 3.79, 3.64,
              3.42, 3.34, 3.29, 3.24, 4.23, 4.1, 4.17, 4.11, 4.05, 4.01, 4.05, 4.01, 3.97, 3.92, 3.83, 3.76, 3.7, 3.64, 3.59, 3.53, 3.47, 3.45, 3.41, 4.24, 4.14, 4.09, 4.02, 3.94,
               3.88, 3.85, 3.81, 3.78, 3.76, 3.75, 3.72, 3.67, 4.28]
    ys24 = [12.46, 12.38, 12.43, 12.27, 12.27, 12.34, 13.4, 12.76, 12.61, 12.62, 12.72, 12.57, 12.47, 12.36, 12.29, 12.22, 12.17, 12.2, 12.19, 12.2, 12.21, 12.22, 12.25, 12.28,
     12.32, 12.44, 12.41, 12.46, 12.54, 12.61, 12.63, 12.74, 12.82, 12.97, 12.92, 13.01, 13.13, 12.88, 12.79, 12.74, 12.63, 12.79, 12.53, 12.46, 12.39, 12.44, 12.32, 12.34, 12.19,
      12.22, 12.18, 12.2, 12.2, 12.23, 12.23, 12.25, 12.28, 12.34, 12.39, 12.46, 12.66, 12.74, 12.55, 12.78, 12.66, 12.73, 12.83, 12.77, 12.86, 12.9, 12.97, 13.05, 13.24, 12.85,
       12.74, 12.68, 12.73, 12.8, 12.74, 12.68, 12.61, 12.38, 12.2, 12.22, 12.19, 12.32, 12.19, 12.29, 12.33, 12.18, 12.22, 12.31, 12.22, 12.27, 12.27, 12.33, 12.38, 12.45, 12.52,
        12.48, 12.55, 12.65, 12.74, 12.83, 12.89, 12.96, 13.02, 13.15, 13.46, 12.85, 12.8, 12.7, 12.73, 12.67, 12.74, 12.81, 12.7, 12.81, 12.3, 12.26, 12.26, 12.26, 12.26, 12.25,
         12.31, 12.25, 12.32, 12.36, 12.4, 12.43, 12.5, 12.56, 12.63, 12.68, 12.77, 12.8, 12.89, 12.94, 13.05, 13.16, 12.89, 12.81, 12.68, 12.78, 12.59, 12.64, 12.49, 12.54, 12.48,
          12.39, 12.44, 12.31, 12.33, 12.29, 12.3, 12.37, 12.36, 12.46, 12.5, 12.38, 12.45, 12.5, 12.53, 12.59, 12.61, 12.79, 12.67, 12.76, 12.86, 12.95, 13.03, 13.1, 13.26, 12.9,
           12.81, 12.85, 12.76, 12.67, 12.58, 12.52, 12.46, 12.39, 12.42, 12.37, 12.37, 12.36, 12.42, 12.33, 12.45, 12.36, 12.48, 12.39, 12.43, 12.56, 12.61, 12.46, 12.67, 13.56,
            13.69, 12.5, 12.57, 12.74, 12.63, 12.84, 12.93, 13.05, 13.13, 13.26, 13.36, 13.05, 12.91, 12.96, 12.88, 12.97, 14.77, 12.79, 12.74, 12.68, 12.64, 12.64, 12.69, 12.74,
             13.06, 12.75, 13.08, 12.81, 12.87, 13.16, 12.89, 13.26, 12.89, 12.92, 13.35, 12.97, 13.05, 13.46, 13.61, 13.89, 14.06, 13.15, 13.27, 13.38, 13.46, 15.33, 15.34, 14.49,
              12.91, 14.58, 12.95, 13.12, 12.87, 13.04, 14.7, 12.85, 13.12, 14.69, 14.72, 12.87, 13.68, 12.89, 12.92, 13.5, 12.96, 13.6, 13.03, 13.74, 13.74, 13.72, 13.99, 14.0,
               14.12, 14.28, 15.29, 15.32, 14.49, 13.06, 14.61, 13.02, 13.07, 14.7, 14.05, 14.74, 14.22, 14.17, 14.19, 15.26, 13.11, 13.13, 13.13, 14.71, 13.71, 14.3, 14.2, 14.22,
                14.18, 13.36, 13.2, 13.32, 13.39, 13.19, 13.24, 13.43, 13.47, 13.51, 13.58, 13.66, 13.77, 13.83, 13.91, 13.98, 14.07, 14.16, 14.28, 14.36, 13.56, 13.59, 13.65,
                 13.69, 13.79, 13.87, 13.91, 13.97, 14.07, 14.17, 14.32, 14.42, 14.44, 14.9]
    
    # 测试数据25
    # 场景：目标为车辆，点云数量为401，无遮挡，L型
    xs25 = [3.24, 3.16, 3.05, 2.9, 2.96, 2.86, 2.82, 2.74, 4.7, 4.77, 4.63, 4.55, 4.42, 4.48, 4.33, 4.17, 4.03, 4.08, 3.9, 3.96, 3.84, 3.76, 3.63, 3.55, 3.46, 3.42, 3.34, 3.26,
     3.16, 3.01, 3.08, 2.93, 2.03, 1.95, 1.89, 1.83, 1.77, 1.72, 1.66, 1.62, 1.57, 4.71, 4.65, 4.54, 4.43, 4.29, 4.37, 4.21, 4.27, 4.15, 4.04, 3.95, 3.73, 3.63, 3.53, 3.42, 3.35,
      3.26, 3.15, 3.04, 2.93, 2.82, 2.86, 2.72, 2.64, 2.62, 2.56, 2.45, 2.34, 2.25, 2.19, 2.14, 2.08, 2.02, 1.96, 1.92, 1.88, 1.83, 1.75, 1.69, 1.65, 1.61, 1.6, 1.58, 1.58, 4.74,
       4.58, 4.67, 4.52, 4.45, 4.3, 4.36, 4.22, 4.09, 4.17, 4.02, 4.03, 3.93, 3.96, 3.83, 3.71, 3.77, 3.64, 3.51, 3.58, 3.39, 3.46, 3.35, 3.22, 3.24, 3.1, 3.16, 3.16, 3.03, 3.09,
        2.96, 2.85, 2.76, 2.65, 2.54, 2.46, 2.35, 2.27, 2.23, 2.19, 2.14, 2.08, 2.02, 1.96, 1.92, 1.89, 1.84, 1.77, 1.73, 1.66, 1.67, 1.64, 4.74, 4.63, 4.55, 4.41, 4.48, 4.33, 4.23,
         4.13, 4.0, 4.06, 3.9, 3.94, 3.84, 3.74, 3.64, 3.52, 3.58, 3.43, 3.32, 3.24, 3.18, 3.09, 2.93, 2.84, 2.74, 2.73, 2.65, 2.56, 2.51, 2.45, 2.45, 2.39, 2.37, 2.34, 2.25, 2.17,
          2.13, 2.07, 2.01, 1.98, 1.92, 1.84, 1.76, 1.7, 1.67, 1.63, 4.73, 4.63, 4.54, 4.56, 4.41, 4.48, 4.33, 4.19, 4.27, 4.13, 4.0, 4.06, 3.94, 3.84, 3.74, 3.64, 3.69, 3.52, 3.57,
           3.44, 3.46, 3.35, 3.39, 3.23, 3.15, 3.07, 2.92, 2.99, 2.84, 2.77, 2.73, 2.63, 2.54, 2.43, 2.35, 2.25, 2.16, 2.13, 2.09, 2.03, 1.97, 1.91, 1.83, 1.76, 1.72, 1.7, 4.78,
            4.71, 4.65, 4.53, 4.43, 4.32, 4.35, 4.21, 4.28, 4.12, 4.03, 3.93, 3.84, 3.75, 3.62, 3.53, 3.42, 3.47, 3.31, 3.36, 3.21, 3.23, 3.13, 3.02, 3.06, 2.94, 2.8, 2.86, 2.73,
             2.64, 2.54, 2.43, 2.46, 2.42, 2.35, 2.36, 2.3, 2.26, 2.21, 2.24, 2.15, 2.18, 2.12, 2.05, 2.04, 1.99, 1.95, 1.89, 1.95, 1.91, 1.86, 1.82, 1.88, 1.81, 1.77, 4.8, 4.72,
              4.64, 4.5, 4.58, 4.43, 4.33, 4.22, 4.14, 4.04, 3.91, 3.97, 3.83, 3.71, 3.76, 3.59, 3.64, 3.52, 3.44, 3.3, 3.36, 3.23, 3.14, 3.05, 2.94, 2.86, 2.86, 2.82, 2.75, 2.74,
               2.66, 2.63, 2.62, 2.54, 2.56, 2.52, 2.51, 2.48, 2.42, 2.37, 2.34, 2.3, 2.27, 4.69, 4.76, 4.62, 4.55, 4.4, 4.48, 4.33, 4.22, 4.14, 3.82, 3.73, 3.7, 3.75, 3.62, 3.63,
                3.55, 3.57, 3.45, 3.45, 3.34, 3.25, 3.21, 3.19, 3.14, 3.03, 3.05, 2.93, 2.98, 2.83, 2.75, 2.69, 2.65, 2.61, 2.57, 2.53, 4.2, 4.27, 3.78, 3.72, 3.62, 3.56, 3.53,
                 3.49, 4.22, 4.28, 3.64, 3.57, 3.17, 4.23, 4.23, 4.16, 3.91, 3.83, 3.71, 3.77, 3.63, 3.65, 3.56, 3.56, 3.5, 3.46, 3.42, 3.36, 3.3, 4.24, 4.13, 4.0, 4.07, 3.94,
                  3.79, 3.72, 3.63, 3.55, 4.05]
    ys25 = [5.0, 4.94, 4.86, 4.67, 4.77, 4.69, 4.72, 4.63, 6.31, 6.36, 6.24, 6.18, 6.06, 6.11, 5.97, 5.86, 5.7, 5.72, 5.59, 5.65, 5.56, 5.47, 5.36, 5.25, 5.17, 5.21, 5.06, 5.03,
     4.94, 4.79, 4.89, 4.73, 4.85, 4.92, 4.97, 5.03, 5.12, 5.22, 5.32, 5.81, 5.82, 6.27, 6.26, 6.15, 6.04, 5.9, 5.98, 5.85, 5.92, 5.8, 5.73, 5.66, 5.47, 5.37, 5.26, 5.17, 5.23,
      5.04, 4.95, 4.84, 4.77, 4.64, 4.71, 4.58, 4.59, 4.62, 4.61, 4.64, 4.64, 4.67, 4.69, 4.72, 4.77, 4.83, 4.87, 4.92, 4.95, 5.05, 5.13, 5.23, 5.28, 5.32, 5.39, 5.49, 5.64,
       6.38, 6.24, 6.33, 6.17, 6.12, 5.97, 6.02, 5.87, 5.74, 5.86, 5.67, 5.73, 5.58, 5.66, 5.51, 5.39, 5.44, 5.33, 5.16, 5.25, 5.1, 5.16, 5.17, 5.01, 5.24, 4.89, 4.95, 5.24,
        4.87, 5.15, 4.9, 4.64, 4.62, 4.62, 4.63, 4.67, 4.66, 4.69, 4.72, 4.77, 4.82, 4.87, 4.92, 4.99, 5.03, 5.08, 5.16, 5.24, 5.3, 5.34, 5.43, 5.52, 6.36, 6.26, 6.17, 6.03,
         6.13, 5.97, 5.86, 5.79, 5.69, 5.71, 5.6, 5.62, 5.54, 5.45, 5.35, 5.25, 5.31, 5.15, 5.09, 5.15, 5.18, 5.0, 4.72, 4.66, 4.65, 4.73, 4.67, 4.66, 4.75, 4.64, 4.75, 4.85,
          4.67, 4.72, 4.74, 4.77, 4.82, 4.89, 4.93, 4.97, 5.04, 5.13, 5.24, 5.32, 5.42, 5.57, 6.34, 6.27, 6.17, 6.22, 6.08, 6.13, 5.98, 5.86, 5.94, 5.8, 5.69, 5.74, 5.65, 5.56,
           5.47, 5.37, 5.42, 5.26, 5.31, 5.19, 5.29, 5.08, 5.15, 4.98, 4.93, 4.84, 4.77, 4.81, 4.7, 4.79, 4.81, 4.78, 4.77, 4.73, 4.76, 4.83, 4.89, 4.92, 4.96, 5.0, 5.07, 5.13,
            5.24, 5.36, 5.46, 5.52, 6.41, 6.36, 6.31, 6.2, 6.08, 5.96, 6.04, 5.88, 5.96, 5.78, 5.72, 5.65, 5.57, 5.49, 5.38, 5.3, 5.21, 5.25, 5.1, 5.15, 5.0, 5.05, 4.96, 4.86,
             4.95, 4.86, 4.76, 4.82, 4.75, 4.75, 4.76, 4.78, 4.99, 5.0, 4.81, 5.06, 5.12, 4.86, 4.91, 5.14, 4.96, 5.19, 5.23, 5.04, 5.35, 5.46, 5.14, 5.22, 5.5, 5.58, 5.28,
              5.34, 5.67, 5.75, 5.37, 6.47, 6.41, 6.34, 6.2, 6.29, 6.16, 6.05, 5.97, 5.87, 5.78, 5.66, 5.78, 5.58, 5.49, 5.52, 5.4, 5.44, 5.34, 5.29, 5.21, 5.23, 5.17, 5.17,
               5.18, 5.23, 4.84, 5.29, 5.31, 4.83, 5.36, 4.84, 4.91, 5.37, 4.93, 5.4, 5.45, 5.98, 5.46, 5.55, 5.63, 5.74, 5.84, 5.95, 6.4, 6.47, 6.35, 6.28, 6.14, 6.23, 6.08,
                5.98, 5.95, 5.59, 5.35, 5.49, 5.52, 5.39, 5.42, 5.4, 5.45, 5.36, 5.64, 5.36, 5.38, 5.42, 6.14, 5.42, 5.47, 5.93, 5.56, 6.04, 5.53, 5.56, 5.64, 5.68, 5.74, 5.8,
                 5.86, 6.05, 6.13, 5.41, 5.45, 5.53, 5.5, 5.55, 5.56, 6.11, 6.15, 5.61, 5.6, 6.14, 6.19, 6.23, 6.16, 6.0, 5.89, 5.78, 5.85, 5.7, 5.86, 5.73, 5.9, 5.96, 6.0,
                  6.02, 6.08, 6.14, 6.3, 6.19, 6.07, 6.14, 6.04, 6.06, 6.06, 6.11, 6.18, 6.26]
    
    # 测试数据26
    # 场景：目标为车辆，点云数量为76，有遮挡，难度较大
    xs26 = [26.03, 25.98, 26.02, 25.91, 25.95, 25.93, 25.96, 25.96, 26.05, 26.13, 26.76, 26.78, 26.83, 25.42, 25.4, 26.07, 26.83, 26.76, 26.86, 26.7, 26.68, 26.69, 25.12, 25.22,
     25.15, 24.71, 26.54, 24.8, 24.73, 24.72, 24.83, 25.01, 25.06, 25.1, 25.15, 23.93, 23.98, 23.98, 24.06, 24.16, 24.36, 24.69, 25.15, 25.66, 26.32, 26.98, 24.52, 25.11, 25.69,
      26.4, 27.19, 24.08, 24.12, 24.17, 24.24, 23.86, 23.92, 24.11, 24.73, 24.98, 25.55, 26.27, 27.07, 23.95, 23.93, 24.13, 24.68, 25.22, 25.76, 26.54, 27.23, 24.26, 24.76, 25.47,
       26.42, 26.84]
    ys26 = [2.45, 2.35, 2.25, 2.15, 2.07, 1.97, 1.88, 1.8, 1.71, 1.62, 2.08, 2.0, 1.9, 1.63, 1.53, 1.48, 1.44, 2.27, 2.18, 2.09, 1.99, 1.89, 1.51, 1.35, 1.25, 2.01, 2.07, 1.76, 1.66,
     1.57, 1.5, 1.42, 1.33, 1.25, 1.16, 1.83, 1.74, 1.66, 1.59, 1.5, 1.43, 1.37, 1.3, 1.23, 1.17, 1.11, 1.4, 1.34, 1.28, 1.22, 1.17, 1.7, 1.63, 1.54, 1.46, 1.61, 1.52, 1.46, 1.41,
      1.33, 1.28, 1.22, 1.15, 1.58, 1.49, 1.41, 1.37, 1.3, 1.24, 1.18, 1.12, 1.46, 1.41, 1.36, 1.31, 1.24]
    
    # 测试数据27
    # 场景：目标为车辆，点云数量为70，有遮挡，难度较大
    xs27 = [20.83, 22.31, 20.91, 20.99, 22.27, 20.77, 20.28, 20.27, 20.33, 20.33, 20.34, 20.31, 20.42, 20.45, 20.52, 20.86, 21.38, 22.23, 20.25, 20.29, 20.3, 20.37, 20.4, 20.49, 20.48,
     20.51, 21.29, 20.34, 20.38, 20.41, 20.45, 21.4, 20.28, 20.27, 20.3, 20.19, 20.24, 20.23, 20.28, 20.27, 20.3, 20.33, 20.41, 21.44, 20.19, 20.24, 20.27, 20.31, 20.33, 20.42, 20.48,
      21.9, 20.22, 20.26, 20.26, 20.29, 20.32, 20.36, 20.45, 21.63, 20.41, 20.47, 20.49, 20.56, 20.76, 22.58, 20.63, 20.65, 20.69, 20.73]
    ys27 = [1.89, 1.64, 1.75, 1.68, 1.55, 1.7, 2.13, 2.06, 1.99, 1.92, 1.85, 1.77, 1.72, 1.64, 1.57, 1.53, 1.49, 1.47, 1.98, 1.92, 1.84, 1.77, 1.71, 1.64, 1.57, 1.5, 1.41, 1.7, 1.63,
     1.56, 1.5, 1.34, 1.91, 1.83, 1.77, 1.94, 1.88, 1.8, 1.73, 1.67, 1.59, 1.52, 1.46, 1.31, 1.83, 1.76, 1.7, 1.63, 1.56, 1.5, 1.43, 1.38, 1.83, 1.77, 1.69, 1.62, 1.55, 1.49, 1.42,
      1.35, 1.6, 1.53, 1.47, 1.4, 1.34, 1.38, 1.58, 1.51, 1.44, 1.37]
    
    # 测试数据28
    # 场景：目标为车辆，点云数量为112，无遮挡，O型
    xs28 = [29.2, 29.14, 29.11, 29.1, 29.1, 29.08, 29.1, 29.09, 29.1, 29.13, 29.17, 29.23, 28.82, 28.91, 28.85, 28.49, 28.43, 28.39, 28.4, 28.45, 28.42, 28.35, 28.42, 28.62, 28.02,
     27.95, 27.9, 27.84, 27.8, 27.73, 27.68, 27.7, 27.67, 27.69, 27.76, 27.78, 27.84, 27.85, 27.91, 28.21, 29.29, 27.32, 27.28, 27.28, 27.29, 27.29, 27.31, 27.35, 27.37, 27.48, 27.65,
      27.6, 27.84, 27.79, 27.62, 27.47, 27.4, 27.37, 27.33, 27.67, 27.44, 27.31, 27.38, 27.19, 27.2, 27.26, 27.28, 27.28, 27.27, 27.14, 27.32, 27.3, 27.18, 27.23, 27.31, 27.39, 27.54,
       27.82, 27.32, 27.4, 27.55, 27.37, 27.29, 27.19, 27.18, 27.14, 27.12, 27.12, 27.11, 27.15, 27.11, 27.16, 27.15, 27.2, 27.24, 27.47, 27.28, 27.16, 27.16, 27.23, 27.11, 27.11, 27.12,
        27.12, 27.13, 27.36, 27.16, 27.24, 27.35, 27.41, 27.53, 27.77]
    ys28 = [2.63, 2.52, 2.41, 2.32, 2.21, 2.1, 2.01, 1.9, 1.8, 1.71, 1.6, 1.5, 2.74, 1.64, 1.53, 2.88, 2.76, 1.97, 1.86, 1.78, 1.67, 1.56, 1.46, 1.38, 2.82, 2.71, 2.61, 2.51, 2.41, 2.3,
     2.21, 2.1, 2.0, 1.91, 1.82, 1.72, 1.63, 1.53, 1.43, 1.36, 1.3, 2.26, 2.17, 2.07, 1.97, 1.88, 1.78, 1.68, 1.6, 1.5, 1.41, 1.32, 1.23, 2.9, 2.78, 2.66, 2.55, 2.46, 2.36, 2.93, 2.82,
      2.71, 2.61, 2.51, 2.41, 2.31, 2.23, 2.13, 2.03, 1.93, 1.84, 1.74, 1.65, 1.55, 1.46, 1.37, 1.28, 1.19, 1.5, 1.41, 1.33, 2.85, 2.75, 2.64, 2.54, 2.45, 2.35, 2.25, 2.16, 2.06, 1.96,
       1.88, 1.78, 1.68, 1.6, 2.76, 2.64, 2.53, 2.44, 2.35, 2.24, 2.15, 2.05, 1.95, 1.87, 1.78, 1.67, 1.59, 1.5, 1.4, 1.32, 1.23]
    
    # 测试数据29
    # 场景：目标为车辆，点云数量为160，无遮挡，O型
    xs29 = [25.83, 25.7, 25.69, 25.65, 25.65, 25.65, 25.65, 25.64, 25.67, 25.69, 25.73, 25.79, 26.01, 25.53, 25.48, 25.5, 25.43, 25.48, 25.39, 25.46, 25.44, 25.44, 25.44, 25.5, 25.47,
     26.07, 25.18, 26.19, 26.16, 26.13, 26.16, 25.08, 26.0, 25.22, 25.12, 24.86, 24.86, 24.67, 24.55, 24.53, 24.52, 24.47, 24.46, 24.47, 24.45, 24.49, 24.52, 24.54, 24.58, 24.6, 24.64,
      24.7, 24.87, 25.14, 24.18, 24.19, 24.22, 24.31, 24.5, 24.81, 25.82, 25.8, 24.43, 24.28, 24.26, 24.2, 24.18, 24.14, 24.08, 24.06, 24.05, 24.06, 24.07, 24.12, 24.12, 24.05, 24.08,
       24.0, 23.81, 23.75, 23.73, 23.71, 23.71, 23.71, 23.74, 23.73, 23.7, 23.72, 23.74, 23.8, 24.05, 24.09, 24.18, 24.13, 24.25, 23.9, 23.8, 23.8, 23.72, 23.72, 23.79, 23.79, 23.81,
        23.8, 23.79, 23.68, 23.82, 23.86, 23.7, 23.77, 23.78, 23.88, 23.93, 24.06, 24.23, 23.9, 23.81, 23.73, 23.72, 23.68, 23.68, 23.68, 23.69, 23.72, 23.71, 23.71, 23.69, 23.67, 23.69,
         23.71, 23.78, 23.83, 23.93, 24.02, 24.17, 23.96, 23.84, 23.75, 23.69, 23.7, 23.84, 23.64, 23.7, 23.67, 23.72, 23.74, 23.75, 23.86, 23.79, 23.71, 23.8, 23.84, 23.94, 24.04, 24.29,
          24.37, 24.08, 24.11, 24.06, 24.13]
    ys29 = [2.95, 2.84, 2.75, 2.65, 2.56, 2.48, 2.38, 2.29, 2.21, 2.12, 2.02, 1.95, 1.87, 3.13, 3.04, 2.95, 2.85, 2.58, 2.49, 2.41, 2.23, 2.14, 2.04, 1.97, 1.87, 3.17, 2.43, 2.34, 2.25,
     2.15, 2.06, 1.8, 1.77, 1.64, 3.13, 3.19, 3.1, 3.0, 2.89, 2.8, 2.7, 2.62, 2.53, 2.44, 2.36, 2.27, 2.19, 2.11, 2.02, 1.93, 1.86, 1.77, 1.69, 1.63, 1.98, 1.9, 1.82, 1.74, 1.66, 1.61,
      1.58, 3.4, 3.13, 3.02, 2.94, 2.84, 2.75, 2.67, 2.57, 2.48, 2.39, 2.32, 2.23, 2.14, 2.07, 3.14, 3.05, 2.95, 2.85, 2.76, 2.67, 2.59, 2.5, 2.41, 2.34, 2.25, 2.16, 2.09, 2.0, 1.92, 1.86,
       1.78, 1.7, 1.6, 1.53, 3.16, 3.06, 2.97, 2.88, 2.79, 2.71, 2.62, 2.55, 2.46, 2.37, 2.29, 2.21, 2.13, 2.04, 1.96, 1.87, 1.8, 1.72, 1.64, 1.57, 3.15, 3.05, 2.95, 2.88, 2.78, 2.69, 2.62,
        2.53, 2.45, 2.36, 2.28, 2.19, 2.1, 2.03, 1.95, 1.86, 1.79, 1.71, 1.63, 1.56, 3.12, 3.02, 2.92, 2.84, 2.75, 2.68, 2.58, 2.5, 2.41, 2.34, 2.25, 2.17, 2.1, 2.01, 1.91, 1.84, 1.76, 1.68,
         1.6, 1.54, 1.58, 3.18, 3.1, 3.0, 2.93]
    
    # 测试数据30
    # 场景：目标为车辆，点云数量为841，无遮挡，L型
    xs30 = [2.03, 2.08, 2.15, 2.2, 2.25, 2.32, 2.38, 2.44, 1.28, 1.32, 1.36, 1.41, 1.45, 1.54, 1.62, 1.67, 1.72, 1.77, 1.81, 1.86, 1.9, 1.96, 2.01, 2.07, 2.12, 2.18, 2.24, 2.29, 2.35, 2.41, 2.47,
     2.54, 2.6, 2.67, 2.73, 2.79, 2.87, 0.95, 1.01, 1.05, 1.09, 1.15, 1.18, 1.23, 1.28, 1.32, 1.36, 1.4, 1.46, 1.49, 1.54, 1.59, 1.72, 1.77, 1.82, 1.88, 1.92, 2.04, 2.09, 2.15, 2.57, 2.62, 2.7,
      2.75, 2.82, 2.89, 2.95, 3.03, 1.59, 1.63, 1.68, 2.1, 2.15, 2.63, 2.7, 2.92, 2.96, 3.03, 3.1, 1.11, 1.58, 1.63, 1.68, 1.72, 2.15, 2.2, 2.68, 2.75, 2.8, 3.02, 3.08, 3.16, 3.23, 1.16, 1.49,
       1.53, 1.57, 1.62, 1.67, 1.88, 2.2, 2.25, 2.69, 2.74, 2.8, 2.86, 2.99, 3.06, 3.13, 3.2, 3.28, 0.93, 0.98, 1.13, 1.47, 1.51, 1.54, 1.58, 1.63, 1.68, 1.71, 1.76, 1.81, 2.16, 2.27, 2.35, 2.38,
        2.44, 2.5, 2.56, 2.61, 2.69, 2.73, 2.8, 2.87, 2.93, 2.99, 3.06, 3.13, 3.2, 3.27, 3.34, 0.5, 0.55, 0.59, 0.64, 0.69, 0.73, 0.78, 0.92, 0.96, 1.01, 1.1, 1.17, 1.22, 1.25, 1.3, 1.34, 1.39,
         1.43, 1.47, 1.51, 1.56, 1.65, 1.7, 1.74, 1.79, 1.97, 2.02, 2.07, 2.13, 2.18, 2.24, 2.29, 2.36, 2.4, 2.47, 2.52, 2.58, 2.64, 2.71, 2.77, 2.83, 2.91, 2.96, 3.02, 3.1, 3.16, 3.23, 0.32, 0.37,
          0.42, 0.46, 0.5, 0.55, 0.59, 0.63, 0.68, 0.73, 0.76, 0.81, 0.86, 0.9, 0.94, 0.99, 1.03, 1.07, 1.12, 1.15, 1.2, 1.25, 1.29, 1.33, 1.38, 1.42, 1.47, 1.51, 1.55, 1.61, 1.66, 1.7, 1.86, 1.9,
           1.97, 2.01, 2.06, 2.12, 2.18, 2.23, 2.29, 2.35, 2.41, 2.45, 2.52, 2.57, 2.64, 2.7, 2.76, 2.83, 2.89, 2.95, 3.01, 3.09, 3.14, 3.21, 3.29, -0.11, -0.07, -0.02, 0.03, 0.07, 0.12, 0.17,
            0.21, 0.26, 0.31, 0.34, 0.39, 0.44, 0.48, 0.52, 0.57, 0.6, 0.64, 0.69, 0.73, 0.77, 0.82, 0.85, 0.9, 0.95, 0.98, 1.03, 1.07, 1.1, 1.15, 1.2, 1.24, 1.29, 1.33, 1.38, 1.43, 1.48, 1.52,
             1.57, 1.62, 1.68, 1.71, 1.87, 1.92, 1.98, 2.02, 2.08, 2.14, 2.18, 2.25, 2.3, 2.35, 2.42, 2.47, 2.53, 2.59, 2.65, 2.71, 2.78, 2.84, 2.9, 2.96, 3.03, 3.09, 3.15, 3.23, 3.3, -0.21,
              -0.16, -0.11, -0.06, -0.02, 0.03, 0.07, 0.11, 0.16, 0.2, 0.24, 0.29, 0.33, 0.37, 0.42, 0.46, 0.5, 0.54, 0.59, 0.63, 0.67, 0.71, 0.75, 0.79, 0.84, 0.87, 0.92, 0.96, 1.0, 1.04, 1.09,
               1.13, 1.18, 1.22, 1.26, 1.31, 1.36, 1.4, 1.46, 1.5, 1.55, 1.6, 1.65, 1.69, 1.74, 1.8, 1.84, 1.9, 1.95, 2.0, 2.05, 2.11, 2.17, 2.23, 2.27, 2.34, 2.39, 2.46, 2.51, 2.57, 2.63, 2.69,
                2.75, 2.81, 2.88, 3.05, 3.11, 3.15, 3.21, 3.27, -0.25, -0.2, -0.15, -0.1, -0.06, -0.01, 0.04, 0.08, 0.12, 0.16, 0.21, 0.25, 0.29, 0.33, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62,
                 0.67, 0.71, 0.75, 0.79, 0.84, 0.88, 0.91, 0.96, 1.01, 1.04, 1.09, 1.13, 1.17, 1.22, 1.26, 1.31, 1.35, 1.41, 1.44, 1.5, 1.54, 1.6, 1.64, 1.7, 1.74, 1.84, 1.89, 1.95, 2.01, 2.06,
                  2.11, 2.17, 2.22, 2.28, 2.34, 2.39, 2.45, 2.51, 2.57, 2.63, 2.7, 2.75, 2.82, 2.87, 3.03, 3.08, 3.13, 3.18, 3.22, 3.28, -0.26, -0.22, -0.17, -0.12, -0.08, -0.03, 0.01, 0.05, 0.1,
                   0.14, 0.18, 0.22, 0.27, 0.31, 0.35, 0.39, 0.44, 0.48, 0.52, 0.56, 0.6, 0.65, 0.69, 0.72, 0.77, 0.85, 0.89, 0.94, 0.97, 1.01, 1.06, 1.1, 1.15, 1.19, 1.23, 1.29, 1.32, 1.37, 1.42,
                    1.47, 1.52, 1.56, 1.62, 1.67, 1.72, 1.77, 1.86, 1.92, 1.97, 2.03, 2.08, 2.14, 2.19, 2.25, 2.31, 2.36, 2.42, 2.48, 2.53, 2.59, 2.66, 2.71, 2.77, 2.91, 2.98, 3.04, 3.11, 3.17,
                     3.25, 3.31, -0.26, -0.21, -0.16, -0.12, -0.07, -0.03, 0.02, 0.05, 0.1, 0.15, 0.2, 0.23, 0.27, 0.32, 0.35, 0.4, 0.45, 0.48, 0.52, 0.61, 0.65, 0.69, 0.73, 0.77, 0.81, 0.85, 0.89,
                      0.94, 0.97, 1.02, 1.06, 1.1, 1.14, 1.24, 1.3, 1.34, 1.39, 1.43, 1.47, 1.52, 1.56, 1.61, 1.67, 1.71, 1.77, 1.82, 1.87, 1.92, 1.97, 2.02, 2.08, 2.14, 2.19, 2.25, 2.3, 2.36, 2.42,
                       2.48, 2.53, 2.6, 2.65, 2.71, 2.76, 2.83, 2.92, 2.97, 3.04, 3.12, 3.17, 3.24, -0.39, -0.32, -0.28, -0.22, -0.18, -0.13, -0.09, -0.05, -0.0, 0.04, 0.08, 0.13, 0.17, 0.21, 0.25,
                        0.29, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.71, 0.75, 0.79, 0.84, 0.87, 0.91, 0.96, 1.0, 1.04, 1.08, 1.14, 1.2, 1.25, 1.29, 1.34, 1.38, 1.41, 1.46, 1.5,
                         1.54, 1.59, 1.65, 1.7, 1.74, 1.8, 1.84, 1.9, 1.95, 2.01, 2.06, 2.12, 2.18, 2.23, 2.29, 2.35, 2.39, 2.45, 2.51, 2.56, 2.63, 2.7, 2.76, 2.83, 2.89, 2.95, 3.02, 3.08, 3.15,
                          3.21, -0.36, -0.3, -0.25, -0.21, -0.16, -0.11, -0.08, -0.03, 0.01, 0.05, 0.1, 0.14, 0.18, 0.22, 0.26, 0.31, 0.36, 0.39, 0.44, 0.47, 0.51, 0.57, 0.6, 0.65, 0.69, 0.73,
                           0.77, 0.81, 0.85, 0.89, 0.93, 0.96, 1.02, 1.06, 1.1, 1.15, 1.19, 1.23, 1.28, 1.33, 1.37, 1.44, 1.48, 1.52, 1.56, 1.62, 1.67, 1.72, 1.87, 1.93, 1.97, 2.03, 2.07, 2.14,
                            2.19, 2.25, 2.31, 2.36, 2.42, 2.53, 2.59, 2.66, 2.73, 2.79, 2.86, 2.92, 2.98, 3.05, 3.12, 3.18, -0.37, -0.3, -0.24, -0.21, -0.15, -0.11, -0.07, -0.02, 0.02, 0.06, 0.11,
                             0.15, 0.19, 0.23, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.61, 0.65, 0.69, 0.73, 0.77, 0.82, 0.85, 0.9, 0.93, 0.97, 1.01, 1.06, 1.11, 1.15, 1.2, 1.24, 1.29,
                              1.33, 1.38, 1.43, 1.48, 1.52, 1.56, 1.62, 1.67, 1.72, 1.76, 1.82, 1.86, 1.92, 1.98, 2.02, 2.65, 2.72, 2.8, 2.85, 2.91, 2.98, 3.04, 3.12, 3.17, -0.21, -0.16, -0.11,
                               -0.07, -0.02, 0.02, 0.06, 0.1, 0.15, 0.19, 0.23, 0.27, 0.31, 0.36, 0.4, 0.44, 0.48, 0.52, 0.57, 0.61, 0.65, 0.69, 0.72, 0.77, 0.82, 0.85, 0.89, 0.93, 0.97, 1.01,
                                1.07, 1.1, 1.16, 1.2, 1.24, 1.29, 1.33, 1.38, 1.42, 1.47]
    ys30 = [5.62, 5.68, 5.77, 5.83, 5.93, 6.0, 6.08, 6.17, 5.84, 5.74, 5.69, 5.61, 5.56, 5.45, 5.4, 5.38, 5.35, 5.36, 5.36, 5.31, 5.3, 5.37, 5.44, 5.51, 5.57, 5.65, 5.7, 5.77, 5.85, 5.95, 6.02,
     6.1, 6.18, 6.29, 6.37, 6.47, 6.61, 5.65, 5.6, 5.57, 5.52, 5.47, 5.42, 5.41, 5.35, 5.32, 5.28, 5.25, 5.24, 5.18, 5.18, 5.15, 5.1, 5.12, 5.17, 5.25, 5.3, 5.44, 5.49, 5.6, 6.13, 6.18, 6.29, 6.33,
      6.43, 6.53, 6.6, 6.75, 4.95, 4.93, 4.94, 5.46, 5.45, 6.1, 6.18, 6.56, 6.57, 6.63, 6.74, 5.3, 4.86, 4.84, 4.87, 4.94, 5.39, 5.44, 6.1, 6.15, 6.23, 6.58, 6.61, 6.7, 6.83, 5.31, 6.14, 4.79, 4.76,
       4.75, 4.8, 5.47, 5.42, 5.44, 6.04, 6.1, 6.17, 6.23, 6.4, 6.48, 6.59, 6.67, 6.81, 5.13, 5.13, 5.3, 6.11, 4.73, 4.62, 4.68, 4.68, 4.67, 4.63, 4.61, 4.6, 5.32, 5.39, 5.67, 5.53, 5.62, 5.66,
        5.77, 5.81, 5.92, 5.97, 6.08, 6.14, 6.23, 6.27, 6.37, 6.43, 6.54, 6.66, 6.78, 5.45, 5.38, 5.2, 5.18, 5.21, 5.19, 5.2, 5.4, 5.18, 5.04, 5.14, 4.71, 4.73, 4.65, 4.67, 4.67, 4.68, 4.6, 4.56,
         4.56, 4.6, 4.6, 4.61, 4.61, 4.62, 5.01, 5.11, 5.18, 5.24, 5.29, 5.38, 5.41, 5.51, 5.55, 5.62, 5.68, 5.74, 5.83, 5.88, 6.0, 6.07, 6.17, 6.23, 6.32, 6.41, 6.49, 6.6, 5.32, 5.16, 5.03, 4.89,
          4.85, 4.79, 4.76, 4.71, 4.67, 4.67, 4.62, 4.59, 4.58, 4.58, 4.59, 4.57, 4.56, 4.52, 4.53, 4.49, 4.49, 4.47, 4.46, 4.45, 4.46, 4.48, 4.49, 4.51, 4.54, 4.62, 4.65, 4.69, 4.87, 4.9, 4.98,
           5.03, 5.07, 5.17, 5.2, 5.29, 5.34, 5.43, 5.51, 5.55, 5.63, 5.7, 5.78, 5.85, 5.92, 6.02, 6.03, 6.17, 6.2, 6.3, 6.34, 6.46, 6.61, 4.9, 4.87, 4.89, 4.77, 4.69, 4.61, 4.55, 4.46, 4.42,
            4.37, 4.35, 4.32, 4.29, 4.27, 4.26, 4.27, 4.24, 4.21, 4.17, 4.17, 4.14, 4.13, 4.12, 4.11, 4.07, 4.07, 4.08, 4.07, 4.1, 4.11, 4.13, 4.18, 4.22, 4.25, 4.29, 4.33, 4.41, 4.43, 4.51, 4.54,
             4.6, 4.65, 4.85, 4.9, 4.95, 5.02, 5.09, 5.12, 5.15, 5.27, 5.32, 5.41, 5.46, 5.54, 5.58, 5.68, 5.74, 5.82, 5.88, 5.99, 6.02, 6.1, 6.16, 6.25, 6.32, 6.4, 6.55, 4.58, 4.57, 4.47, 4.42,
              4.39, 4.35, 4.26, 4.2, 4.19, 4.13, 4.11, 4.08, 4.09, 4.08, 4.05, 4.03, 3.99, 3.96, 3.94, 3.93, 3.92, 3.9, 3.9, 3.87, 3.89, 3.88, 3.91, 3.91, 3.95, 3.98, 3.98, 4.04, 4.07, 4.13, 4.16,
               4.19, 4.28, 4.29, 4.34, 4.41, 4.46, 4.54, 4.58, 4.63, 4.67, 4.75, 4.79, 4.84, 4.91, 4.95, 5.03, 5.1, 5.17, 5.26, 5.29, 5.4, 5.45, 5.52, 5.58, 5.67, 5.73, 5.82, 5.88, 5.94, 6.04, 6.42,
                6.48, 6.46, 6.49, 6.5, 4.55, 4.45, 4.36, 4.31, 4.26, 4.23, 4.16, 4.11, 4.12, 4.06, 4.02, 4.0, 3.98, 3.94, 3.94, 3.92, 3.89, 3.88, 3.85, 3.83, 3.84, 3.81, 3.83, 3.81, 3.83, 3.82,
                 3.82, 3.84, 3.85, 3.96, 3.91, 3.94, 3.97, 4.0, 4.04, 4.08, 4.12, 4.15, 4.23, 4.26, 4.35, 4.41, 4.47, 4.53, 4.59, 4.65, 4.77, 4.81, 4.85, 4.94, 4.99, 5.09, 5.13, 5.24, 5.29, 5.36,
                  5.42, 5.49, 5.57, 5.63, 5.71, 5.79, 5.85, 5.92, 5.97, 6.63, 6.61, 6.59, 6.55, 6.51, 6.53, 4.51, 4.32, 4.31, 4.23, 4.18, 4.15, 4.08, 4.05, 4.03, 4.02, 3.99, 3.94, 3.92, 3.88, 3.87,
                   3.82, 3.81, 3.81, 3.78, 3.86, 3.96, 3.93, 3.9, 3.84, 3.89, 3.78, 3.82, 3.83, 3.8, 3.84, 3.85, 3.87, 3.93, 3.97, 4.02, 4.1, 4.12, 4.2, 4.23, 4.29, 4.35, 4.43, 4.49, 4.56, 4.62,
                    4.68, 4.79, 4.86, 4.92, 4.99, 5.06, 5.13, 5.2, 5.29, 5.35, 5.42, 5.49, 5.55, 5.62, 5.7, 5.75, 5.83, 5.87, 6.11, 6.18, 6.24, 6.32, 6.46, 6.54, 6.6, 4.48, 4.33, 4.3, 4.21, 4.17,
                     4.11, 4.09, 4.07, 4.0, 3.99, 3.92, 3.95, 3.87, 3.85, 3.85, 3.83, 3.8, 3.81, 3.78, 3.91, 3.87, 3.78, 3.77, 3.77, 3.74, 3.74, 3.75, 3.77, 3.76, 3.79, 3.78, 3.82, 3.87, 4.15, 4.32,
                      4.38, 4.33, 4.3, 4.26, 4.31, 4.33, 4.41, 4.48, 4.57, 4.62, 4.68, 4.75, 4.8, 4.85, 4.94, 4.98, 5.08, 5.12, 5.23, 5.28, 5.37, 5.44, 5.5, 5.58, 5.64, 5.71, 5.78, 5.81, 5.89, 6.06,
                       6.14, 6.23, 6.32, 6.38, 6.46, 4.77, 4.41, 4.37, 4.26, 4.2, 4.16, 4.13, 4.06, 4.03, 3.99, 3.94, 3.92, 3.93, 3.86, 3.84, 3.82, 3.8, 3.78, 3.79, 3.75, 3.75, 3.72, 3.73, 3.74,
                        3.72, 3.77, 3.76, 3.83, 3.87, 3.87, 3.75, 3.75, 3.78, 3.78, 3.86, 4.25, 4.41, 4.46, 4.46, 4.48, 4.44, 4.42, 4.35, 4.34, 4.38, 4.45, 4.54, 4.59, 4.67, 4.72, 4.79, 4.84, 4.92,
                         4.97, 5.06, 5.13, 5.2, 5.27, 5.33, 5.4, 5.44, 5.49, 5.56, 5.62, 5.71, 5.8, 5.89, 6.03, 6.04, 6.2, 6.24, 6.33, 6.39, 6.46, 4.44, 4.3, 4.24, 4.19, 4.17, 4.12, 4.11, 4.08,
                          4.08, 4.05, 4.05, 3.95, 3.98, 3.96, 3.94, 3.88, 3.79, 3.72, 3.71, 3.69, 3.7, 3.69, 3.69, 3.69, 3.68, 3.69, 3.68, 3.7, 3.69, 3.7, 3.7, 3.73, 3.8, 4.0, 3.98, 3.95, 3.99,
                           4.05, 4.09, 4.14, 4.2, 4.44, 4.41, 4.38, 4.42, 4.49, 4.59, 4.67, 4.84, 4.88, 4.93, 5.0, 5.04, 5.14, 5.19, 5.26, 5.34, 5.37, 5.46, 5.58, 5.65, 5.75, 5.9, 5.97, 6.04, 6.17,
                            6.21, 6.28, 6.41, 6.43, 4.72, 4.36, 4.27, 4.18, 4.16, 4.08, 4.05, 4.03, 3.96, 3.94, 3.9, 3.87, 3.85, 3.82, 3.8, 3.77, 3.76, 3.74, 3.72, 3.71, 3.71, 3.7, 3.69, 3.69,
                             3.68, 3.69, 3.69, 3.7, 3.69, 3.7, 3.71, 3.73, 3.76, 3.87, 3.93, 3.9, 3.96, 4.04, 4.08, 4.16, 4.2, 4.23, 4.35, 4.4, 4.39, 4.46, 4.54, 4.61, 4.65, 4.7, 4.77, 4.8, 4.88,
                              4.94, 5.72, 5.86, 5.99, 5.99, 6.08, 6.17, 6.23, 6.31, 6.4, 4.21, 4.19, 4.12, 4.08, 4.04, 4.02, 3.99, 3.96, 3.89, 3.88, 3.86, 3.82, 3.81, 3.78, 3.72, 3.72, 3.7, 3.72,
                               3.68, 3.69, 3.69, 3.69, 3.7, 3.69, 3.7, 3.68, 3.74, 3.71, 3.74, 3.84, 3.83, 3.84, 3.97, 4.02, 4.05, 4.08, 4.14, 4.27, 4.26, 4.28]
    
    xs31 = [-0.26, -0.22, -0.18, -0.15, -0.1, -0.06, 0.0, 0.07, 0.1, 0.43, 0.51, -0.7, -0.66, -0.62, -0.59, -0.55, -0.51, -0.48, -0.43, -0.39, -0.36, -0.33, -0.29, -0.26, -0.21, -0.17, -0.13, -0.09, -0.04, -0.01, 0.03, 0.07, 0.12, 0.14, 0.19, 0.24, 0.29, 0.35, -0.12, -0.08, -0.03, -0.0, 0.04, 0.09, 0.12, 0.16, 0.19, 0.24, 0.3, 0.39, 0.46, 0.56, 0.65, -0.88, -0.84, -0.8, -0.76, -0.73, -0.69, -0.65, -0.61, -0.57, -0.54, -0.5, -0.46, -0.42, -0.38, -0.34, -0.3, -0.27, -0.23, -0.19, -0.16, -0.43, -0.38, -0.35, -0.31, -0.28, -0.25, -0.2, -0.16, -0.13, -0.09, -0.05, -0.01, 0.04, 0.08, 0.11, 0.15, 0.18, 0.22, 0.37, 0.43, 0.52, 0.62, 0.71, 0.83, -0.99, -0.92, -0.87, -0.83, -0.79, -0.75, -0.73, -0.7, -0.65, -0.62, -0.58, -0.53, -0.5, -0.46, 0.33, 0.4, 0.47, 0.58, 0.67, 0.78, -0.04, -0.01, 0.03, 0.06, 0.11, 0.14, 0.19, -1.1, -1.05, -1.01, -0.97, -0.94, -0.87, -0.85, -0.8, 0.1, 0.3, 0.38, 0.46, 0.54, 0.61, 0.74, 0.86, -0.76, -0.73, -0.7, -0.67, -0.61, -0.58, -0.54, -0.5, -0.48, -0.44, -0.39, -0.36, -0.33, -0.3, -0.24, -0.21, -0.18, -0.14, -0.92, -0.85, -0.77, 0.7, 0.78, 0.86, 0.97, -0.36, -0.25, -0.18, -0.14, -0.02, 0.11, 0.16, 0.18, 0.23, 0.09, 0.11, 0.16, 0.3, 0.33, -0.91, -0.79, -0.85, -0.8, -0.72, -0.67, -0.67, -0.62, -0.58, -0.56, -0.53, -0.46, -0.42, -1.2, -1.13, -1.09, -1.05, -1.03, -0.98, -0.94, 0.12, 0.22, 0.25, 0.29, 0.33, 0.72, 0.8, 0.91, -0.47, -0.38, -0.35, -0.31, -0.26, -0.24, -0.19, -0.07, -0.06, -0.02, 0.0, 0.04, 0.08, -1.01, -0.97, -0.84, -0.89, -0.79, -0.64, -0.6, -0.53, -1.29, -1.27, -1.19, -1.15, -1.11, -1.07, -0.14, -0.09, -0.05, -0.02, 0.01, 0.03, 0.07, 0.12, 0.23, 0.25, 0.29, 0.33, 0.77, 0.83, -0.65, -0.62, -0.58, -0.55, -0.5, -0.48, -0.42, -0.39, -0.36, -0.33, -0.29, -0.25, -0.22, -0.16, -1.12, -1.08, -1.04, -1.01, -0.98, -0.94, -0.91, -0.86, -0.84, -0.79, -0.75, -0.72, -0.69, -1.33, -1.3, -1.23, -1.19, -1.15, 0.5, 0.56, 0.78, 0.9, 0.99, -0.31, -0.29, -0.24, -0.2, -0.16, -0.13, -0.08, -0.05, -0.02, 0.02, 0.07, 0.12, 0.24, 0.3, 0.37, 0.43, -0.77, -0.74, -0.71, -0.67, -0.63, -0.59, -0.57, -0.52, -0.5, -0.45, -0.42, -0.39, -0.34, -1.11, -1.09, -1.05, -1.01, -0.98, -0.95, -0.93, -0.88, -0.85, -0.81, -1.33, -1.3, -1.26, -1.23, -1.18, -1.16, 0.05, 0.1, 0.22, 0.27, 0.34, 0.4, 0.47, 0.53, 0.64, 0.88, 0.96, -0.47, -0.44, -0.4, -0.37, -0.32, -0.29, -0.25, -0.22, -0.2, -0.15, -0.12, -0.09, -0.04, 0.01, -0.85, -0.82, -0.79, -0.76, -0.73, -0.68, -0.65, -0.62, -0.58, -0.54, -0.51, -1.19, -1.16, -1.13, -1.09, -1.06, -1.03, -0.99, -0.96, -0.92, -0.89, -1.37, -1.33, -1.3, -1.27, -1.24, 0.74, 0.85, 1.02, -0.19, -0.15, -0.11, -0.08, -0.03, 0.02, 0.07, 0.13, 0.17, 0.23, 0.38, 0.45, 0.54, 0.64, -0.6, -0.58, -0.55, -0.51, -0.48, -0.44, -0.41, -0.37, -0.34, -0.31, -0.27, -0.23, -1.0, -0.95, -0.92, -0.89, -0.86, -0.82, -0.8, -0.75, -0.71, -0.67, -0.66, -1.29, -1.26, -1.23, -1.19, -1.15, -1.13, -1.09, -1.06, -1.03, -1.43, -1.4, -1.35, -1.33, 0.4, 0.48, 0.57, 0.68, 0.81, 0.92, 1.05, -0.32, -0.26, -0.23, -0.2, -0.16, -0.13, -0.08, -0.04, 0.0, 0.05, 0.09, 0.18, 0.25, -0.68, -0.65, -0.62, -0.59, -0.55, -0.53, -0.49, -0.46, -0.42, -0.38, -0.35, -1.02, -0.99, -0.96, -0.92, -0.9, -0.86, -0.82, -0.78, -0.75, -0.72, -1.29, -1.26, -1.22, -1.19, -1.14, -1.12, -1.09, -1.06, -1.45, -1.42, -1.39, -1.35, -1.32, -0.01, 0.03, 0.08, 0.14, 0.21, 0.31, 0.38, 0.47, 0.55, 0.65, 0.77, 0.89, 1.03, 1.11, -0.46, -0.43, -0.39, -0.36, -0.34, -0.28, -0.24, -0.21, -0.18, -0.14, -0.08, -0.05, -0.8, -0.76, -0.72, -0.69, -0.66, -0.63, -0.6, -0.56, -0.53, -0.5, -1.09, -1.05, -1.03, -1.0, -0.95, -0.92, -0.91, -0.86, -0.83, -1.36, -1.32, -1.29, -1.25, -1.22, -1.19, -1.16, -1.12, -1.46, -1.43, -1.39, 1.03, -0.04, 0.01, 0.08, 0.17, 0.22, 0.31, 0.39, 0.49, 0.62, 0.71, 0.83, 0.9, -0.4, -0.38, -0.34, -0.31, -0.27, -0.24, -0.19, -0.16, -0.11, -0.07, -0.7, -0.68, -0.65, -0.61, -0.59, -0.54, -0.51, -0.47, -0.44, -0.96, -0.93, -0.91, -0.87, -0.84, -0.8, -0.77, -0.75, -1.19, -1.16, -1.12, -1.1, -1.06, -1.03, -0.99, -1.43, -1.39, -1.37, -1.33, -1.29, -1.26, -1.23, -1.46, -0.04, 0.02, 0.1, 0.17, 0.26, 0.35, 0.44, 0.74, -0.38, -0.34, -0.31, -0.27, -0.24, -0.19, -0.16, -0.12, -0.08, -0.65, -0.62, -0.59, -0.54, -0.51, -0.48, -0.44, -0.41, -0.88, -0.84, -0.82, -0.78, -0.74, -0.71, -0.68, -1.1, -1.07, -1.04, -1.0, -0.97, -0.94, -0.91, -1.29, -1.26, -1.23, -1.2, -1.17, -1.13, -1.47, -1.42, -1.4, -1.37, -1.34, -0.06, -0.0, 0.04, 0.11, 0.16, 0.41, -0.37, -0.32, -0.29, -0.25, -0.21, -0.17, -0.15, -0.1, -0.6, -0.56, -0.53, -0.51, -0.46, -0.43, -0.39, -0.82, -0.8, -0.73, -0.7, -0.68, -0.65, -0.62, -1.02, -0.99, -0.96, -0.91, -0.89, -0.86, -1.18, -1.15, -1.12, -1.09, -1.05, -1.35, -1.3, -1.27, -1.25, -1.22, -1.48, -1.44, -1.42, -1.38]
    ys31 = [5.15, 5.14, 5.14, 5.1, 5.1, 5.07, 5.25, 5.07, 5.05, 5.0, 5.22, 4.99, 4.95, 4.9, 4.83, 4.77, 4.76, 4.73, 4.7, 4.7, 4.64, 4.66, 4.63, 4.62, 4.6, 4.6, 4.58, 4.58, 4.57, 4.55, 4.58, 4.54, 4.57, 4.58, 4.62, 4.62, 4.66, 4.8, 4.37, 4.38, 4.37, 4.36, 4.37, 4.36, 4.38, 4.36, 4.35, 4.41, 4.52, 4.74, 4.96, 5.22, 5.45, 4.78, 4.74, 4.69, 4.7, 4.64, 4.62, 4.59, 4.59, 4.55, 4.55, 4.5, 4.5, 4.47, 4.46, 4.47, 4.44, 4.41, 4.41, 4.42, 4.4, 4.29, 4.31, 4.26, 4.28, 4.28, 4.21, 4.19, 4.18, 4.21, 4.2, 4.2, 4.19, 4.25, 4.27, 4.21, 4.19, 4.17, 4.16, 4.53, 4.7, 4.95, 5.2, 5.49, 5.8, 4.62, 4.58, 4.6, 4.6, 4.71, 4.77, 4.43, 4.41, 4.38, 4.35, 4.31, 4.31, 4.31, 4.32, 4.43, 4.66, 4.79, 5.08, 5.39, 5.73, 4.07, 4.05, 4.07, 4.07, 4.1, 4.03, 4.03, 4.53, 4.51, 4.51, 4.54, 4.48, 4.7, 4.67, 4.73, 4.04, 4.26, 4.46, 4.66, 4.86, 5.14, 5.45, 5.79, 4.14, 4.13, 4.12, 4.11, 4.11, 4.07, 4.08, 4.07, 4.04, 4.04, 4.0, 4.02, 3.99, 3.98, 4.0, 3.98, 3.96, 3.96, 4.57, 4.68, 5.22, 5.15, 5.35, 5.56, 5.83, 3.82, 3.8, 3.79, 3.85, 4.26, 5.0, 4.95, 4.92, 4.89, 3.82, 3.76, 3.8, 4.14, 4.15, 4.05, 5.22, 4.03, 4.04, 4.03, 4.02, 3.89, 3.9, 3.87, 3.83, 3.84, 4.02, 4.05, 4.26, 4.26, 4.28, 4.16, 4.06, 4.13, 4.1, 3.71, 3.78, 3.78, 3.78, 3.78, 5.16, 5.36, 5.66, 4.03, 3.8, 3.82, 3.84, 3.88, 3.82, 3.82, 3.93, 3.74, 3.71, 3.69, 3.73, 3.67, 3.99, 3.96, 5.23, 3.9, 3.9, 3.88, 3.77, 3.87, 4.35, 4.17, 4.16, 4.13, 4.09, 4.01, 3.71, 3.74, 3.77, 3.66, 3.66, 3.59, 3.58, 3.65, 3.81, 3.78, 3.8, 3.79, 5.21, 5.31, 3.81, 3.76, 3.74, 3.74, 3.73, 3.73, 3.77, 3.75, 3.73, 3.73, 3.71, 3.72, 3.7, 3.72, 4.05, 4.08, 4.01, 3.91, 3.87, 3.88, 3.84, 3.84, 3.83, 3.8, 3.83, 3.79, 3.8, 4.35, 4.29, 4.12, 4.09, 4.06, 4.42, 4.5, 5.13, 5.49, 5.73, 3.37, 3.37, 3.38, 3.39, 3.4, 3.4, 3.42, 3.41, 3.44, 3.42, 3.5, 3.55, 3.82, 3.96, 4.08, 4.23, 3.45, 3.4, 3.41, 3.39, 3.37, 3.37, 3.37, 3.38, 3.4, 3.4, 3.39, 3.41, 3.39, 3.67, 3.65, 3.64, 3.6, 3.58, 3.52, 3.48, 3.47, 3.48, 3.46, 3.87, 3.82, 3.8, 3.76, 3.73, 3.74, 3.37, 3.46, 3.68, 3.79, 3.92, 4.04, 4.23, 4.34, 4.63, 5.34, 5.54, 3.25, 3.22, 3.24, 3.22, 3.23, 3.24, 3.22, 3.23, 3.24, 3.23, 3.21, 3.24, 3.27, 3.31, 3.3, 3.29, 3.27, 3.26, 3.26, 3.24, 3.23, 3.23, 3.22, 3.24, 3.24, 3.58, 3.54, 3.5, 3.47, 3.46, 3.43, 3.38, 3.35, 3.31, 3.31, 3.75, 3.68, 3.64, 3.65, 3.61, 5.06, 5.4, 5.95, 3.09, 3.1, 3.15, 3.18, 3.23, 3.32, 3.38, 3.48, 3.58, 3.72, 4.02, 4.24, 4.45, 4.75, 3.08, 3.09, 3.07, 3.06, 3.07, 3.08, 3.07, 3.07, 3.06, 3.06, 3.05, 3.04, 3.14, 3.12, 3.12, 3.11, 3.09, 3.09, 3.07, 3.07, 3.06, 3.07, 3.06, 3.4, 3.37, 3.34, 3.3, 3.27, 3.26, 3.24, 3.18, 3.14, 3.55, 3.45, 3.44, 3.44, 4.05, 4.3, 4.55, 4.87, 5.29, 5.6, 6.02, 2.9, 2.92, 2.94, 2.95, 2.96, 3.0, 3.04, 3.08, 3.12, 3.18, 3.27, 3.45, 3.68, 2.89, 2.91, 2.88, 2.88, 2.87, 2.89, 2.87, 2.87, 2.86, 2.86, 2.9, 2.98, 2.99, 2.94, 2.94, 2.92, 2.91, 2.9, 2.89, 2.91, 2.9, 3.16, 3.1, 3.1, 3.07, 3.07, 3.02, 3.01, 3.0, 3.3, 3.24, 3.2, 3.19, 3.16, 3.0, 3.07, 3.12, 3.26, 3.48, 3.75, 3.96, 4.19, 4.43, 4.73, 5.12, 5.44, 5.87, 6.06, 2.78, 2.78, 2.78, 2.82, 2.77, 2.84, 2.85, 2.86, 2.85, 2.87, 2.96, 2.98, 2.79, 2.79, 2.77, 2.77, 2.78, 2.78, 2.78, 2.77, 2.79, 2.75, 2.92, 2.91, 2.88, 2.87, 2.85, 2.84, 2.83, 2.82, 2.81, 3.07, 3.03, 3.04, 2.99, 2.97, 2.95, 2.94, 2.92, 3.21, 3.14, 3.1, 5.9, 2.88, 2.93, 3.14, 3.38, 3.53, 3.78, 4.03, 4.32, 4.7, 5.03, 5.41, 5.54, 2.64, 2.68, 2.67, 2.66, 2.69, 2.7, 2.72, 2.76, 2.81, 2.84, 2.63, 2.6, 2.58, 2.6, 2.61, 2.62, 2.64, 2.68, 2.64, 2.8, 2.77, 2.72, 2.71, 2.71, 2.69, 2.68, 2.64, 2.87, 2.87, 2.86, 2.79, 2.83, 2.8, 2.76, 2.95, 2.91, 2.84, 2.85, 2.84, 2.89, 2.88, 3.01, 2.89, 3.01, 3.28, 3.44, 3.74, 3.99, 4.27, 5.63, 2.66, 2.65, 2.66, 2.68, 2.7, 2.72, 2.76, 2.79, 2.8, 2.58, 2.59, 2.58, 2.61, 2.6, 2.6, 2.62, 2.62, 2.64, 2.65, 2.63, 2.62, 2.58, 2.6, 2.59, 2.7, 2.72, 2.68, 2.7, 2.66, 2.66, 2.65, 2.8, 2.78, 2.77, 2.74, 2.74, 2.73, 2.93, 2.89, 2.86, 2.83, 2.81, 2.82, 2.96, 2.96, 3.23, 3.36, 4.46, 2.67, 2.68, 2.66, 2.7, 2.73, 2.72, 2.74, 2.78, 2.57, 2.58, 2.59, 2.59, 2.62, 2.63, 2.67, 2.65, 2.65, 2.83, 2.82, 2.77, 2.74, 2.75, 2.73, 2.72, 2.68, 2.69, 2.66, 2.69, 2.79, 2.82, 2.83, 2.77, 2.73, 2.86, 2.93, 2.94, 2.97, 2.9, 2.99, 2.92, 2.87, 2.88]
    
    time_start = time.time()
    
    # 导入测试数据
    # 最佳效果：测试数据10，25，30
    pts = np.array([xs31, ys31]).T
    print('pts', pts.shape)
    
    #~ # 调整角度及位置
    #~ pts[:, 0], pts[:, 1] = transform_2d_point_clouds(pts[:, 0], pts[:, 1], -0.2, 0, 0)
    
    method = 1
    
    # 拟合包围盒，方法一
    if method == 1:
        x0, y0, l, w, phi, polygon, polygon_approximated, polygon_approximated_is_closed = fit_2d_obb(pts[:, 0], pts[:, 1])
        
    # 拟合包围盒，方法二（迭代计算）
    elif method == 2:
        x0, y0, l, w, phi, polygon, polygon_approximated, polygon_approximated_is_closed = fit_2d_obb_using_iterative_procedure(pts[:, 0], pts[:, 1])
    
    # 测试结果
    time_cost = time.time() - time_start
    print('time cost of the process:', round(time_cost, 6), 's')
    print()
    
    print('x0', round(x0, 3), 'm')
    print('y0', round(y0, 3), 'm')
    print('l', round(l, 3), 'm')
    print('w', round(w, 3), 'm')
    print('phi', round(phi, 3), 'rad')
    print()
    
    #~ # 打印点云数据
    #~ print('points_x')
    #~ pts_x = pts[:, 0]
    #~ pts_x = np.around(pts_x, 2)
    #~ pts_x = list(pts_x)
    #~ print(pts_x)
    #~ print('points_y')
    #~ pts_y = pts[:, 1]
    #~ pts_y = np.around(pts_y, 2)
    #~ pts_y = list(pts_y)
    #~ print(pts_y)
    
    # 在画布中创建1行6列的坐标轴实例，画布大小18inch*4inch
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    
    # 第1坐标系，绘制原始点云
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].axis('equal')
    axes[0].scatter(pts[:, 0], pts[:, 1], color='gray', s=0.1)
    
    # 第2坐标系，绘制点云轮廓
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].axis('equal')
    axes[1].scatter(pts[:, 0], pts[:, 1], color='gray', s=0.1)
    for i in range(polygon.shape[0]):
        x1 = polygon[i, 0, 0]
        y1 = polygon[i, 0, 1]
        x2 = polygon[(i + 1) % polygon.shape[0], 0, 0]
        y2 = polygon[(i + 1) % polygon.shape[0], 0, 1]
        axes[1].plot([x1, x2], [y1, y2], color='blue', linewidth=1)
    
    # 第3坐标系，绘制近似多边形
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].axis('equal')
    axes[2].scatter(pts[:, 0], pts[:, 1], color='gray', s=0.1)
    n = polygon_approximated.shape[0]
    num = n if polygon_approximated_is_closed else n - 1
    for i in range(num):
        x1 = polygon_approximated[i, 0, 0]
        y1 = polygon_approximated[i, 0, 1]
        x2 = polygon_approximated[(i + 1) % n, 0, 0]
        y2 = polygon_approximated[(i + 1) % n, 0, 1]
        axes[2].plot([x1, x2], [y1, y2], color='red', linewidth=1)
    
    # 第4坐标系，绘制包围盒
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].axis('equal')
    axes[3].scatter(pts[:, 0], pts[:, 1], color='gray', s=0.1)
    
    # 计算包围盒顶点坐标
    obb_xs, obb_ys = compute_obb_vertex_coordinates(x0, y0, l, w, phi)
    
    for i in range(4):
        x1 = obb_xs[i]
        y1 = obb_ys[i]
        x2 = obb_xs[(i + 1) % 4]
        y2 = obb_ys[(i + 1) % 4]
        axes[3].plot([x1, x2], [y1, y2], color='k', linewidth=1)
    
    # 第5坐标系，绘制最小包围圆
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    axes[4].axis('equal')
    axes[4].scatter(pts[:, 0], pts[:, 1], color='gray', s=0.1)
    
    # 提取最小包围圆
    xmec, ymec, radius = find_min_enclosing_circle_using_minEnclosingCircle(pts[:, 0], pts[:, 1])
    
    theta = np.linspace(0, 2 * PI, 200)
    x = radius * np.cos(theta) + xmec
    y = radius * np.sin(theta) + ymec
    axes[4].plot(x, y, color="darkred", linewidth=2)
    
    # 第6坐标系，绘制投影点、局部坐标系、辅助线
    axes[5].set_xticks([])
    axes[5].set_yticks([])
    axes[5].axis('equal')
    axes[5].scatter(pts[:, 0], pts[:, 1], color='gray', s=0.1)
    
    # 计算另一边方向角
    phi_second = phi + PI / 2 if phi < PI / 2 else phi - PI / 2
    
    # 将点云投影至局部坐标系
    _, _, xp1, yp1, xp2, yp2 = project_point_clouds_on_line(pts[:, 0], pts[:, 1], 0, 0, phi)
    _, _, xp3, yp3, xp4, yp4 = project_point_clouds_on_line(pts[:, 0], pts[:, 1], 0, 0, phi_second)
    
    # 绘制投影点
    axes[5].scatter(xp1, yp1, color='black', s=1)
    axes[5].scatter(xp2, yp2, color='black', s=1)
    axes[5].scatter(xp3, yp3, color='black', s=1)
    axes[5].scatter(xp4, yp4, color='black', s=1)
    
    # 绘制局部坐标系
    axes[5].plot([0, xp1], [0, yp1], color='black', linewidth=1)
    axes[5].plot([0, xp2], [0, yp2], color='black', linewidth=1)
    axes[5].plot([0, xp3], [0, yp3], color='black', linewidth=1)
    axes[5].plot([0, xp4], [0, yp4], color='black', linewidth=1)
    
    # 计算投影点与原点的距离，选出近点和远点
    dd1 = xp1 ** 2 + yp1 ** 2
    dd2 = xp2 ** 2 + yp2 ** 2
    
    if dd1 < dd2:
        xpl_near, ypl_near = xp1, yp1
        xpl_far, ypl_far = xp2, yp2
    else:
        xpl_near, ypl_near = xp2, yp2
        xpl_far, ypl_far = xp1, yp1
        
    dd3 = xp3 ** 2 + yp3 ** 2
    dd4 = xp4 ** 2 + yp4 ** 2
    
    if dd3 < dd4:
        xpw_near, ypw_near = xp3, yp3
        xpw_far, ypw_far = xp4, yp4
    else:
        xpw_near, ypw_near = xp4, yp4
        xpw_far, ypw_far = xp3, yp3
    
    # 根据近点和远点计算包围盒顶点
    xnn = xpl_near + xpw_near
    ynn = ypl_near + ypw_near
    
    xnf = xpl_near + xpw_far
    ynf = ypl_near + ypw_far
    
    xfn = xpl_far + xpw_near
    yfn = ypl_far + ypw_near
    
    # 绘制辅助线
    axes[5].plot([xnn, xpl_near], [ynn, ypl_near], color='darkred', linewidth=0.5, linestyle='--')
    axes[5].plot([xnn, xpw_near], [ynn, ypw_near], color='darkred', linewidth=0.5, linestyle='--')
    axes[5].plot([xnf, xpw_far], [ynf, ypw_far], color='darkred', linewidth=0.5, linestyle='--')
    axes[5].plot([xfn, xpl_far], [yfn, ypl_far], color='darkred', linewidth=0.5, linestyle='--')
    
    # 绘制包围盒
    for i in range(4):
        x1 = obb_xs[i]
        y1 = obb_ys[i]
        x2 = obb_xs[(i + 1) % 4]
        y2 = obb_ys[(i + 1) % 4]
        axes[5].plot([x1, x2], [y1, y2], color='k', linewidth=1)
    
    # 去除顶部和右侧的框线，spines是一个字典，其中的键包括right、left、top、bottom
    axes[5].spines['right'].set_color('none')
    axes[5].spines['top'].set_color('none')

    # 将底部和左侧的框线移到x=0、y=0处
    axes[5].spines['bottom'].set_position(('data', 0))
    axes[5].spines['left'].set_position(('data', 0))
    
    # 保存图片，dpi单位为点/inch
    fig.savefig('result.png', dpi=200)
    
    plt.show()
