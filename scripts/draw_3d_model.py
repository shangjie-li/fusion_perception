# -*- coding: UTF-8 -*-

import numpy as np
import math

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

PI = 3.14159

def project_point_clouds(xyz, mat, height, width, crop=True):
    # 功能：将三维点云投影至图像
    # 输入：xyz <class 'numpy.ndarray'> (n, 4) 代表三维点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      mat <class 'numpy.ndarray'> (3, 4) 代表投影矩阵
    #      height <class 'int'> 图像高度
    #      width <class 'int'> 图像宽度
    #      crop <class 'bool'> 是否裁剪掉超出图像的部分
    # 输出：clouds_xyz <class 'numpy.ndarray'> (n, 4) 代表三维点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      clouds_uv <class 'numpy.ndarray'> (n, 2) 代表图像坐标[u, v]，n为点的数量
    
    uv = mat.dot(xyz.T).T
    uv = np.true_divide(uv[:, :2], uv[:, [-1]])
    
    if not crop:
        clouds_xyz = xyz
        clouds_uv = uv
    else:
        idxs = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        clouds_xyz = xyz[idxs]
        clouds_uv = uv[idxs]
    
    return clouds_xyz, clouds_uv

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

def test_point_in_image(uv, height, width):
    # 功能：测试点是否在图像中
    # 输入：uv <class 'numpy.ndarray'> (n, 2) 代表图像坐标[u, v]，n为点的数量
    #      height <class 'int'> 图像高度
    #      width <class 'int'> 图像宽度
    # 输出：idxs <class 'numpy.ndarray'> (n,) 点在图像中为True，否则为False
    
    idxs = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    
    return idxs

def find_boundary_close_to_origin_in_polygon(polygon):
    # 功能：保留多边形各顶点中靠近坐标原点的部分
    # 输入：polygon <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    # 输出：polygon_remained <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    #      polygon_the_other <class 'numpy.ndarray'> (n, 1, 2) n为轮廓点数
    
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
        the_other_idxs = list(range(min_idx + 1, num)) + list(range(0, max_idx)) if min_idx < num else list(range(0, max_idx))
    else:
        remain_idxs = list(range(max_idx, num)) + list(range(0, min_idx + 1))
        the_other_idxs = list(range(min_idx + 1, max_idx))
    
    polygon_remained = polygon_sorted[remain_idxs, :, :]
    polygon_the_other = polygon_sorted[the_other_idxs, :, :]
    
    return polygon_remained, polygon_the_other

def draw_3d_model(img, polygon, height, mat, color=(255, 0, 0), thickness=1):
    # 功能：绘制目标三维边界框
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      polygon <class 'numpy.ndarray'> (n, 1, 3) n为轮廓点数，代表底面多边形
    #      height <class 'float'> 代表高度
    #      mat <class 'numpy.ndarray'> (3, 4) 代表投影矩阵
    #      color <class 'tuple'> 边界框颜色
    #      thickness <class 'int'> 边界框宽度
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      display <class 'bool'> 显示边界框标志位
    
    frame_height = img.shape[0]
    frame_width = img.shape[1]
    
    # 当三维边界框的顶点超出图像时，取消绘制
    img_copy = img.copy()
    display = True
    
    # 将多边形各顶点按逆时针方向排序
    polygon = sort_polygon(polygon)
    
    # 提取多边形各顶点中靠近坐标原点的部分和远离坐标原点的部分
    poc, pof = find_boundary_close_to_origin_in_polygon(polygon)
    
    # STEP1
    # 绘制目标侧面的竖边，由于视线遮挡，只绘制靠近坐标原点一侧的竖边
    num = poc.shape[0]
    for i in range(num):
        xyz = np.array([[poc[i, 0, 0], poc[i, 0, 1], poc[i, 0, 2], 1]])
        _, uv_1 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
        if not test_point_in_image(uv_1, frame_height, frame_width)[0]: display = False
        
        xyz = np.array([[poc[i, 0, 0], poc[i, 0, 1], poc[i, 0, 2] + height, 1]])
        _, uv_2 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
        if not test_point_in_image(uv_2, frame_height, frame_width)[0]: display = False
        
        pt_1 = (int(uv_1[0, 0]), int(uv_1[0, 1]))
        pt_2 = (int(uv_2[0, 0]), int(uv_2[0, 1]))
        cv2.line(img, pt_1, pt_2, color, thickness)
    
    # STEP2
    # 绘制目标底面的横边，由于视线遮挡，只绘制靠近坐标原点一侧的横边
    num = poc.shape[0]
    for i in range(num - 1):
        xyz = np.array([[poc[i, 0, 0], poc[i, 0, 1], poc[i, 0, 2], 1]])
        _, uv_1 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
        if not test_point_in_image(uv_1, frame_height, frame_width)[0]: display = False
        
        xyz = np.array([[poc[i + 1, 0, 0], poc[i + 1, 0, 1], poc[i + 1, 0, 2], 1]])
        _, uv_2 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
        if not test_point_in_image(uv_2, frame_height, frame_width)[0]: display = False
        
        pt_1 = (int(uv_1[0, 0]), int(uv_1[0, 1]))
        pt_2 = (int(uv_2[0, 0]), int(uv_2[0, 1]))
        cv2.line(img, pt_1, pt_2, color, thickness)
    
    # STEP3
    # 绘制目标顶面的横边，判断是否需要绘制远离坐标原点一侧的横边，如果是则绘制
    flag = False
    
    xyz = np.array([[poc[0, 0, 0], poc[0, 0, 1], poc[0, 0, 2] + height, 1]])
    _, uv_1 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
    if not test_point_in_image(uv_1, frame_height, frame_width)[0]: display = False
    
    xyz = np.array([[poc[-1, 0, 0], poc[-1, 0, 1], poc[-1, 0, 2] + height, 1]])
    _, uv_2 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
    if not test_point_in_image(uv_2, frame_height, frame_width)[0]: display = False
    
    slope = (uv_2[0, 1] - uv_1[0, 1]) / (uv_2[0, 0] - uv_1[0, 0])
    
    # 从靠近坐标原点一侧的点判断
    num = poc.shape[0]
    for i in range(num):
        if i != 0 and i != num - 1:
            xyz = np.array([[poc[i, 0, 0], poc[i, 0, 1], poc[i, 0, 2] + height, 1]])
            _, uv = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
            if not test_point_in_image(uv, frame_height, frame_width)[0]: display = False
            du = uv[0, 0] - uv_1[0, 0]
            dv = uv[0, 1] - uv_1[0, 1]
            if dv > slope * du: flag = True
    
    # 从远离坐标原点一侧的点判断
    num = pof.shape[0]
    for i in range(num):
        if i != 0 and i != num - 1:
            xyz = np.array([[poc[i, 0, 0], poc[i, 0, 1], poc[i, 0, 2] + height, 1]])
            _, uv = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
            if not test_point_in_image(uv, frame_height, frame_width)[0]: display = False
            du = uv[0, 0] - uv_1[0, 0]
            dv = uv[0, 1] - uv_1[0, 1]
            if dv < slope * du: flag = True
    
    if flag:
        num = polygon.shape[0]
        for i in range(num):
            xyz = np.array([[polygon[i, 0, 0], polygon[i, 0, 1], polygon[i, 0, 2] + height, 1]])
            _, uv_1 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
            if not test_point_in_image(uv_1, frame_height, frame_width)[0]: display = False
            
            xyz = np.array([[polygon[(i + 1) % num, 0, 0], polygon[(i + 1) % num, 0, 1], polygon[(i + 1) % num, 0, 2] + height, 1]])
            _, uv_2 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
            if not test_point_in_image(uv_2, frame_height, frame_width)[0]: display = False
            
            pt_1 = (int(uv_1[0, 0]), int(uv_1[0, 1]))
            pt_2 = (int(uv_2[0, 0]), int(uv_2[0, 1]))
            cv2.line(img, pt_1, pt_2, color, thickness)
    else:
        num = poc.shape[0]
        for i in range(num - 1):
            xyz = np.array([[poc[i, 0, 0], poc[i, 0, 1], poc[i, 0, 2] + height, 1]])
            _, uv_1 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
            if not test_point_in_image(uv_1, frame_height, frame_width)[0]: display = False
            
            xyz = np.array([[poc[i + 1, 0, 0], poc[i + 1, 0, 1], poc[i + 1, 0, 2] + height, 1]])
            _, uv_2 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
            if not test_point_in_image(uv_2, frame_height, frame_width)[0]: display = False
            
            pt_1 = (int(uv_1[0, 0]), int(uv_1[0, 1]))
            pt_2 = (int(uv_2[0, 0]), int(uv_2[0, 1]))
            cv2.line(img, pt_1, pt_2, color, thickness)
    
    if display:
        return img, display
    else:
        return img_copy, display

