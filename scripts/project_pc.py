# -*- coding: UTF-8 -*-

import numpy as np
import math

def transform_3d_point_clouds(xyz, mat):
    # 功能：对三维点云进行旋转、平移
    # 输入：xyz <class 'numpy.ndarray'> (n, 4) 代表齐次坐标[x, y, z, 1]，n为点的数量
    #      mat <class 'numpy.ndarray'> (4, 4) 代表包括旋转、平移的转换矩阵
    # 输出：xyz_new <class 'numpy.ndarray'> (n, 4) 代表旋转、平移后的齐次坐标[x, y, z, 1]，n为点的数量
    
    xyz_new = mat.dot(xyz.T).T
    
    return xyz_new

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

def limit_pc_view(xyz, area_number, fov_angle):
    # 功能：限制点云视场角度
    # 输入：xyz <class 'numpy.ndarray'> (n, 4) 代表齐次坐标[x, y, z, 1]，n为点的数量
    #      area_number <class 'int'> 代表点云视场区域编号，1为x正向，2为y负向，3为x负向，4为y正向
    #      fov_angle <class 'float'> 代表水平视场角
    # 输出：xyz <class 'numpy.ndarray'> (n, 4) 代表齐次坐标[x, y, z, 1]，n为点的数量
    
    alpha = 90 - 0.5 * fov_angle
    k = math.tan(alpha * math.pi / 180.0)
    if area_number == 1:
        xyz = xyz[np.logical_and((xyz[:, 0] > k * xyz[:, 1]), (xyz[:, 0] > -k * xyz[:, 1]))]
    elif area_number == 2:
        xyz = xyz[np.logical_and((-xyz[:, 1] > k * xyz[:, 0]), (-xyz[:, 1] > -k * xyz[:, 0]))]
    elif area_number == 3:
        xyz = xyz[np.logical_and((-xyz[:, 0] > k * xyz[:, 1]), (-xyz[:, 0] > -k * xyz[:, 1]))]
    elif area_number == 4:
        xyz = xyz[np.logical_and((xyz[:, 1] > k * xyz[:, 0]), (xyz[:, 1] > -k * xyz[:, 0]))]
        
    return xyz
    
def limit_pc_range(xyz, sensor_height, higher_limit, lower_limit, min_distance, max_distance):
    # 功能：限制点云距离
    # 输入：xyz <class 'numpy.ndarray'> (n, 4) 代表齐次坐标[x, y, z, 1]，n为点的数量
    #      sensor_height <class 'float'> 代表传感器距离底面高度，单位为米
    #      higher_limit <class 'float'>  代表相对地面的限制高度，单位为米
    #      lower_limit <class 'float'>   代表相对地面的限制高度，单位为米
    #      min_distance <class 'float'>  代表相对传感器的限制距离，单位为米
    #      max_distance <class 'float'>  代表相对传感器的限制距离，单位为米
    # 输出：xyz <class 'numpy.ndarray'> (n, 4) 代表齐次坐标[x, y, z, 1]，n为点的数量
    
    xyz = xyz[np.logical_and((xyz[:, 0] ** 2 + xyz[:, 1] ** 2 > min_distance ** 2), (xyz[:, 0] ** 2 + xyz[:, 1] ** 2 < max_distance ** 2))]
    xyz = xyz[np.logical_and((xyz[:, 2] > lower_limit - sensor_height), (xyz[:, 2] < higher_limit - sensor_height))]
    
    return xyz
