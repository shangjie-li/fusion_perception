# -*- coding: UTF-8 -*-

import numpy as np
import math

def project_xyz(xyz, mat, height, width, crop=True):
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

def project_xyzlwhp(xyzlwhp, mat, height, width, crop=True):
    # 功能：将三维雷达点云投影至图像
    # 输入：xyzlwhp <class 'numpy.ndarray'> (n, 7) 代表三维雷达点云的齐次坐标[x, y, z, l, w, h, p]，n为点的数量
    #      mat <class 'numpy.ndarray'> (3, 4) 代表投影矩阵
    #      height <class 'int'> 图像高度
    #      width <class 'int'> 图像宽度
    #      crop <class 'bool'> 是否裁剪掉超出图像的部分
    # 输出：clouds_xyz <class 'numpy.ndarray'> (n, 4) 代表三维雷达点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      clouds_uv <class 'numpy.ndarray'> (n, 2) 代表图像坐标[u, v]，n为点的数量
    #      clouds_lwhp <class 'numpy.ndarray'> (n, 4) 代表长宽高和方向角信息
    
    n = xyzlwhp.shape[0]
    xyz = np.hstack((xyzlwhp[:, :3], np.ones((n, 1))))
    lwhp = xyzlwhp[:, 3:]
    
    uv = mat.dot(xyz.T).T
    uv = np.true_divide(uv[:, :2], uv[:, [-1]])
    
    if not crop:
        clouds_xyz = xyz
        clouds_uv = uv
        clouds_lwhp = lwhp
    else:
        idxs = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        clouds_xyz = xyz[idxs]
        clouds_uv = uv[idxs]
        clouds_lwhp = lwhp[idxs]
    
    return clouds_xyz, clouds_uv, clouds_lwhp
