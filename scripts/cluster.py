# -*- coding: UTF-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def cluster_2d_point_clouds_using_DBSCAN(xs, ys, eps=0.5, min_samples=5, leaf_size=30):
    # 功能：对平面点云聚类
    #      应用DBSCAN(Density-Based Spatial Clustering of Applications with Noise)基于密度的聚类算法
    #      将具有足够密度的点云划分为簇，无需给定聚类中心的数量
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    #      eps <class 'float'> 将两点视为同类的最大距离，该值不决定类中最远的两个点的距离
    #      min_samples <class 'int'> 最少聚类点数
    #      leaf_size <class 'int'> 传递给ball_tree或kd_tree的参数，会影响查找的速度
    # 输出：labels <class ''> 存储所有点所在的聚类中心ID，其中噪点为-1
    
    pts = np.array((xs, ys)).T
    
    # DBSCAN类输入的参数：
    #   eps: float, default=0.5 将两点视为同类的最大距离，该值不决定类中最远的两个点的距离
    #   min_samples: int, default=5 最少聚类点数
    #   metric: string, or callable, default=’euclidean’ 计算距离时使用的度量方式，一般情况下无需设置
    #   metric_params: dict, default=None 度量的关键参数，一般情况下无需设置
    #   algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’ 查找临近点的算法，一般情况下无需设置
    #   leaf_size: int, default=30 传递给ball_tree或kd_tree的参数，会影响查找的速度
    #   p: float, default=None 用于Minkowski度量，一般情况下无需设置
    #   n_jobs: int, default=None 并行运行，一般情况下无需设置
    # DBSCAN类保存的结果：
    #   db.core_sample_indices_存储所有被聚类点的索引
    #   db.components_存储所有被聚类点的坐标
    #   db.labels_存储所有点所在的聚类中心ID，其中噪点为-1
    
    # 设置聚类密度：在0.5m范围内，点云数量超过5个
    # 经测试，当目标点云数量为500时，聚类过程耗时约3ms
    db = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size).fit(pts)
    
    # labels <class 'numpy.ndarray'> (n,)
    labels = db.labels_
    
    return labels

def cluster_2d_point_clouds(xs, ys):
    # 功能：对二维点云聚类
    #      应用DBSCAN(Density-Based Spatial Clustering of Applications with Noise)基于密度的聚类算法
    #      将具有足够密度的点云划分为簇，无需给定聚类中心的数量
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表X坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表Y坐标
    # 输出：xsc <class 'numpy.ndarray'> (n,) 有聚类结果时，代表聚类点簇的X坐标，否则为输入的X坐标
    #      ysc <class 'numpy.ndarray'> (n,) 有聚类结果时，代表聚类点簇的Y坐标，否则为输入的Y坐标
    #      is_clustered <class 'bool'> 是否有聚类结果
    
    # 对平面点云聚类
    labels = cluster_2d_point_clouds_using_DBSCAN(xs, ys)
    
    # 若无聚类结果，则退出
    if labels.max() < 0:
        xsc = xs
        ysc = ys
        is_clustered = False
    else:
        # 滤除离群点，并保留最大的聚类簇
        best_label = -1
        num_best = 0
        for label in range(labels.max() + 1):
            num = np.where(labels == label)[0].shape[0]
            if num > num_best:
                best_label = label
                num_best = num
        
        # idxs <class 'numpy.ndarray'> (n,)
        idxs = np.where(labels == best_label)[0]
        xsc = xs[idxs]
        ysc = ys[idxs]
        is_clustered = True
    return xsc, ysc, is_clustered

