# -*- coding: UTF-8 -*- 

import numpy as np
import rospy
import time
import math
import os
import sys

from matplotlib import cm
from sklearn.cluster import DBSCAN
from sensor_msgs.msg import PointCloud2, PointField, Image
from visualization_msgs.msg import Marker, MarkerArray
from perception_msgs.msg import Obstacle, ObstacleArray

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from yolov5_detector import Yolov5Detector, draw_one_box
from kalman_filter import KalmanFilter2D, KalmanFilter4D
from obj import Object
from calib import Calib

from numpy_pc2 import pointcloud2_to_xyz_array
from project_pc import project_point_clouds, limit_pc_view, limit_pc_range
from project_pc import transform_3d_point_clouds
from cluster_pc import cluster_point_clouds
from fit_3d_model import fit_3d_model_of_cube, fit_3d_model_of_cylinder
from fit_3d_model import transform_2d_point_clouds, find_nearest_point, compute_obb_vertex_coordinates
from draw_3d_model import draw_3d_model
from get_ellipse import get_ellipse

COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

class JetColor(object):
    def __init__(self):
        # 功能：初始化JetColor对象
        
        # cmap <class 'numpy.ndarray'> (256, 3) 存储256种颜色，由深到浅
        cmap = np.zeros((256, 3), np.uint8)
        
        # 使用JET伪彩色
        for i in range(256):
            cmap[i, 0] = cm.jet(i)[0] * 255
            cmap[i, 1] = cm.jet(i)[1] * 255
            cmap[i, 2] = cm.jet(i)[2] * 255
            
        self.color_map = cmap
        self.num_color = cmap.shape[0]
        
    def get_jet_color(self, idx):
        # 功能：获取RGB颜色
        # 输入：idx <class 'float'> 索引值，范围[0, 255]
        # 输出：color <class 'tuple'> RGB颜色
        
        idx = int(idx)
        idx = max(idx, 0)
        idx = min(idx, 255)
        
        # 根据idx从cmap中选取颜色
        c = self.color_map[idx]
        color = (int(c[0]), int(c[1]), int(c[2]))
        
        return color

def draw_point_clouds_from_main_view(img, xs, ys, zs, mat, jc, min_distance=1.5, max_distance=50, circle_mode=True, radius=2):
    # 功能：在图像上绘制点云
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      xs <class 'numpy.ndarray'> (n,) 代表X坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表Y坐标
    #      zs <class 'numpy.ndarray'> (n,) 代表Z坐标
    #      mat <class 'numpy.ndarray'> (3, 4) 代表投影矩阵
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    # 建立三维点云的齐次坐标
    num = xs.shape[0]
    xyz = np.ones((num, 4))
    xyz[:, 0] = xs
    xyz[:, 1] = ys
    xyz[:, 2] = zs
    
    # 将点云投影至图像平面
    height = img.shape[0]
    width = img.shape[1]
    xyz, uv = project_point_clouds(xyz, mat, height, width)
    
    # 计算点云中各点的距离
    depth = np.sqrt(np.square(xyz[:, 0]) + np.square(xyz[:, 1]))
    depth_range = max_distance - min_distance
    n = jc.num_color
    
    if circle_mode:
        num = uv.shape[0]
        for pt in range(num):
            # 根据距离选取点云中各点的颜色
            intensity = ((depth[pt] - min_distance) / depth_range) * n
            color = jc.get_jet_color(intensity)
            cv2.circle(img, (int(uv[pt][0]), int(uv[pt][1])), radius, color, thickness=-1)
            
    else:
        num = uv.shape[0]
        for pt in range(num):
            # 根据距离选取点云中各点的颜色
            intensity = ((depth[pt] - min_distance) / depth_range) * n
            color = jc.get_jet_color(intensity)
            img[int(uv[pt][1]), int(uv[pt][0])] = np.array([color[0], color[1], color[2]])
    
    return img

def draw_point_clouds_from_bev_view(img, xs, ys, center_alignment=True, circle_mode=False, color=(96, 96, 96), radius=1):
    # 功能：在图像上绘制点云
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      xs <class 'numpy.ndarray'> (n,) 代表X坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表Y坐标
    #      color <class 'tuple'> RGB颜色
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    xs = xs * 10
    ys = - ys * 10
    xs = xs.astype(int)
    ys = ys.astype(int)
    
    height = img.shape[0]
    width = img.shape[1]
    if center_alignment:
        xst, yst = transform_2d_point_clouds(xs, ys, 0, width / 2, height / 2)
    else:
        xst, yst = transform_2d_point_clouds(xs, ys, 0, 0, height / 2)
    
    if circle_mode:
        num = xst.shape[0]
        for pt in range(num):
            cv2.circle(img, (int(xst[pt]), int(yst[pt])), radius, color, thickness=-1)
            
    else:
        color = np.array([color[0], color[1], color[2]])
        idxs = (xst >= 0) & (xst < width) & (yst >= 0) & (yst < height)
        xst = xst[idxs]
        yst = yst[idxs]
        
        num = xst.shape[0]
        for pt in range(num):
            img[int(yst[pt]), int(xst[pt])] = color
            
    return img

def draw_object_info(img, classname, number, xref, yref, vx, vy, uv_1, uv_2, color, display_class=True, display_id=True, display_xx=True, l=0, w=0, h=0, phi=0):
    # 功能：在图像上绘制目标信息
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      classname <class 'str'> 类别
    #      xref <class 'float'> X坐标
    #      yref <class 'float'> Y坐标
    #      vx <class 'float'> X方向速度
    #      vy <class 'float'> Y方向速度
    #      uv_1 <class 'numpy.ndarray'> (1, 2) 代表图像坐标[u, v]
    #      uv_1 <class 'numpy.ndarray'> (1, 2) 代表图像坐标[u, v]
    #      color <class 'tuple'> RGB颜色
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    u1, v1, u2, v2 = int(uv_1[0, 0]), int(uv_1[0, 1]), int(uv_2[0, 0]), int(uv_2[0, 1])
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.4
    font_thickness = 1
    
    if display_class or display_id:
        if display_class and display_id:
            text_str = classname + ' ' + str(number)
        elif display_class and not display_id:
            text_str = classname
        elif not display_class and display_id:
            text_str = str(number)
        
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        cv2.rectangle(img, (u1, v1), (u1 + text_w, v1 - text_h - 4), color, -1)
        cv2.putText(img, text_str, (u1, v1 - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    if display_xx:
        text_lo = '(%.1fm, %.1fm)' % (xref, yref)
        text_w_lo, _ = cv2.getTextSize(text_lo, font_face, font_scale, font_thickness)[0]
        cv2.rectangle(img, (u2, v2), (u2 + text_w_lo, v2 + text_h + 4), color, -1)
        cv2.putText(img, text_lo, (u2, v2 + text_h + 1), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        text_ve = '(%.1fm/s, %.1fm/s)' % (vx, vy)
        text_w_ve, _ = cv2.getTextSize(text_ve, font_face, font_scale, font_thickness)[0]
        cv2.rectangle(img, (u2, v2 + text_h + 4), (u2 + text_w_ve, v2 + 2 * text_h + 8), color, -1)
        cv2.putText(img, text_ve, (u2, v2 + 2 * text_h + 5), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        text_sc = '(%.1fm, %.1fm, %.1fm, %.1frad)' % (l, w, h, phi)
        text_w_sc, _ = cv2.getTextSize(text_sc, font_face, font_scale, font_thickness)[0]
        cv2.rectangle(img, (u2, v2 + 2 * text_h + 8), (u2 + text_w_sc, v2 + 3 * text_h + 12), color, -1)
        cv2.putText(img, text_sc, (u2, v2 + 3 * text_h + 9), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img

def draw_object_model_from_main_view(img, objs, mat, frame, display_frame, display_class, display_id, display_state, thickness=1, fitting_mode=False):
    # 功能：在图像上绘制目标轮廓
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      objs <class 'list'> 存储目标检测结果
    #      mat <class 'numpy.ndarray'> (3, 4) 代表投影矩阵
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    # 按距离从远到近进行排序
    num = len(objs)
    dds = []
    for i in range(num):
        dd = objs[i].xref ** 2 + objs[i].yref ** 2
        dds.append(dd)
    idxs = list(np.argsort(dds))
    
    for i in reversed(idxs):
        if fitting_mode:
            # 绘制三维边界框
            xst, yst = compute_obb_vertex_coordinates(objs[i].x0, objs[i].y0, objs[i].l, objs[i].w, objs[i].phi)
            z = objs[i].z0 - objs[i].h / 2
            polygon = np.zeros((4, 1, 3))
            for j in range(4):
                polygon[j, 0, :] = np.array([xst[j], yst[j], z])
            
            height = objs[i].h
            color = objs[i].color
            img, flag = draw_3d_model(img, polygon, height, mat, color, thickness)
            
            # 选取三维边界框中的顶点，绘制目标信息
            pdds = []
            for j in range(4):
                pdd = polygon[j, 0, 0] ** 2 + polygon[j, 0, 1] ** 2
                pdds.append(pdd)
            pidxs = list(np.argsort(pdds))
            pidx = pidxs[0]
            
            frame_height = img.shape[0]
            frame_width = img.shape[1]
            
            xyz = np.array([[polygon[pidx, 0, 0], polygon[pidx, 0, 1], polygon[pidx, 0, 2] + height, 1]])
            _, uv_1 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
            xyz = np.array([[polygon[pidx, 0, 0], polygon[pidx, 0, 1], polygon[pidx, 0, 2], 1]])
            _, uv_2 = project_point_clouds(xyz, mat, frame_height, frame_width, crop=False)
            
            display_info = False
            if display_class or display_id or display_state: display_info = True
            if flag and display_info:
                img = draw_object_info(img, objs[i].classname, objs[i].number,
                 objs[i].xref, objs[i].yref, objs[i].vx, objs[i].vy, uv_1, uv_2, objs[i].color,
                  display_class, display_id, display_state)
        else:
            # 绘制二维边界框
            xyxy = objs[i].box
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(img, c1, c2, objs[i].color, thickness, lineType=cv2.LINE_AA)
            
            # 选取二维边界框中的顶点，绘制目标信息
            uv_1 = np.array([int(xyxy[0]), int(xyxy[1])]).reshape(1, 2)
            uv_2 = np.array([int(xyxy[0]), int(xyxy[3])]).reshape(1, 2)
            
            if display_class or display_id or display_state:
                img = draw_object_info(img, objs[i].classname, objs[i].number,
                 objs[i].xref, objs[i].yref, objs[i].vx, objs[i].vy, uv_1, uv_2, objs[i].color,
                  display_class, display_id, display_state, l=objs[i].l, w=objs[i].w, h=objs[i].h, phi=objs[i].phi)
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.4
    font_thickness = 1
    if display_frame:
        cv2.putText(img, str(frame), (10, 20), font_face, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
    
    return img

def draw_object_model_from_bev_view(img, objs, display_obj_pc=False, display_gate=True, center_alignment=True, circle_mode=False, thickness=1):
    # 功能：在图像上绘制目标轮廓
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      objs <class 'list'> 存储目标检测结果
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    num = len(objs)
    for i in range(num):
        # 绘制目标点云
        if display_obj_pc:
            img = draw_point_clouds_from_bev_view(img, objs[i].xs, objs[i].ys, center_alignment, circle_mode, objs[i].color, radius=1)
        
        if objs[i].has_orientation:
            # 盒子模型
            xst, yst = compute_obb_vertex_coordinates(objs[i].x0, objs[i].y0, objs[i].l, objs[i].w, objs[i].phi)
            
            xst = xst * 10
            yst = - yst * 10
            xst = xst.astype(int)
            yst = yst.astype(int)
            
            height = img.shape[0]
            width = img.shape[1]
            if center_alignment:
                xst, yst = transform_2d_point_clouds(xst, yst, 0, width / 2, height / 2)
            else:
                xst, yst = transform_2d_point_clouds(xst, yst, 0, 0, height / 2)
            
            for j in range(4):
                pt_1 = (int(xst[j]), int(yst[j]))
                pt_2 = (int(xst[(j + 1) % 4]), int(yst[(j + 1) % 4]))
                cv2.line(img, pt_1, pt_2, objs[i].color, thickness)
            
        else:
            # 圆点模型
            xmec = int(objs[i].x0 * 10)
            ymec = int(- objs[i].y0 * 10)
            radius = int((objs[i].l / 2) * 10)
            
            cv2.circle(img, (xmec, ymec), radius, objs[i].color, thickness)
        
        if display_gate:
            # 绘制跟踪门椭圆
            a, b = objs[i].tracker.compute_association_gate(objs[i].tracker.gate_threshold)
            x, y = objs[i].xref, objs[i].yref
            
            xs, ys = get_ellipse(x, y, a, b, 0)
            xs, ys = np.array(xs), np.array(ys)
            
            img = draw_point_clouds_from_bev_view(img, xs, ys, center_alignment, circle_mode, objs[i].color, radius=1)
        
    return img

def fuse(xyz, uv, classes, scores, boxes, img_height, img_width, fitting_mode=False):
    # 功能：点云与图像实例匹配
    # 输入：xyz <class 'numpy.ndarray'> (n, 4) 代表三维点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      uv  <class 'numpy.ndarray'> (n, 2) 代表图像坐标[u, v]，n为点的数量
    #      classes <class 'numpy.ndarray'> (N,) N为目标数量
    #      scores <class 'numpy.ndarray'> (N,) N为目标数量
    #      boxes <class 'numpy.ndarray'> (N, 4) N为目标数量
    #      img_height <class 'int'> 图像高度
    #      img_width <class 'int'> 图像宽度
    # 输出：objs <class 'list'> 存储目标检测结果

    objs = []
    num = classes.shape[0] if classes is not None else 0
    
    # 遍历每个目标
    for i in range(num):
        u1, v1, u2, v2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        idxs = (uv[:, 0] >= u1) & (uv[:, 0] < u2) & (uv[:, 1] >= v1) & (uv[:, 1] < v2)
        xyz_chosen = xyz[idxs]
        xs, ys, zs = xyz_chosen[:, 0], xyz_chosen[:, 1], xyz_chosen[:, 2]
        
        if xs.shape[0] > 0:
            # 对三维点云聚类
            xsc, ysc, zsc, is_clustered = cluster_point_clouds(xs, ys, zs)
            
            if is_clustered:
                # 创建目标对象
                obj = Object()
                obj.classname, obj.score, obj.box = str(classes[i]), float(scores[i]), boxes[i]
                obj.xs, obj.ys, obj.zs = xsc, ysc, zsc
                
                if classes[i] in ['car', 'bus', 'truck']:  # 拟合3D盒子模型
                    if fitting_mode:
                        obj.x0, obj.y0, obj.z0, obj.l, obj.w, obj.h, obj.phi, obj.has_orientation = fit_3d_model_of_cube(xsc, ysc, zsc)
                    else:
                        max_xsc, min_xsc = max(xsc), min(xsc)
                        max_ysc, min_ysc = max(ysc), min(ysc)
                        max_zsc, min_zsc = max(zsc), min(zsc)
                        
                        obj.x0 = (max_xsc + min_xsc) / 2
                        obj.y0 = (max_ysc + min_ysc) / 2
                        obj.z0 = (max_zsc + min_zsc) / 2
                        
                        if classes[i] == 'car':
                            obj.l, obj.w, obj.h = 5.0, 2.0, 1.8
                            _, _, _, _, _, _, obj.phi, obj.has_orientation = fit_3d_model_of_cube(xsc, ysc, zsc)
                        elif classes[i] == 'bus':
                            obj.l, obj.w, obj.h = 8.0, 3.0, 2.8
                            _, _, _, _, _, _, obj.phi, obj.has_orientation = fit_3d_model_of_cube(xsc, ysc, zsc)
                        elif classes[i] == 'truck':
                            obj.l, obj.w, obj.h = 6.0, 2.5, 2.5
                            _, _, _, _, _, _, obj.phi, obj.has_orientation = fit_3d_model_of_cube(xsc, ysc, zsc)
                
                elif classes[i] in ['person']:  # 拟合3D圆点模型
                    if fitting_mode:
                        obj.x0, obj.y0, obj.z0, obj.l, obj.w, obj.h, obj.phi, obj.has_orientation = fit_3d_model_of_cylinder(xsc, ysc, zsc)
                        obj.l, obj.w = min(obj.l, 1.0), min(obj.w, 1.0)
                    else:
                        max_xsc, min_xsc = max(xsc), min(xsc)
                        max_ysc, min_ysc = max(ysc), min(ysc)
                        max_zsc, min_zsc = max(zsc), min(zsc)
                        
                        obj.x0 = (max_xsc + min_xsc) / 2
                        obj.y0 = (max_ysc + min_ysc) / 2
                        obj.z0 = (max_zsc + min_zsc) / 2
                        
                        obj.l, obj.w, obj.h = 0.5, 0.5, 1.8
                        obj.phi, obj.has_orientation = 0.0, False
                
                obj.xref, obj.yref = obj.x0, obj.y0
                objs.append(obj)
                
    return objs

def track(number, objs_tracked, objs_temp, objs_detected, blind_update_limit, frame_rate, COLORS, fitting_mode=False):
    # 数据关联与跟踪
    num = len(objs_tracked)
    for j in range(num):
        flag = False
        idx = 0
        ddm = float('inf')
        
        # 计算残差加权范数
        n = len(objs_detected)
        for k in range(n):
            zx = objs_detected[k].xref
            zy = objs_detected[k].yref
            dd = objs_tracked[j].tracker.compute_the_residual(zx, zy)
            if dd < ddm and dd < objs_tracked[j].tracker.gate_threshold:
                idx = k
                ddm = dd
                flag = True
        
        if flag:
            # 匹配成功，预测并更新
            objs_tracked[j].tracker.predict()
            objs_tracked[j].tracker.update(objs_detected[idx].xref, objs_detected[idx].yref)
            objs_tracked[j].tracker_l.predict()
            objs_tracked[j].tracker_l.update(objs_detected[idx].l)
            objs_tracked[j].tracker_w.predict()
            objs_tracked[j].tracker_w.update(objs_detected[idx].w)
            
            # 继承检测结果中的参数
            obj = objs_detected[idx]
            obj.tracker = objs_tracked[j].tracker
            obj.tracker_l = objs_tracked[j].tracker_l
            obj.tracker_w = objs_tracked[j].tracker_w
            obj.number = objs_tracked[j].number
            obj.color = objs_tracked[j].color
            objs_tracked[j] = obj
            
            # 修改更新中断次数
            objs_tracked[j].tracker_blind_update = 0
            objs_detected.pop(idx)
        else:
            # 匹配不成功，只预测
            objs_tracked[j].tracker.predict()
            
            # 修改更新中断次数
            objs_tracked[j].tracker_blind_update += 1
        
        # 输出滤波值
        objs_tracked[j].xref = objs_tracked[j].tracker.xx[0, 0]
        objs_tracked[j].vx = objs_tracked[j].tracker.xx[1, 0]
        objs_tracked[j].yref = objs_tracked[j].tracker.xx[2, 0]
        objs_tracked[j].vy = objs_tracked[j].tracker.xx[3, 0]
        
        objs_tracked[j].x0 = objs_tracked[j].xref
        objs_tracked[j].y0 = objs_tracked[j].yref
        
        if fitting_mode:
            objs_tracked[j].l = objs_tracked[j].tracker_l.xx[0, 0]
            objs_tracked[j].w = objs_tracked[j].tracker_w.xx[0, 0]
            
            if objs_tracked[j].l < 0: objs_tracked[j].l = 0
            if objs_tracked[j].w < 0: objs_tracked[j].w = 0
        
    # 删除长时间未跟踪的目标
    objs_remained = []
    num = len(objs_tracked)
    for j in range(num):
        if objs_tracked[j].tracker_blind_update <= blind_update_limit:
            objs_remained.append(objs_tracked[j])
    objs_tracked = objs_remained
    
    # 增广跟踪列表
    num = len(objs_temp)
    for j in range(num):
        flag = False
        idx = 0
        ddm = float('inf')
        
        # 计算残差加权范数
        n = len(objs_detected)
        for k in range(n):
            zx = objs_detected[k].xref
            zy = objs_detected[k].yref
            dd = objs_temp[j].tracker.compute_the_residual(zx, zy)
            if dd < ddm and dd < objs_temp[j].tracker.gate_threshold:
                idx = k
                ddm = dd
                flag = True
        
        if flag:
            zx = objs_detected[idx].xref
            zy = objs_detected[idx].yref
            x = objs_temp[j].tracker.xx[0, 0]
            y = objs_temp[j].tracker.xx[2, 0]
            vx = (zx - x) / objs_temp[j].tracker.ti
            vy = (zy - y) / objs_temp[j].tracker.ti
            
            # 继承检测结果中的参数
            obj = objs_detected[idx]
            obj.tracker = objs_temp[j].tracker
            obj.tracker_l = objs_temp[j].tracker_l
            obj.tracker_w = objs_temp[j].tracker_w
            objs_temp[j] = obj
            
            # 对跟踪的位置、速度重新赋值
            objs_temp[j].tracker.xx[0, 0] = zx
            objs_temp[j].tracker.xx[1, 0] = vx
            objs_temp[j].tracker.xx[2, 0] = zy
            objs_temp[j].tracker.xx[3, 0] = vy
            
            objs_temp[j].xref = objs_temp[j].tracker.xx[0, 0]
            objs_temp[j].vx = objs_temp[j].tracker.xx[1, 0]
            objs_temp[j].yref = objs_temp[j].tracker.xx[2, 0]
            objs_temp[j].vy = objs_temp[j].tracker.xx[3, 0]
            
            # 增加ID和颜色等属性
            number += 1
            objs_temp[j].number = number
            objs_temp[j].color = COLORS[number % len(COLORS)]
            objs_tracked.append(objs_temp[j])
            
            objs_detected.pop(idx)
    
    # 增广临时跟踪列表
    objs_temp = objs_detected
    num = len(objs_temp)
    for j in range(num):
        # 初始化卡尔曼滤波器，对目标进行跟踪
        objs_temp[j].tracker = KalmanFilter4D(1 / frame_rate, objs_temp[j].xref, objs_temp[j].vx,
         objs_temp[j].yref, objs_temp[j].vy, sigma_ax=1, sigma_ay=1, sigma_ox=0.1, sigma_oy=0.1,
          gate_threshold=400)
        
        # 初始化卡尔曼滤波器，对相关参数进行平滑
        objs_temp[j].tracker_l = KalmanFilter2D(1 / frame_rate, objs_temp[j].l, 0, sigma_ax=1, sigma_ox=0.1)
        objs_temp[j].tracker_w = KalmanFilter2D(1 / frame_rate, objs_temp[j].w, 0, sigma_ax=1, sigma_ox=0.1)
    
    return number, objs_tracked, objs_temp

def publish_marker_msg(pub, header, frame_rate, objs, random_number=True):
    markerarray = MarkerArray()
    
    num = len(objs)
    for i in range(num):
        marker = Marker()
        marker.header = header
        
        # 提取目标信息
        obj = objs[i]
        
        # 设置该标记的命名空间和ID，ID应该是独一无二的
        # 具有相同命名空间和ID的标记将会覆盖前一个
        marker.ns = obj.classname
        if random_number:
            marker.id = i
        else:
            marker.id = obj.number
        
        # 设置标记类型
        if obj.has_orientation:
            marker.type = Marker.CUBE
        else:
            marker.type = Marker.CYLINDER
        
        # 设置标记行为：ADD为添加，DELETE为删除
        marker.action = Marker.ADD
        
        # 设置标记位姿
        marker.pose.position.x = obj.x0
        marker.pose.position.y = obj.y0
        marker.pose.position.z = obj.z0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(0.5 * obj.phi)
        marker.pose.orientation.w = math.cos(0.5 * obj.phi)
        
        # 设置标记尺寸
        marker.scale.x = obj.l
        marker.scale.y = obj.w
        marker.scale.z = obj.h
    
        # 设置标记颜色，确保不透明度alpha不为0
        marker.color.r = obj.color[0] / 255.0
        marker.color.g = obj.color[1] / 255.0
        marker.color.b = obj.color[2] / 255.0
        marker.color.a = 0.75
        
        marker.lifetime = rospy.Duration(1 / frame_rate)
        marker.text = '(' + str(obj.vx) + ', ' + str(obj.vy) + ')'
        
        markerarray.markers.append(marker)
    
    pub.publish(markerarray)

def publish_obstacle_msg(pub, header, frame_rate, objs, random_number=True):
    obstaclearray = ObstacleArray()
    
    num = len(objs)
    for i in range(num):
        obstacle = Obstacle()
        obstacle.header = header
        
        # 提取目标信息
        obj = objs[i]
        
        # 设置该障碍物的命名空间和ID，ID应该是独一无二的
        # 具有相同命名空间和ID的障碍物将会覆盖前一个
        obstacle.ns = obj.classname
        if random_number:
            obstacle.id = i
        else:
            obstacle.id = obj.number
        
        # 设置障碍物位姿
        obstacle.pose.position.x = obj.x0
        obstacle.pose.position.y = obj.y0
        obstacle.pose.position.z = obj.z0
        obstacle.pose.orientation.x = 0.0
        obstacle.pose.orientation.y = 0.0
        obstacle.pose.orientation.z = math.sin(0.5 * obj.phi)
        obstacle.pose.orientation.w = math.cos(0.5 * obj.phi)
        
        # 设置障碍物尺寸
        obstacle.scale.x = obj.l
        obstacle.scale.y = obj.w
        obstacle.scale.z = obj.h
        
        # 设置障碍物速度
        obstacle.v_validity = False if random_number else True
        obstacle.vx = obj.vx
        obstacle.vy = obj.vy
        obstacle.vz = 0
        
        # 设置障碍物加速度
        obstacle.a_validity = False
        obstacle.ax = 0
        obstacle.ay = 0
        obstacle.az = 0
        
        obstaclearray.obstacles.append(obstacle)
    
    pub.publish(obstaclearray)

def image_callback(image):
    global cv_stamps
    global cv_images
    
    cv_stamp = image.header.stamp.secs + 0.000000001 * image.header.stamp.nsecs
    cv_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    
    # 更新图像序列
    if len(cv_stamps) < 30:
        cv_stamps.append(cv_stamp)
        cv_images.append(cv_image)
    else:
        cv_stamps.pop(0)
        cv_images.pop(0)
        cv_stamps.append(cv_stamp)
        cv_images.append(cv_image)

def point_clouds_callback(pc):
    time_start_all = time.time()
    global objs_detected
    global objs_tracked
    global objs_temp
    global number
    global frame
    frame += 1
    
    global display_obj_pc, display_gate, display_class, display_id, display_state
    
    # 动态设置终端输出
    global print_time
    global print_objects_info
    print_time = rospy.get_param("~print_time")
    print_objects_info = rospy.get_param("~print_objects_info")
    
    # 相机与激光雷达消息同步
    global cv_stamps
    global cv_images
    
    # secs为秒，nsecs为纳秒
    lidar_stamp = pc.header.stamp.secs + 0.000000001 * pc.header.stamp.nsecs
    et_m, stamp_idx = float('inf'), 0
    for t in range(len(cv_stamps)):
        et = abs(cv_stamps[t] - lidar_stamp)
        if et < et_m:
            et_m = et
            stamp_idx = t
    cv_stamp = cv_stamps[stamp_idx]
    
    # 载入图像
    cv_image = cv_images[stamp_idx]
    cv_image = cv_image[:, :, (2, 1, 0)]
    
    # 图像实例分割
    time_start = time.time()
    classes, scores, boxes = detector.run(cv_image, items=items, conf_thres=conf_thres)
    time_segmentation = time.time() - time_start
    
    # 载入点云
    xyz_raw = pointcloud2_to_xyz_array(pc, remove_nans=True)
    xyz = xyz_raw.copy()
    
    # 点云透视投影
    time_start = time.time()
    if pc_view_crop:
        xyz = limit_pc_view(xyz, area_number, fov_angle)
    if pc_range_crop:
        xyz = limit_pc_range(xyz, sensor_height, higher_limit, lower_limit, min_distance, max_distance)
    xyz, uv = project_point_clouds(xyz, calib.projection_l2i, window_height, window_width)
    time_projection = time.time() - time_start
    
    # 图像与点云实例匹配
    time_start = time.time()
    objs = fuse(xyz, uv, classes, scores, boxes, window_height, window_width)
    time_fusion = time.time() - time_start
    
    # 目标跟踪
    time_start = time.time()
    objs_detected = []
    num = len(objs)
    for i in range(num):
        obj = Object()
        obj.classname = objs[i].classname
        obj.score = objs[i].score
        obj.box = objs[i].box
        
        obj.xs = objs[i].xs
        obj.ys = objs[i].ys
        obj.zs = objs[i].zs
        
        obj.x0 = objs[i].x0
        obj.y0 = objs[i].y0
        obj.z0 = objs[i].z0
        obj.l = objs[i].l
        obj.w = objs[i].w
        obj.h = objs[i].h
        obj.phi = objs[i].phi
        obj.has_orientation = objs[i].has_orientation
        
        obj.xref = objs[i].xref
        obj.yref = objs[i].yref
        objs_detected.append(obj)
    
    number, objs_tracked, objs_temp = track(number, objs_tracked, objs_temp, objs_detected, blind_update_limit, frame_rate, COLORS)
    if number >= max_id:
        number = 0
    time_tracking = time.time() - time_start
    
    # 模式切换
    if processing_mode == 'D':
        publish_marker_msg(pub_marker, pc.header, frame_rate, objs, True)
        publish_obstacle_msg(pub_obstacle, pc.header, frame_rate, objs, True)
        objs = objs
        display_gate = False
    elif processing_mode == 'DT':
        publish_marker_msg(pub_marker, pc.header, frame_rate, objs, True)
        publish_marker_msg(pub_marker_tracked, pc.header, frame_rate, objs_tracked, False)
        publish_obstacle_msg(pub_obstacle, pc.header, frame_rate, objs_tracked, False)
        objs = objs_tracked
        display_obj_pc = False
    else:
        raise Exception('processing_mode is not "D" or "DT".')
    
    # 可视化
    time_start = time.time()
    if display_image_raw:
        window_image_raw = cv_image.copy()
    if display_image_segmented:
        window_image_segmented = cv_image.copy()
        num = len(objs)
        for i in range(num):
            if objs[i].tracker_blind_update > 0:
                continue
            classname = objs[i].classname
            score = objs[i].score
            box = objs[i].box
            color = objs[i].color
            draw_one_box(window_image_segmented, classname, score, box, color, line_thickness=1)
    if display_point_clouds_raw:
        window_point_clouds_raw = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255
        window_point_clouds_raw = draw_point_clouds_from_bev_view(window_point_clouds_raw, xyz_raw[:, 0], xyz_raw[:, 1], center_alignment=False, circle_mode=False, color=(96, 96, 96), radius=1)
    if display_point_clouds_projected:
        window_point_clouds_projected = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255
        window_point_clouds_projected = draw_point_clouds_from_main_view(window_point_clouds_projected, xyz[:, 0], xyz[:, 1], xyz[:, 2], calib.projection_l2i, jc, circle_mode=True, radius=2)
    
    if display_segmentation_result:
        window_segmentation_result = cv_image.copy()
        if classes is not None:
            for i in reversed(range(classes.shape[0])):
                classname = classes[i]
                score = scores[i]
                box = boxes[i]
                color = COLORS[i % len(COLORS)]
                draw_one_box(window_segmentation_result, classname, score, box, color, line_thickness=1)
    if display_fusion_result:
        window_fusion_result = cv_image.copy()
        num = len(objs)
        for i in range(num):
            if objs[i].tracker_blind_update > 0:
                continue
            classname = objs[i].classname
            score = objs[i].score
            box = objs[i].box
            color = objs[i].color
            draw_one_box(window_fusion_result, classname, score, box, color, line_thickness=1)
            xs = objs[i].xs
            ys = objs[i].ys
            zs = objs[i].zs
            window_fusion_result = draw_point_clouds_from_main_view(window_fusion_result, xs, ys, zs, calib.projection_l2i, jc, circle_mode=True, radius=1)
    if display_calibration_result:
        window_calibration_result = cv_image.copy()
        window_calibration_result = draw_point_clouds_from_main_view(window_calibration_result, xyz[:, 0], xyz[:, 1], xyz[:, 2], calib.projection_l2i, jc, circle_mode=True, radius=1)
    
    if display_2d_modeling_result:
        window_2d_modeling_result = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255
        window_2d_modeling_result = draw_point_clouds_from_bev_view(window_2d_modeling_result, xyz_raw[:, 0], xyz_raw[:, 1], center_alignment=False, circle_mode=False, color=(96, 96, 96), radius=1)
        window_2d_modeling_result = draw_object_model_from_bev_view(window_2d_modeling_result, objs, display_obj_pc, display_gate, center_alignment=False, circle_mode=False, thickness=1)
    if display_3d_modeling_result:
        window_3d_modeling_result = cv_image.copy()
        window_3d_modeling_result = draw_object_model_from_main_view(window_3d_modeling_result, objs, calib.projection_l2i, frame, display_frame, display_class, display_id, display_state, thickness=2)
    time_display = time.time() - time_start
    
    # 显示与保存
    if display_image_raw:
        cv2.namedWindow("image_raw", cv2.WINDOW_NORMAL)
        cv2.imshow("image_raw", window_image_raw)
        video_image_raw.write(window_image_raw)
    if display_image_segmented:
        cv2.namedWindow("image_segmented", cv2.WINDOW_NORMAL)
        cv2.imshow("image_segmented", window_image_segmented)
        video_image_segmented.write(window_image_segmented)
    if display_point_clouds_raw:
        cv2.namedWindow("point_clouds_raw", cv2.WINDOW_NORMAL)
        cv2.imshow("point_clouds_raw", window_point_clouds_raw)
        video_point_clouds_raw.write(window_point_clouds_raw)
    if display_point_clouds_projected:
        cv2.namedWindow("point_clouds_projected", cv2.WINDOW_NORMAL)
        cv2.imshow("point_clouds_projected", window_point_clouds_projected)
        video_point_clouds_projected.write(window_point_clouds_projected)
    
    if display_segmentation_result:
        cv2.namedWindow("segmentation_result", cv2.WINDOW_NORMAL)
        cv2.imshow("segmentation_result", window_segmentation_result)
        video_segmentation_result.write(window_segmentation_result)
    if display_fusion_result:
        cv2.namedWindow("fusion_result", cv2.WINDOW_NORMAL)
        cv2.imshow("fusion_result", window_fusion_result)
        video_fusion_result.write(window_fusion_result)
    if display_calibration_result:
        cv2.namedWindow("calibration_result", cv2.WINDOW_NORMAL)
        cv2.imshow("calibration_result", window_calibration_result)
        video_calibration_result.write(window_calibration_result)
    
    if display_2d_modeling_result:
        cv2.namedWindow("2d_modeling_result", cv2.WINDOW_NORMAL)
        cv2.imshow("2d_modeling_result", window_2d_modeling_result)
        video_2d_modeling_result.write(window_2d_modeling_result)
    if display_3d_modeling_result:
        cv2.namedWindow("3d_modeling_result", cv2.WINDOW_NORMAL)
        cv2.imshow("3d_modeling_result", window_3d_modeling_result)
        video_3d_modeling_result.write(window_3d_modeling_result)
    
    # 显示窗口时按Esc键终止程序
    display_now = False
    if display_image_raw or display_image_segmented or display_point_clouds_raw or display_point_clouds_projected:
        display_now = True
    if display_segmentation_result or display_fusion_result or display_2d_modeling_result or display_3d_modeling_result:
        display_now = True
    
    if display_now and cv2.waitKey(1) == 27:
        print("\nReceived the shutdown signal.\n")
        if display_image_raw:
            cv2.destroyWindow("image_raw")
            video_image_raw.release()
            print("Save video of image_raw.")
        if display_image_segmented:
            cv2.destroyWindow("image_segmented")
            video_image_segmented.release()
            print("Save video of image_segmented.")
        if display_point_clouds_raw:
            cv2.destroyWindow("point_clouds_raw")
            video_point_clouds_raw.release()
            print("Save video of point_clouds_raw.")
        if display_point_clouds_projected:
            cv2.destroyWindow("point_clouds_projected")
            video_point_clouds_projected.release()
            print("Save video of point_clouds_projected.")
        
        if display_segmentation_result:
            cv2.destroyWindow("segmentation_result")
            video_segmentation_result.release()
            print("Save video of segmentation_result.")
        if display_fusion_result:
            cv2.destroyWindow("fusion_result")
            video_fusion_result.release()
            print("Save video of fusion_result.")
        if display_calibration_result:
            cv2.destroyWindow("calibration_result")
            video_calibration_result.release()
            print("Save video of calibration_result.")
        
        if display_2d_modeling_result:
            cv2.destroyWindow("2d_modeling_result")
            video_2d_modeling_result.release()
            print("Save video of 2d_modeling_result.")
        if display_3d_modeling_result:
            cv2.destroyWindow("3d_modeling_result")
            video_3d_modeling_result.release()
            print("Save video of 3d_modeling_result.")
        rospy.signal_shutdown("Everything is over now.")
    
    # 记录耗时情况
    time_all = time.time() - time_start_all
    time_segmentation = round(time_segmentation, 3)
    time_projection = round(time_projection, 3)
    time_fusion = round(time_fusion, 3)
    time_tracking = round(time_tracking, 3)
    time_display = round(time_display, 3)
    time_all = round(time_all, 3)
    
    if print_time:
        print()
        print("image_stamp               ", cv_stamp)
        print("lidar_stamp               ", lidar_stamp)
        print("time cost of segmentation ", time_segmentation)
        print("time cost of projection   ", time_projection)
        print("time cost of fusion       ", time_fusion)
        print("time cost of tracking     ", time_tracking)
        print("time cost of display      ", time_display)
        print("time cost of all          ", time_all)
    
    if print_objects_info:
        print()
        num = len(objs)
        for j in range(num):
            print()
            print('ID')
            print(objs[j].number)
            print('xx')
            print(objs[j].tracker.xx)
            print('pp')
            print(objs[j].tracker.pp)
    
    if record_objects_info:
        num = len(objs)
        for j in range(num):
            with open(filename_objects_info, 'a') as fob:
                fob.write('time_stamp:%.3f frame:%d id:%d x:%.3f vx:%.3f y:%.3f vy:%.3f x0:%.3f y0:%.3f z0:%.3f l:%.3f w:%.3f h:%.3f phi:%.3f' % (
                    lidar_stamp, frame, objs[j].number, objs[j].xref, objs[j].vx, objs[j].yref, objs[j].vy, objs[j].x0, objs[j].y0, objs[j].z0, objs[j].l, objs[j].w, objs[j].h, objs[j].phi))
                fob.write('\n')
    
    if record_time:
        with open(filename_time, 'a') as fob:
            fob.write('frame:%d amount:%d segmentation:%.3f projection:%.3f fusion:%.3f tracking:%.3f display:%.3f all:%.3f' % (
                frame, len(objs), time_segmentation, time_projection, time_fusion, time_tracking, time_display, time_all))
            fob.write('\n')

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("dt")
    
    # 动态设置终端输出
    print_time = False
    print_objects_info = False
    
    # 记录结果
    record_objects_info = rospy.get_param("~record_objects_info")
    if record_objects_info:
        filename_objects_info = 'result.txt'
        with open(filename_objects_info, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    # 记录耗时
    record_time = rospy.get_param("~record_time")
    if record_time:
        filename_time = 'time_cost.txt'
        with open(filename_time, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    # 设置ROS消息名称
    sub_image_topic = rospy.get_param("~sub_image_topic")
    sub_point_clouds_topic = rospy.get_param("~sub_point_clouds_topic")
    pub_marker_topic = rospy.get_param("~pub_marker_topic")
    pub_marker_tracked_topic = rospy.get_param("~pub_marker_tracked_topic")
    pub_obstacle_topic = 'obstacle_array'
    
    # 设置标定参数
    calibration_file = rospy.get_param("~calibration_file")
    cwd = os.getcwd()
    calib_pth_idx = -1
    while cwd[calib_pth_idx] != '/':
        calib_pth_idx -= 1
    calib_pth = cwd[:calib_pth_idx] + '/conf/' + calibration_file
    calib = Calib(calib_pth, print_mat=True)
    
    # 限制点云视场角度
    pc_view_crop = rospy.get_param("~pc_view_crop")
    area_number = rospy.get_param("~area_number")
    fov_angle = rospy.get_param("~fov_angle")
    
    # 限制点云距离
    pc_range_crop = rospy.get_param("~pc_range_crop")
    sensor_height = rospy.get_param("~sensor_height")
    higher_limit = rospy.get_param("~higher_limit")
    lower_limit = rospy.get_param("~lower_limit")
    min_distance = rospy.get_param("~min_distance")
    max_distance = rospy.get_param("~max_distance")
    
    # 初始化Yolov5Detector
    detector = Yolov5Detector()
    
    # 准备图像序列
    print('Waiting for topic...')
    cv_stamps, cv_images = [], []
    rospy.Subscriber(sub_image_topic, Image, image_callback, queue_size=1, buff_size=52428800)
    while len(cv_stamps) < 30:
        time.sleep(1)
    print('  Done.\n')
    
    # 初始化检测列表、跟踪列表、临时跟踪列表
    objs_detected = []
    objs_tracked = []
    objs_temp = []
    number = 0
    frame = 0
    
    # 设置更新中断次数的限制
    blind_update_limit = rospy.get_param("~blind_update_limit")
    
    # 设置工作帧率
    frame_rate = rospy.get_param("~frame_rate")
    
    # 设置目标的最大ID
    max_id = rospy.get_param("~max_id")
    
    # 设置处理模式及对象
    processing_mode = rospy.get_param("~processing_mode")
    processing_object = rospy.get_param("~processing_object")
    
    if processing_object == 'both':
        items = [0, 2, 5, 7]
        conf_thres = 0.25
    elif processing_object == 'car':
        items = [2, 5, 7]
        conf_thres = 0.25
    elif processing_object == 'person':
        items = [0]
        conf_thres = 0.25
    else:
        raise Exception('processing_object is not "car" or "person" or "both".')
    
    # 设置显示窗口
    display_image_raw = rospy.get_param("~display_image_raw")
    display_image_segmented = rospy.get_param("~display_image_segmented")
    display_point_clouds_raw = rospy.get_param("~display_point_clouds_raw")
    display_point_clouds_projected = rospy.get_param("~display_point_clouds_projected")
    
    display_segmentation_result = rospy.get_param("~display_segmentation_result")
    display_fusion_result = rospy.get_param("~display_fusion_result")
    display_calibration_result = rospy.get_param("~display_calibration_result")
    
    display_2d_modeling_result = rospy.get_param("~display_2d_modeling_result")
    display_obj_pc = rospy.get_param("~display_obj_pc")
    display_gate = rospy.get_param("~display_gate")
    
    display_3d_modeling_result = rospy.get_param("~display_3d_modeling_result")
    display_frame = rospy.get_param("~display_frame")
    display_class = rospy.get_param("~display_class")
    display_id = rospy.get_param("~display_id")
    display_state = rospy.get_param("~display_state")
    
    window_height = cv_images[0].shape[0]
    window_width = cv_images[0].shape[1]
    
    if display_image_raw:
        video_path = 'image_raw.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_image_raw = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    if display_image_segmented:
        video_path = 'image_segmented.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_image_segmented = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    if display_point_clouds_raw:
        video_path = 'point_clouds_raw.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_point_clouds_raw = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    if display_point_clouds_projected:
        video_path = 'point_clouds_projected.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_point_clouds_projected = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    
    if display_segmentation_result:
        video_path = 'segmentation_result.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_segmentation_result = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    if display_fusion_result:
        video_path = 'fusion_result.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_fusion_result = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    if display_calibration_result:
        video_path = 'calibration_result.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_calibration_result = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    
    if display_2d_modeling_result:
        video_path = '2d_modeling_result.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_2d_modeling_result = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    if display_3d_modeling_result:
        video_path = '3d_modeling_result.mp4'
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_3d_modeling_result = cv2.VideoWriter(video_path, video_format, frame_rate, (window_width, window_height), True)
    
    # 使用JET伪彩色绘制点云
    jc = JetColor()
    
    # 发布消息队列设为1，订阅消息队列设为1，并保证订阅消息缓冲区足够大
    # 这样可以实现每次订阅最新的节点消息，避免因队列消息拥挤而导致的雷达点云延迟
    rospy.Subscriber(sub_point_clouds_topic, PointCloud2, point_clouds_callback, queue_size=1, buff_size=52428800)
    pub_marker = rospy.Publisher(pub_marker_topic, MarkerArray, queue_size=1)
    pub_marker_tracked = rospy.Publisher(pub_marker_tracked_topic, MarkerArray, queue_size=1)
    pub_obstacle = rospy.Publisher(pub_obstacle_topic, ObstacleArray, queue_size=1)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
