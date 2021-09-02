# -*- coding: UTF-8 -*-

import os
import sys
import rospy
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import threading

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from perception_msgs.msg import Obstacle, ObstacleArray

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    
from config import Confs
from config import Topks
from config import Object
from config import Items
from config import BasicColor
from config import LidarColor
from config import RadarColor
from config import PriorColor
from calib import Calib

from mono_estimator import MonoEstimator
from yolact_detector import YolactDetector
from yolact_detector import draw_segmentation_result
from kalman_filter import KalmanFilter4D

from fit_3d_model import fit_2d_obb_using_iterative_procedure
from fit_3d_model import transform_2d_point_clouds
from cluster import cluster_2d_point_clouds
from project import project_xyz, project_xyzlwhp
from numpy_pc2 import pointcloud2_to_xyz_array
from numpy_mmw import mmw_to_xyzlwhp_array

from visualization import get_stamp
from visualization import publish_marker_msg
from visualization import publish_obstacle_msg
from visualization import draw_point_clouds_from_main_view
from visualization import draw_point_clouds_from_bev_view
from visualization import draw_object_model_from_main_view
from visualization import draw_object_model_from_bev_view

image_lock = threading.Lock()
lidar_lock = threading.Lock()
radar_lock = threading.Lock()

def estimate_by_monocular(mono, masks, classes, scores, boxes):
    objs = []
    u, v = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]
    num = classes.shape[0] if classes is not None else 0
    
    for i in range(num):
        obj = Object()
        obj.mask = masks[i].clone() # clone for tensor
        obj.classname = classes[i]
        obj.score = scores[i]
        obj.box = boxes[i].copy() # copy for ndarray
        
        x, y, z = mono.uv_to_xyz(u[i], v[i])
        obj.x0 = z
        obj.y0 = -x
        
        if obj.classname == Items[0]:
            obj.l = 0.5
            obj.w = 0.5
            obj.h = 2.0
        elif obj.classname in Items[1:4]:
            obj.l = 2.0
            obj.w = 2.0
            obj.h = 2.0
        
        obj.phi = 0.0
        obj.color = BasicColor
        objs.append(obj)
        
    return objs

def fuse_lidar(objs, xyz, uv):
    # 功能：点云与图像实例匹配
    # 输入：objs <class 'list'> 存储目标检测结果
    #      xyz <class 'numpy.ndarray'> (n, 4) 代表三维点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      uv  <class 'numpy.ndarray'> (n, 2) 代表图像坐标[u, v]，n为点的数量
    
    xyz_img = np.zeros((win_h, win_w, 3))
    for pt in range(xyz.shape[0]):
        xyz_img[int(uv[pt, 1]), int(uv[pt, 0])] = xyz[pt, :3]
    
    num = len(objs)
    for i in range(num):
        mask = objs[i].mask.byte().cpu().numpy()
        idxs = np.where(mask)
        xs_temp = xyz_img[idxs[0], idxs[1], 0]
        ys_temp = xyz_img[idxs[0], idxs[1], 1]
        
        idxs = np.logical_and(xs_temp[:], ys_temp[:])
        xs = xs_temp[idxs]
        ys = ys_temp[idxs]
        
        if xs.shape[0] > 0:
            xsc, ysc, is_clustered = cluster_2d_point_clouds(xs, ys)
            if is_clustered:
                ysc = ysc * math.cos(sensor_depression)
                xsc, ysc = transform_2d_point_clouds(xsc, ysc, rotation_lidar_to_front, 0, 0)
                x0, y0, l, w, phi, _, _, _ = fit_2d_obb_using_iterative_procedure(xsc, ysc)
                
                objs[i].color = LidarColor
                objs[i].refined_by_lidar = True
                objs[i].x0, objs[i].y0, objs[i].phi = x0, y0, phi
                
                name = objs[i].classname
                if name == Items[0]:
                    objs[i].phi = 0.0
                elif name == Items[3]:
                    objs[i].l, objs[i].w, objs[i].h = l, w, 3.0
                elif name in Items[1:3]:
                    objs[i].l, objs[i].w, objs[i].h = l, w, 2.0

def fuse_radar(objs, xyz, uv, lwhp):
    # 功能：雷达点云与图像实例匹配
    # 输入：objs <class 'list'> 存储目标检测结果
    #      xyz <class 'numpy.ndarray'> (n, 4) 代表三维雷达点云的齐次坐标[x, y, z, 1]，n为点的数量
    #      uv  <class 'numpy.ndarray'> (n, 2) 代表图像坐标[u, v]，n为点的数量
    
    n = uv.shape[0]
    if n == 0:
        return
    
    objs_uv = []
    num = len(objs)
    for i in range(num):
        u = (objs[i].box[0] + objs[i].box[2]) / 2
        v = (objs[i].box[1] + objs[i].box[3]) / 2
        objs_uv.append([u, v])
    objs_uv = np.array(objs_uv)
    objs_uv = np.expand_dims(obj_uv, axis=0).repeat(n, axis=0) # [n, num, 2]
    
    r_uv = uv.copy()
    r_uv = np.expand_dims(r_uv, axis=1).repeat(num, axis=1) # [n, num, 2]
    
    dis = (r_uv[:, :, 0] - objs_uv[:, :, 0]) ** 2 + (r_uv[:, :, 1] - objs_uv[:, :, 1]) ** 2 # [n, num]
    best_objs_for_radar_dis = dis.min(axis=1) # [n]
    best_objs_for_radar_idx = dis.argmin(axis=1) # [n]
    
    for j in range(n):
        d = best_objs_for_radar_dis[j]
        i = best_objs_for_radar_idx[j]
        if d < 100 ** 2:
            x0, y0 = xyz[j, 0], xyz[j, 1]
            l, w, h, phi = lwhp[j]
            
            if objs[i].refined_by_lidar:
                objs[i].color = PriorColor
                objs[i].refined_by_radar = True
            else:
                objs[i].color = RadarColor
                objs[i].refined_by_lidar = True
                objs[i].x0, objs[i].y0, objs[i].phi = x0, y0, phi
                
                name = objs[i].classname
                if name == Items[0]:
                    objs[i].phi = 0.0
                elif name == Items[3]:
                    objs[i].l, objs[i].w, objs[i].h = l, w, 3.0
                elif name in Items[1:3]:
                    objs[i].l, objs[i].w, objs[i].h = l, w, 2.0

def track(number, objs_tracked, objs_temp, objs_detected, blind_update_limit, frame_rate):
    # 数据关联与跟踪
    num = len(objs_tracked)
    for j in range(num):
        # 计算残差加权范数
        flag, idx, ddm = False, 0, float('inf')
        n = len(objs_detected)
        for k in range(n):
            zx = objs_detected[k].x0
            zy = objs_detected[k].y0
            dd = objs_tracked[j].tracker.compute_the_residual(zx, zy)
            if dd < ddm and dd < objs_tracked[j].tracker.gate_threshold:
                flag, idx, ddm = True, k, dd
        
        if flag and objs_tracked[j].classname == objs_detected[idx].classname:
            # 匹配成功，预测并更新
            objs_tracked[j].tracker.predict()
            objs_tracked[j].tracker.update(objs_detected[idx].x0, objs_detected[idx].y0)
            
            # 继承检测结果中的参数
            objs_tracked[j].mask = objs_detected[idx].mask.clone()
            objs_tracked[j].score = objs_detected[idx].score
            objs_tracked[j].box = objs_detected[idx].box.copy()
            objs_tracked[j].l = objs_detected[idx].l
            objs_tracked[j].w = objs_detected[idx].w
            objs_tracked[j].h = objs_detected[idx].h
            objs_tracked[j].phi = objs_detected[idx].phi
            objs_tracked[j].color = objs_detected[idx].color
            objs_tracked[j].refined_by_lidar = objs_detected[idx].refined_by_lidar
            objs_tracked[j].refined_by_radar = objs_detected[idx].refined_by_radar
            
            # 修改更新中断次数
            objs_tracked[j].tracker_blind_update = 0
            objs_detected.pop(idx)
        else:
            # 匹配不成功，只预测
            objs_tracked[j].tracker.predict()
            
            # 修改更新中断次数
            objs_tracked[j].tracker_blind_update += 1
        
        # 输出滤波值
        objs_tracked[j].x0 = objs_tracked[j].tracker.xx[0, 0]
        objs_tracked[j].vx = objs_tracked[j].tracker.xx[1, 0]
        objs_tracked[j].y0 = objs_tracked[j].tracker.xx[2, 0]
        objs_tracked[j].vy = objs_tracked[j].tracker.xx[3, 0]
        
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
        # 计算残差加权范数
        flag, idx, ddm = False, 0, float('inf')
        n = len(objs_detected)
        for k in range(n):
            zx = objs_detected[k].x0
            zy = objs_detected[k].y0
            dd = objs_temp[j].tracker.compute_the_residual(zx, zy)
            if dd < ddm and dd < objs_temp[j].tracker.gate_threshold:
                flag, idx, ddm = True, k, dd
        
        if flag and objs_temp[j].classname == objs_detected[idx].classname:
            zx = objs_detected[idx].x0
            zy = objs_detected[idx].y0
            x = objs_temp[j].tracker.xx[0, 0]
            y = objs_temp[j].tracker.xx[2, 0]
            vx = (zx - x) / objs_temp[j].tracker.ti
            vy = (zy - y) / objs_temp[j].tracker.ti
            
            # 继承检测结果中的参数
            objs_temp[j].mask = objs_detected[idx].mask.clone()
            objs_temp[j].score = objs_detected[idx].score
            objs_temp[j].box = objs_detected[idx].box.copy()
            objs_temp[j].l = objs_detected[idx].l
            objs_temp[j].w = objs_detected[idx].w
            objs_temp[j].h = objs_detected[idx].h
            objs_temp[j].phi = objs_detected[idx].phi
            objs_temp[j].color = objs_detected[idx].color
            objs_temp[j].refined_by_lidar = objs_detected[idx].refined_by_lidar
            objs_temp[j].refined_by_radar = objs_detected[idx].refined_by_radar
            
            # 对跟踪的位置、速度重新赋值
            objs_temp[j].x0 = objs_temp[j].tracker.xx[0, 0] = zx
            objs_temp[j].vx = objs_temp[j].tracker.xx[1, 0] = vx
            objs_temp[j].y0 = objs_temp[j].tracker.xx[2, 0] = zy
            objs_temp[j].vy = objs_temp[j].tracker.xx[3, 0] = vy
            
            # 增加ID
            number += 1
            objs_temp[j].number = number
            objs_tracked.append(objs_temp[j])
            objs_detected.pop(idx)
    
    # 增广临时跟踪列表
    objs_temp = objs_detected
    num = len(objs_temp)
    for j in range(num):
        # 初始化卡尔曼滤波器，对目标进行跟踪
        objs_temp[j].tracker = KalmanFilter4D(
            1 / frame_rate, objs_temp[j].x0, objs_temp[j].vx, objs_temp[j].y0, objs_temp[j].vy,
            sigma_ax=1, sigma_ay=1, sigma_ox=0.1, sigma_oy=0.1, gate_threshold=400
        )
    
    return number, objs_tracked, objs_temp

def image_callback(image):
    global image_stamps, image_frames
    stamp = get_stamp(image.header)
    data = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    
    image_lock.acquire()
    if len(image_stamps) < 30:
        image_stamps.append(stamp)
        image_frames.append(data)
    else:
        image_stamps.pop(0)
        image_frames.pop(0)
        image_stamps.append(stamp)
        image_frames.append(data)
    image_lock.release()

def lidar_callback(lidar):
    global lidar_stamps, lidar_frames
    stamp = get_stamp(lidar.header)
    data = pointcloud2_to_xyz_array(lidar, remove_nans=True)
    
    lidar_lock.acquire()
    if len(lidar_stamps) < 1:
        lidar_stamps.append(stamp)
        lidar_frames.append(data)
    else:
        lidar_stamps.pop(0)
        lidar_frames.pop(0)
        lidar_stamps.append(stamp)
        lidar_frames.append(data)
    lidar_lock.release()

def radar_callback(radar):
    global radar_stamps, radar_frames
    stamp = time.time()
    data = mmw_to_xyzlwhp_array(radar)
    
    radar_lock.acquire()
    if len(radar_stamps) < 1:
        radar_stamps.append(stamp)
        radar_frames.append(data)
    else:
        radar_stamps.pop(0)
        radar_frames.pop(0)
        radar_stamps.append(stamp)
        radar_frames.append(data)
    radar_lock.release()

def fusion_callback(event):
    time_all_start = time.time()
    global objs_tracked, objs_temp, number, frame
    frame += 1
    
    global image_stamps, image_frames
    global lidar_stamps, lidar_frames
    global radar_stamps, radar_frames
    
    lidar_valid = False
    lidar_lock.acquire()
    if len(lidar_frames) > 0:
        lidar_stamp = lidar_stamps[-1]
        lidar_frame = lidar_frames[-1].copy()
        lidar_valid = True
    lidar_lock.release()
    
    radar_valid = False
    radar_lock.acquire()
    if len(radar_frames) > 0:
        radar_stamp = radar_stamps[-1]
        radar_frame = radar_frames[-1].copy()
        if time.time() - radar_stamp < 1.0:
            radar_valid = True
        else:
            radar_stamps.pop(0)
            radar_frames.pop(0)
    radar_lock.release()
    
    image_lock.acquire()
    if lidar_valid:
        stamp_err = [abs(x - lidar_stamp) for x in image_stamps]
        idx = stamp_err.index(min(stamp_err))
        image_stamp = image_stamps[idx]
        image_frame = image_frames[idx].copy()
    else:
        image_stamp = image_stamps[-1]
        image_frame = image_frames[-1].copy()
    image_lock.release()
    
    current_image = image_frame.copy()
    current_image = current_image[:, :, ::-1] # to BGR
    
    # 实例分割与视觉估计
    time_segmentation_start = time.time()
    masks, classes, scores, boxes = detector.run(image_frame, items=Items,
        score_thresholds=Confs, top_ks=Topks)
    objs = estimate_by_monocular(mono, masks, classes, scores, boxes)
    time_segmentation = time.time() - time_segmentation_start
    
    # 融合lidar
    time_lidar_fusion_start = time.time()
    if lidar_valid:
        lidar_mat = calib_lidar.projection_s2i
        lidar_xyz, lidar_uv = project_xyz(lidar_frame, lidar_mat, win_h, win_w)
        fuse_lidar(objs, lidar_xyz, lidar_uv)
    time_lidar_fusion = time.time() - time_lidar_fusion_start
    
    # 融合radar
    time_radar_fusion_start = time.time()
    if radar_valid:
        radar_mat = calib_radar.projection_s2i
        radar_xyz, radar_uv, radar_lwhp = project_xyzlwhp(radar_frame, radar_mat, win_h, win_w)
        fuse_radar(objs, radar_xyz, radar_uv, radar_lwhp)
    time_radar_fusion = time.time() - time_radar_fusion_start
    
    # 目标跟踪
    time_tracking_start = time.time()
    if processing_mode == 'DT':
        number, objs_tracked, objs_temp = track(number, objs_tracked, objs_temp, objs,
            blind_update_limit, frame_rate)
        if number >= max_id:
            number = 0
    time_tracking = time.time() - time_tracking_start
    
    # 模式切换与发送消息
    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()
    
    if processing_mode == 'D':
        publish_marker_msg(pub_marker, header, frame_rate, objs, random_number=True)
        publish_obstacle_msg(pub_obstacle, header, frame_rate, objs, random_number=True)
    elif processing_mode == 'DT':
        objs = objs_tracked
        publish_marker_msg(pub_marker, header, frame_rate, objs, random_number=False)
        publish_obstacle_msg(pub_obstacle, header, frame_rate, objs, random_number=False)
    
    # 可视化
    time_display_start = time.time()
    if display_image_raw:
        win_image_raw = current_image.copy()
    if display_image_segmented:
        win_image_segmented = current_image.copy()
        num = len(objs)
        for i in range(num):
            if objs[i].tracker_blind_update > 0:
                continue
            win_image_segmented = draw_segmentation_result(
                win_image_segmented, objs[i].mask, objs[i].classname, objs[i].score,
                objs[i].box, BasicColor
            )
    if display_lidar_projected:
        win_lidar_projected = current_image.copy()
        if lidar_valid:
            win_lidar_projected = draw_point_clouds_from_main_view(
                win_lidar_projected, lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2],
                lidar_mat, use_jet=True, radius=2
            )
    if display_radar_projected:
        win_radar_projected = current_image.copy()
        if radar_valid:
            win_radar_projected = draw_point_clouds_from_main_view(
                win_radar_projected, radar_xyz[:, 0], radar_xyz[:, 1], radar_xyz[:, 2],
                radar_mat, use_jet=False, color=RadarColor, radius=8
            )
    if display_2d_modeling:
        win_2d_modeling = np.ones((win_h, win_w, 3), dtype=np.uint8) * 255
        if lidar_valid:
            win_2d_modeling = draw_point_clouds_from_bev_view(
                win_2d_modeling, lidar_xyz[:, 0], lidar_xyz[:, 1],
                center_alignment=False, color=(158, 158, 158), radius=1
            )
        if radar_valid:
            win_2d_modeling = draw_point_clouds_from_bev_view(
                win_2d_modeling, radar_xyz[:, 0], radar_xyz[:, 1],
                center_alignment=False, color=RadarColor, radius=4
            )
        win_2d_modeling = draw_object_model_from_bev_view(
            win_2d_modeling, objs, display_gate,
            center_alignment=False, axis=True, thickness=2
        )
    if display_3d_modeling:
        win_3d_modeling = current_image.copy()
        win_3d_modeling = draw_object_model_from_main_view(
            win_3d_modeling, objs, frame, display_frame,
            display_obj_state, thickness=2
        )
    
    # 显示与保存
    if display_image_raw:
        cv2.namedWindow("image_raw", cv2.WINDOW_NORMAL)
        cv2.imshow("image_raw", win_image_raw)
        v_image_raw.write(win_image_raw)
    if display_image_segmented:
        cv2.namedWindow("image_segmented", cv2.WINDOW_NORMAL)
        cv2.imshow("image_segmented", win_image_segmented)
        v_image_segmented.write(win_image_segmented)
    if display_lidar_projected:
        cv2.namedWindow("lidar_projected", cv2.WINDOW_NORMAL)
        cv2.imshow("lidar_projected", win_lidar_projected)
        v_lidar_projected.write(win_lidar_projected)
    if display_radar_projected:
        cv2.namedWindow("radar_projected", cv2.WINDOW_NORMAL)
        cv2.imshow("radar_projected", win_radar_projected)
        v_radar_projected.write(win_radar_projected)
    if display_2d_modeling:
        cv2.namedWindow("2d_modeling", cv2.WINDOW_NORMAL)
        cv2.imshow("2d_modeling", win_2d_modeling)
        v_2d_modeling.write(win_2d_modeling)
    if display_3d_modeling:
        cv2.namedWindow("3d_modeling", cv2.WINDOW_NORMAL)
        cv2.imshow("3d_modeling", win_3d_modeling)
        v_3d_modeling.write(win_3d_modeling)
    
    # 显示窗口时按Esc键终止程序
    display = [display_image_raw, display_image_segmented, display_lidar_projected,
        display_radar_projected, display_2d_modeling, display_3d_modeling].count(True) > 0
    if display and cv2.waitKey(1) == 27:
        if display_image_raw:
            cv2.destroyWindow("image_raw")
            v_image_raw.release()
            print("Save video of image_raw.")
        if display_image_segmented:
            cv2.destroyWindow("image_segmented")
            v_image_segmented.release()
            print("Save video of image_segmented.")
        if display_lidar_projected:
            cv2.destroyWindow("lidar_projected")
            v_lidar_projected.release()
            print("Save video of lidar_projected.")
        if display_radar_projected:
            cv2.destroyWindow("radar_projected")
            v_radar_projected.release()
            print("Save video of radar_projected.")
        if display_2d_modeling:
            cv2.destroyWindow("2d_modeling")
            v_2d_modeling.release()
            print("Save video of 2d_modeling.")
        if display_3d_modeling:
            cv2.destroyWindow("3d_modeling")
            v_3d_modeling.release()
            print("Save video of 3d_modeling.")
        print("\nReceived the shutdown signal.\n")
        rospy.signal_shutdown("Everything is over now.")
    time_display = time.time() - time_display_start
    
    # 记录耗时情况
    time_segmentation = round(time_segmentation, 3)
    time_lidar_fusion = round(time_lidar_fusion, 3) if lidar_valid else None
    time_radar_fusion = round(time_radar_fusion, 3) if radar_valid else None
    time_tracking = round(time_tracking, 3) if processing_mode == 'DT' else None
    time_display = round(time_display, 3) if display else None
    time_all = round(time.time() - time_all_start, 3)
    
    if print_time:
        print("image_stamp          ", image_stamp)
        print("time of segmentation ", time_segmentation)
        print("time of lidar fusion ", time_lidar_fusion)
        print("time of radar fusion ", time_radar_fusion)
        print("time of tracking     ", time_tracking)
        print("time of display      ", time_display)
        print("time of all          ", time_all)
        print()
    
    if print_objects_info:
        num = len(objs)
        for j in range(num):
            n = objs[j].number if processing_mode == 'DT' else j
            print('Object %d x0:%.2f y0:%.2f %s' % (n, objs[j].x0, objs[j].y0, objs[j].classname))
        print()
    
    if record_time:
        time_lidar_fusion = 0.0 if time_lidar_fusion is None else time_lidar_fusion
        time_radar_fusion = 0.0 if time_radar_fusion is None else time_radar_fusion
        time_tracking = 0.0 if time_tracking is None else time_tracking
        time_display = 0.0 if time_display is None else time_display
        with open(filename_time, 'a') as fob:
            content = 'frame:%d amount:%d segmentation:%.3f fusion:%.3f ' + \
                'tracking:%.3f display:%.3f all:%.3f'
            fob.write(
                content % (
                frame, len(objs), time_segmentation, time_lidar_fusion + time_radar_fusion,
                time_tracking, time_display, time_all)
            )
            fob.write('\n')
            
    if record_objects_info:
        num = len(objs)
        for j in range(num):
            n = objs[j].number if processing_mode == 'DT' else j
            with open(filename_objects_info, 'a') as fob:
                content = 'stamp:%.3f frame:%d id:%d xref:%.3f vx:%.3f yref:%.3f vy:%.3f ' + \
                    'x0:%.3f y0:%.3f z0:%.3f l:%.3f w:%.3f h:%.3f phi:%.3f'
                fob.write(
                    content % (
                    image_stamp, frame, n, objs[j].x0, objs[j].vx, objs[j].y0, objs[j].vy,
                    objs[j].x0, objs[j].y0, objs[j].z0, objs[j].l, objs[j].w, objs[j].h, objs[j].phi)
                )
                fob.write('\n')

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("dt")
    
    # 记录耗时
    print_time = rospy.get_param("~print_time")
    record_time = rospy.get_param("~record_time")
    if record_time:
        filename_time = 'time_cost.txt'
        with open(filename_time, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    # 记录结果
    print_objects_info = rospy.get_param("~print_objects_info")
    record_objects_info = rospy.get_param("~record_objects_info")
    if record_objects_info:
        filename_objects_info = 'result.txt'
        with open(filename_objects_info, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    use_lidar = rospy.get_param("~use_lidar")
    use_radar = rospy.get_param("~use_radar")
    
    sensor_height = rospy.get_param("~sensor_height")
    sensor_depression = rospy.get_param("~sensor_depression") * math.pi / 180.
    if use_lidar:
        rotation_lidar_to_front = rospy.get_param("~rotation_lidar_to_front") * math.pi / 180.
    if use_radar:
        rotation_radar_to_front = rospy.get_param("~rotation_radar_to_front") * math.pi / 180.
    
    # 设置ROS消息名称
    sub_image_topic = rospy.get_param("~sub_image_topic")
    if use_lidar:
        sub_lidar_topic = rospy.get_param("~sub_lidar_topic")
    if use_radar:
        sub_radar_topic = rospy.get_param("~sub_radar_topic")
    
    pub_marker_topic = rospy.get_param("~pub_marker_topic")
    pub_obstacle_topic = rospy.get_param("~pub_obstacle_topic")
    frame_id = rospy.get_param("~frame_id")
    
    # 设置标定参数
    cwd = os.getcwd()
    calib_pth_idx = -1
    while cwd[calib_pth_idx] != '/':
        calib_pth_idx -= 1
    
    calibration_image_file = rospy.get_param("~calibration_image_file")
    f_path = cwd[:calib_pth_idx] + '/conf/' + calibration_image_file
    if not os.path.exists(f_path):
            raise ValueError("%s doesn't exist." % (f_path))
    mono = MonoEstimator(f_path, sensor_height, sensor_depression, print_info=True)
    
    if use_lidar:
        calibration_lidar_file = rospy.get_param("~calibration_lidar_file")
        f_path = cwd[:calib_pth_idx] + '/conf/' + calibration_lidar_file
        if not os.path.exists(f_path):
            raise ValueError("%s doesn't exist." % (f_path))
        calib_lidar = Calib(f_path, print_mat=True, sensor_name='lidar')
    
    if use_radar:
        calibration_radar_file = rospy.get_param("~calibration_radar_file")
        f_path = cwd[:calib_pth_idx] + '/conf/' + calibration_radar_file
        if not os.path.exists(f_path):
            raise ValueError("%s doesn't exist." % (f_path))
        calib_radar = Calib(f_path, print_mat=True, sensor_name='radar')
    
    # 初始化YolactDetector
    detector = YolactDetector()
    
    # 准备图像序列
    print('Waiting for topic...')
    image_stamps, image_frames = [], []
    rospy.Subscriber(sub_image_topic, Image, image_callback, queue_size=1, buff_size=52428800)
    while len(image_stamps) < 30:
        time.sleep(1)
    
    # 发布消息队列设为1，订阅消息队列设为1，并保证订阅消息缓冲区足够大
    # 这样可以实现每次订阅最新的节点消息，避免因队列消息拥挤而导致的雷达点云延迟
    
    # 准备激光雷达点云序列
    lidar_stamps, lidar_frames = [], []
    if use_lidar:
        rospy.Subscriber(sub_lidar_topic, PointCloud2, lidar_callback, queue_size=1, buff_size=52428800)
    
    # 准备毫米波雷达序列
    radar_stamps, radar_frames = [], []
    if use_radar:
        rospy.Subscriber(sub_radar_topic, MarkerArray, radar_callback, queue_size=1, buff_size=52428800)
    
    # 完成所需传感器数据序列
    print('  Done.\n')
    
    # 初始化检测列表、跟踪列表、临时跟踪列表
    objs_tracked = []
    objs_temp = []
    number = 0
    frame = 0
    
    # 设置更新中断次数的限制、工作帧率、最大ID
    blind_update_limit = rospy.get_param("~blind_update_limit")
    frame_rate = rospy.get_param("~frame_rate")
    max_id = rospy.get_param("~max_id")
    
    # 设置显示窗口
    display_image_raw = rospy.get_param("~display_image_raw")
    display_image_segmented = rospy.get_param("~display_image_segmented")
    
    if use_lidar:
        display_lidar_projected = rospy.get_param("~display_lidar_projected")
    else:
        display_lidar_projected = False
    
    if use_radar:
        display_radar_projected = rospy.get_param("~display_radar_projected")
    else:
        display_radar_projected = False
    
    display_2d_modeling = rospy.get_param("~display_2d_modeling")
    display_gate = rospy.get_param("~display_gate")
    
    display_3d_modeling = rospy.get_param("~display_3d_modeling")
    display_frame = rospy.get_param("~display_frame")
    display_obj_state = rospy.get_param("~display_obj_state")
    
    win_h = image_frames[0].shape[0]
    win_w = image_frames[0].shape[1]
    
    if display_image_raw:
        v_path = 'image_raw.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image_raw = cv2.VideoWriter(v_path, v_format, frame_rate, (win_w, win_h), True)
    if display_image_segmented:
        v_path = 'image_segmented.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image_segmented = cv2.VideoWriter(v_path, v_format, frame_rate, (win_w, win_h), True)
    if display_lidar_projected:
        v_path = 'lidar_projected.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_lidar_projected = cv2.VideoWriter(v_path, v_format, frame_rate, (win_w, win_h), True)
    if display_radar_projected:
        v_path = 'radar_projected.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_radar_projected = cv2.VideoWriter(v_path, v_format, frame_rate, (win_w, win_h), True)
    if display_2d_modeling:
        v_path = '2d_modeling.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_2d_modeling = cv2.VideoWriter(v_path, v_format, frame_rate, (win_w, win_h), True)
    if display_3d_modeling:
        v_path = '3d_modeling.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_3d_modeling = cv2.VideoWriter(v_path, v_format, frame_rate, (win_w, win_h), True)
    
    # 启动数据融合线程
    processing_mode = rospy.get_param("~processing_mode")
    if processing_mode == 'D':
        display_gate = False
    elif processing_mode == 'DT':
        display_gate = display_gate
    else:
        raise ValueError("processing_mode must be 'D' of 'DT'.")
        
    rospy.Timer(rospy.Duration(1 / frame_rate), fusion_callback)
    pub_marker = rospy.Publisher(pub_marker_topic, MarkerArray, queue_size=1)
    pub_obstacle = rospy.Publisher(pub_obstacle_topic, ObstacleArray, queue_size=1)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
