import rospy
from visualization_msgs.msg import Marker, MarkerArray
from perception_msgs.msg import Obstacle, ObstacleArray

import cv2
import math
import numpy as np
from project import project_xyz
from fit_3d_model import transform_2d_point_clouds, compute_obb_vertex_coordinates
from get_ellipse import get_ellipse
from color import JetColor

def get_stamp(header):
    return header.stamp.secs + 0.000000001 * header.stamp.nsecs

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
        marker.type = Marker.CUBE
        
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

def draw_point_clouds_from_main_view(img, xs, ys, zs, mat,
    use_jet=True, min_distance=1.5, max_distance=50, color=(96, 96, 96), radius=2):
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
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = xs, ys, zs
    
    # 将点云投影至图像平面
    height, width = img.shape[0], img.shape[1]
    xyz, uv = project_xyz(xyz, mat, height, width)
    
    if use_jet:
        # 计算点云中各点的距离
        depth = np.sqrt(np.square(xyz[:, 0]) + np.square(xyz[:, 1]))
        depth_range = max_distance - min_distance
        
        jc = JetColor()
        n = jc.num_color
        num = uv.shape[0]
        for pt in range(num):
            # 根据距离选取点云中各点的颜色
            color = jc.get_jet_color(((depth[pt] - min_distance) / depth_range) * n)
            cv2.circle(img, (int(uv[pt][0]), int(uv[pt][1])), radius, color, thickness=-1)
    else:
        num = uv.shape[0]
        for pt in range(num):
            cv2.circle(img, (int(uv[pt][0]), int(uv[pt][1])), radius, color, thickness=-1)
            
    return img

def draw_point_clouds_from_bev_view(img, xs, ys,
    center_alignment=True, color=(96, 96, 96), radius=1):
    # 功能：在图像上绘制点云
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      xs <class 'numpy.ndarray'> (n,) 代表X坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表Y坐标
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    xs = (xs * 10).astype(int)
    ys = (- ys * 10).astype(int)
    
    height, width = img.shape[0], img.shape[1]
    if center_alignment:
        xst, yst = transform_2d_point_clouds(xs, ys, 0, width / 2, height / 2)
    else:
        xst, yst = transform_2d_point_clouds(xs, ys, 0, 0, height / 2)
    
    if radius == 1:
        color = np.array([color[0], color[1], color[2]])
        idxs = (xst >= 0) & (xst < width) & (yst >= 0) & (yst < height)
        xst, yst = xst[idxs], yst[idxs]
        
        num = xst.shape[0]
        for pt in range(num):
            img[int(yst[pt]), int(xst[pt])] = color
            
    else:
        num = xst.shape[0]
        for pt in range(num):
            cv2.circle(img, (int(xst[pt]), int(yst[pt])), radius, color, thickness=-1)
    
    return img

def draw_object_info(img, classname, number, xref, yref, vx, vy, uv_1, uv_2, color,
    display_state=True, l=0, w=0, h=0, phi=0):
    # 功能：在图像上绘制目标信息
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      classname <class 'str'> 类别
    #      number <class 'int'> 编号
    #      xref <class 'float'> X坐标
    #      yref <class 'float'> Y坐标
    #      vx <class 'float'> X方向速度
    #      vy <class 'float'> Y方向速度
    #      uv_1 <class 'numpy.ndarray'> (1, 2) 代表在图像坐标[u, v]处绘制class和id
    #      uv_2 <class 'numpy.ndarray'> (1, 2) 代表在图像坐标[u, v]处绘制state
    #      color <class 'tuple'> RGB颜色
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    u1, v1, u2, v2 = int(uv_1[0, 0]), int(uv_1[0, 1]), int(uv_2[0, 0]), int(uv_2[0, 1])
    
    f_face = cv2.FONT_HERSHEY_DUPLEX
    f_scale = 0.4
    f_thickness = 1
    white = (255, 255, 255)
    
    text_str = classname + ' ' + str(number)
    text_w, text_h = cv2.getTextSize(text_str, f_face, f_scale, f_thickness)[0]
    cv2.rectangle(img, (u1, v1), (u1 + text_w, v1 - text_h - 4), color, -1)
    cv2.putText(img, text_str, (u1, v1 - 3), f_face, f_scale, white, f_thickness, cv2.LINE_AA)
    
    if display_state:
        text_lo = '(%.1fm, %.1fm)' % (xref, yref)
        text_w_lo, _ = cv2.getTextSize(text_lo, f_face, f_scale, f_thickness)[0]
        cv2.rectangle(img, (u2, v2), (u2 + text_w_lo, v2 + text_h + 4), color, -1)
        cv2.putText(img, text_lo, (u2, v2 + text_h + 1), f_face, f_scale, white, f_thickness, cv2.LINE_AA)
        
        text_ve = '(%.1fm/s, %.1fm/s)' % (vx, vy)
        text_w_ve, _ = cv2.getTextSize(text_ve, f_face, f_scale, f_thickness)[0]
        cv2.rectangle(img, (u2, v2 + text_h + 4), (u2 + text_w_ve, v2 + 2 * text_h + 8), color, -1)
        cv2.putText(img, text_ve, (u2, v2 + 2 * text_h + 5), f_face, f_scale, white, f_thickness, cv2.LINE_AA)
        
        text_sc = '(%.1fm, %.1fm, %.1fm, %.1frad)' % (l, w, h, phi)
        text_w_sc, _ = cv2.getTextSize(text_sc, f_face, f_scale, f_thickness)[0]
        cv2.rectangle(img, (u2, v2 + 2 * text_h + 8), (u2 + text_w_sc, v2 + 3 * text_h + 12), color, -1)
        cv2.putText(img, text_sc, (u2, v2 + 3 * text_h + 9), f_face, f_scale, white, f_thickness, cv2.LINE_AA)
    
    return img

def draw_object_model_from_main_view(img, objs, frame, display_frame=True,
    display_state=True, thickness=2):
    # 功能：在图像上绘制目标轮廓
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      objs <class 'list'> 存储目标检测结果
    #      frame <class 'int'> 帧数
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    # 按距离从远到近进行排序
    num = len(objs)
    dds = []
    for i in range(num):
        dd = objs[i].x0 ** 2 + objs[i].y0 ** 2
        dds.append(dd)
    idxs = list(np.argsort(dds))
    
    for i in reversed(idxs):
        # 绘制二维边界框
        xyxy = objs[i].box
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img, c1, c2, objs[i].color, thickness)
        
        # 选取二维边界框中的顶点，绘制目标信息
        uv_1 = np.array([int(xyxy[0]), int(xyxy[1])]).reshape(1, 2)
        uv_2 = np.array([int(xyxy[0]), int(xyxy[3])]).reshape(1, 2)
        
        img = draw_object_info(img, objs[i].classname, objs[i].number,
            objs[i].x0, objs[i].y0, objs[i].vx, objs[i].vy, uv_1, uv_2, objs[i].color,
            display_state, objs[i].l, objs[i].w, objs[i].h, objs[i].phi)
    
    f_face = cv2.FONT_HERSHEY_DUPLEX
    f_scale = 0.4
    f_thickness = 1
    red = (0, 0, 255)
    if display_frame:
        cv2.putText(img, str(frame), (10, 20), f_face, f_scale, red, f_thickness, cv2.LINE_AA)
    
    return img

def draw_object_model_from_bev_view(img, objs, display_gate=True,
    center_alignment=True, axis=True, thickness=2):
    # 功能：在图像上绘制目标轮廓
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      objs <class 'list'> 存储目标检测结果
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    num = len(objs)
    for i in range(num):
        height, width = img.shape[0], img.shape[1]
        h2, w2 = height // 2, width // 2
        if axis:
            black = (0, 0, 0)
            if center_alignment:
                cv2.line(img, (w2, h2), (w2 + 10, h2), black, thickness)
                cv2.line(img, (w2, h2), (w2, h2 - 10), black, thickness)
            else:
                cv2.line(img, (0, h2), (10, h2), black, thickness)
                cv2.line(img, (0, h2), (0, h2 - 10), black, thickness)
        
        xst, yst = compute_obb_vertex_coordinates(objs[i].x0, objs[i].y0, objs[i].l, objs[i].w, objs[i].phi)
        xst, yst = (xst * 10).astype(int), (- yst * 10).astype(int)
        if center_alignment:
            xst, yst = transform_2d_point_clouds(xst, yst, 0, width / 2, height / 2)
        else:
            xst, yst = transform_2d_point_clouds(xst, yst, 0, 0, height / 2)
        for j in range(4):
            pt_1 = (int(xst[j]), int(yst[j]))
            pt_2 = (int(xst[(j + 1) % 4]), int(yst[(j + 1) % 4]))
            cv2.line(img, pt_1, pt_2, objs[i].color, thickness)
        
        if display_gate:
            # 绘制跟踪门椭圆
            a, b = objs[i].tracker.compute_association_gate(objs[i].tracker.gate_threshold)
            x, y = objs[i].x0, objs[i].y0
            xs, ys = get_ellipse(x, y, a, b, 0)
            xs, ys = np.array(xs), np.array(ys)
            
            img = draw_point_clouds_from_bev_view(img, xs, ys, center_alignment, objs[i].color, radius=1)
        
    return img
