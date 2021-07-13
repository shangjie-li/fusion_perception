# -*- coding: UTF-8 -*- 

import numpy as np
import rospy
import time
import math
import os
import sys

from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from perception_msgs.msg import Obstacle, ObstacleArray

from obj import Object

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

def radar_callback(radar):
    global radar_valid, radar_objs
    
    num = len(radar.markers) 
    if num > 0:
        radar_valid = True
        radar_objs = []
        
        for i in range(num):
            m = radar.markers[i]
            obj = Object()
            obj.classname = m.ns
            
            obj.x0 = m.pose.position.x
            obj.y0 = m.pose.position.y
            obj.z0 = m.pose.position.z
            obj.l = m.scale.x
            obj.w = m.scale.y
            obj.h = m.scale.z
            
            alpha = m.pose.orientation.z / m.pose.orientation.w
            obj.phi = 2 * math.atan(alpha)
            obj.has_orientation = True
            
            obj.xref = m.pose.position.x
            obj.yref = m.pose.position.y
            obj.vx = 0
            obj.vy = 0
            
            obj.number = m.id
            
            radar_objs.append(obj)
    else:
        radar_valid = False
        radar_msg = None

def dt_callback(dt):
    global dt_valid, dt_objs
    
    num = len(dt.obstacles)
    if num > 0:
        dt_valid = True
        dt_objs = []
        
        for i in range(num):
            m = dt.obstacles[i]
            obj = Object()
            obj.classname = m.ns
            
            obj.x0 = m.pose.position.x
            obj.y0 = m.pose.position.y
            obj.z0 = m.pose.position.z
            obj.l = m.scale.x
            obj.w = m.scale.y
            obj.h = m.scale.z
            
            alpha = m.pose.orientation.z / m.pose.orientation.w
            obj.phi = 2 * math.atan(alpha)
            obj.has_orientation = True
            
            obj.xref = m.pose.position.x
            obj.yref = m.pose.position.y
            obj.vx = m.vx
            obj.vy = m.vy
            
            obj.number = m.id
            
            dt_objs.append(obj)
    else:
        dt_valid = False
        dt_msg = None

def combine_objs(objs1, objs2, distance_threshold=0.5):
    n1, n2 = len(objs1), len(objs2)
    objs = []
    
    if n1 == 0 and n2 == 0:
        return []
    elif n1 > 0 and n2 == 0:
        return objs1
    elif n2 > 0 and n1 == 0:
        return objs2
    
    for i in range(n1):
        obj = Object()
        obj.classname = objs1[i].classname
        
        obj.x0 = objs1[i].x0
        obj.y0 = objs1[i].y0
        obj.z0 = objs1[i].z0
        obj.l = objs1[i].l
        obj.w = objs1[i].w
        obj.h = objs1[i].h
        
        obj.phi = objs1[i].phi
        obj.has_orientation = objs1[i].has_orientation
        
        obj.xref = objs1[i].xref
        obj.yref = objs1[i].yref
        obj.vx = objs1[i].vx
        obj.vy = objs1[i].vy
        
        obj.number = objs1[i].number
        objs.append(obj)
    
    objs2_ass_list = [0] * n2
    
    for i in range(n1):
        x1 = objs1[i].x0
        y1 = objs1[i].y0
        
        for j in range(n2):
            x2 = objs2[j].x0
            y2 = objs2[j].y0
            dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dis < distance_threshold:
                objs2_ass_list[j] = 1

    for i in range(n2):
        if objs2_ass_list[i] == 1:
            continue
        obj = Object()
        obj.classname = objs2[i].classname
        
        obj.x0 = objs2[i].x0
        obj.y0 = objs2[i].y0
        obj.z0 = objs2[i].z0
        obj.l = objs2[i].l
        obj.w = objs2[i].w
        obj.h = objs2[i].h
        
        obj.phi = objs2[i].phi
        obj.has_orientation = objs2[i].has_orientation
        
        obj.xref = objs2[i].xref
        obj.yref = objs2[i].yref
        obj.vx = objs2[i].vx
        obj.vy = objs2[i].vy
        
        obj.number = objs2[i].number
        objs.append(obj)

    return objs

def timer_callback(event):
    global radar_valid, radar_objs
    global dt_valid, dt_objs
    objs = []
    
    if radar_valid and dt_valid:
        objs = combine_objs(radar_objs, dt_objs)
    elif radar_valid and not dt_valid:
        objs = radar_objs
    elif dt_valid and not radar_valid:
        objs = dt_objs
    
    header = Header()
    header.frame_id = 'base_link'
    header.stamp = rospy.Time.now()
    
    publish_marker_msg(pub_marker, header, frame_rate, objs, False)
    publish_obstacle_msg(pub_obstacle, header, frame_rate, objs, False)
    
    print(event)

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("fp")
    
    # 设置ROS消息名称
    sub_radar_topic = "/radar_topic"
    sub_dt_topic = "/dt"
    
    radar_valid, radar_objs = False, None
    dt_valid, dt_objs = False, None
    
    pub_marker_topic = "/fusion_marker_array"
    pub_obstacle_topic = "/fusion_obstacle_array"
    
    # 发布消息队列设为1，订阅消息队列设为1，并保证订阅消息缓冲区足够大
    # 这样可以实现每次订阅最新的节点消息，避免因队列消息拥挤而导致的雷达点云延迟
    rospy.Subscriber(sub_radar_topic, MarkerArray, radar_callback, queue_size=1, buff_size=52428800)
    rospy.Subscriber(sub_dt_topic, ObstacleArray, dt_callback, queue_size=1, buff_size=52428800)
    
    frame_rate = 10
    rospy.Timer(rospy.Duration(1 / frame_rate), timer_callback)
    
    pub_marker = rospy.Publisher(pub_marker_topic, MarkerArray, queue_size=1)
    pub_obstacle = rospy.Publisher(pub_obstacle_topic, ObstacleArray, queue_size=1)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
