# -*- coding: UTF-8 -*- 

import numpy as np
import math
import cv2

class Calib:
    def __init__(self, file_path, print_mat=True):
        fs = cv2.FileStorage(file_path, cv2.FileStorage_READ)
        if(not fs.isOpened()):
            print("No file: %s" %file_path)
        
        tr_raw = fs.getNode('LidarToCameraMat').mat()
        # x轴旋转矩阵(gamma) y轴旋转矩阵(beta) z轴旋转矩阵(alpha)
        gamma = fs.getNode('RotationAngleX').real() * math.pi / 180
        rx = np.array([[1, 0, 0, 0],
                       [0, math.cos(gamma), -math.sin(gamma), 0],
                       [0, math.sin(gamma), math.cos(gamma), 0],
                       [0, 0, 0, 1]], np.float32)
        beta = fs.getNode('RotationAngleY').real() * math.pi / 180
        ry = np.array([[math.cos(beta), 0, math.sin(beta), 0],
                       [0, 1, 0, 0],
                       [-math.sin(beta), 0, math.cos(beta), 0],
                       [0, 0, 0, 1]], np.float32)
        alpha = fs.getNode('RotationAngleZ').real() * math.pi / 180
        rz = np.array([[math.cos(alpha), -math.sin(alpha), 0, 0],
                       [math.sin(alpha), math.cos(alpha), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],],np.float32)
        tr = rz.dot(ry.dot(rx.dot(tr_raw)))
        if print_mat:
            print("transformation_lidar_to_camera:")
            print(tr)
            print()
        self.transformation_lidar_to_camera = tr
        self.projection_l2i = fs.getNode('ProjectionMat').mat().dot(tr)
        self.projection_c2i = fs.getNode('ProjectionMat').mat()
