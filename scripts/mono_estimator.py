# -*- coding: UTF-8 -*- 

import numpy as np
import math
import cv2
from math import sin, cos

class MonoEstimator():
    def __init__(self, file_path, print_info=True):
        fs = cv2.FileStorage(file_path, cv2.FileStorage_READ)
        
        mat = fs.getNode('ProjectionMat').mat()
        self.fx = int(mat[0, 0])
        self.fy = int(mat[1, 1])
        self.u0 = int(mat[0, 2])
        self.v0 = int(mat[1, 2])
        
        self.height = fs.getNode('Height').real()
        self.depression = fs.getNode('DepressionAngle').real() * math.pi / 180.0
        
        if print_info:
            print('Calibration of camera:')
            print('  Parameters: fx(%d) fy(%d) u0(%d) v0(%d)' % (self.fx, self.fy, self.u0, self.v0))
            print('  Height: %.2fm' % self.height)
            print('  DepressionAngle: %.2frad' % self.depression)
            print()
        
    def box_to_xyz(self, box, height):
        # 由图像坐标计算世界坐标
        # 世界坐标系的原点位于相机中心，XZ平面平行于地面，X指向相机右侧，Z指向相机前方
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2

        if abs(y2 - y1) == 0:
            return float('inf'), float('inf'), float('inf')

        z = self.fy * height / abs(y2 - y1)
        x = z * (u - self.u0) / self.fx
        y = z * (v - self.v0) / self.fy
        return x, y, z
