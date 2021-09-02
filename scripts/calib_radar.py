# -*- coding: UTF-8 -*- 

import numpy as np
import math
import cv2

class CalibRadar:
    def __init__(self, file_path, print_info=True):
        fs = cv2.FileStorage(file_path, cv2.FileStorage_READ)
        
        self.angle_to_front = fs.getNode('RotationAngleToFront').real() * math.pi / 180.0
        self.depression = fs.getNode('DepressionAngle').real() * math.pi / 180.0
        
        if print_info:
            print('Calibration of radar:')
            print('  RotationAngleToFront: %.2frad' % self.angle_to_front)
            print('  DepressionAngle: %.2frad' % self.depression)
            print()
