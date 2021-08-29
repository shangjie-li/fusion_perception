# -*- coding: UTF-8 -*-

import os
import sys
cwd = os.getcwd()
idx = -1
while cwd[idx] != '/':
    idx -= 1
sys.path.append(cwd[:idx] + '/modules/yolov5')

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import numpy as np
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def create_random_color():
    # 功能：产生随机RGB颜色
    # 输出：color <class 'tuple'> 颜色
    
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    color = (r, g, b)
    
    return color

def draw_one_box(img, classname, score, xyxy, color=[0, 255, 0], line_thickness=3):
    # 功能：绘制box
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      classname <class 'str'>
    #      score <class 'float'>
    #      xyxy <class 'numpy.ndarray'> (4,)
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    
    tf = max(line_thickness - 1, 1)  # font thickness
    label = f'{classname} {score:.2f}'
    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class Yolov5Detector():
    def __init__(self):
        # Initialize
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
    
        # Load model
        weights_path = cwd[:idx] + '/modules/yolov5' + '/yolov5s.pt'
        self.model = attempt_load(weights_path, map_location=self.device)  # load FP32 model
        
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(img_size=640, s=self.stride)  # check img_size
        
        if self.half:
            self.model.half()  # to FP16
        
        # Get names
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # Run inference
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
    
    def run(self, img0, items=None, conf_thres=0.25):
        # 功能：运行Yolov5网络以检测图像中的目标
        # 输入：img0 <class 'numpy.ndarray'> (frame_height, frame_width, 3)
        #      items <class 'list'> 保留的目标类别，保留所有时为None，常用值包括0-'person' 1-'bicycle' 2-'car' 3-'motorcycle' 5-'bus' 6-'train' 7-'truck' 9-'traffic light'
        #      conf_thres <class 'float'> 置信度阈值
        # 输出：classes <class 'numpy.ndarray'> (N,) N为目标数量，无目标时为None
        #      scores <class 'numpy.ndarray'> (N,) N为目标数量，无目标时为None
        #      boxes <class 'numpy.ndarray'> (N, 4) N为目标数量，无目标时为None
        
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        with torch.no_grad():
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            pred = self.model(img, augment=False)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45, classes=items)
            
            det = pred[0]  # det <class 'torch.Tensor'> (N, 6) N(>=0)为目标数量
            if len(det):
                classes, scores, boxes = [], [], []
                
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls in reversed(det):  # 按置信度从高到低
                    classes.append(str(self.names[int(cls)]))
                    scores.append(float(conf))
                    boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                
                return np.array(classes), np.array(scores), np.array(boxes)
            else:
                return None, None, None

if __name__ == '__main__':
    img = cv2.imread('/home/lishangjie/fusion-perception_doc/result/2021-05-31/0.jpeg')
    print('img', type(img), img.shape)
    
    t1 = time.time()
    detector = Yolov5Detector()
    t2 = time.time()
    classes, scores, boxes = detector.run(img, items=[0, 2, 5, 7], conf_thres=0.25)
    t3 = time.time()
    
    print()
    if classes is not None:
        print('classes', type(classes), classes.shape, '\n', classes)
        print('scores', type(scores), scores.shape, '\n', scores)
        print('boxes', type(boxes), boxes.shape, '\n', boxes)
        
        num = classes.shape[0]
        for i in range(num):
            draw_one_box(img, classes[i], scores[i], boxes[i], color=[random.randint(0, 255) for _ in range(3)], line_thickness=1)
    
    print()
    print('time cost all:', round(t3 - t1, 3))
    print('time cost per image:', round(t3 - t2, 3))
    
    save_path = 'test.jpeg'
    cv2.imwrite(save_path, img)

