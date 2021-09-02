# -*- coding: UTF-8 -*-

Items = ['person', 'car', 'truck', 'bus']
Confs = [0.15, 0.25, 0.25, 0.25]
Topks = [10, 10, 10, 10]

BasicColor = (158, 158, 158)
LidarColor = (255, 0, 0)
RadarColor = (0, 255, 0)
PriorColor = (0, 0, 255)

class Object():
    def __init__(self):
        self.mask = None                 # <class 'torch.Tensor'> torch.Size([frame_height, frame_width])
        self.classname = None            # <class 'str'>
        self.score = None                # <class 'float'>
        self.box = None                  # <class 'numpy.ndarray'> (4,)
        
        self.x0 = 0                      # <class 'float'>
        self.y0 = 0                      # <class 'float'>
        self.z0 = 0                      # <class 'float'>
        self.l = 0                       # <class 'float'>
        self.w = 0                       # <class 'float'>
        self.h = 0                       # <class 'float'>
        self.phi = 0                     # <class 'float'> [0, pi)
        
        self.vx = 0                      # <class 'float'>
        self.vy = 0                      # <class 'float'>
        
        self.number = 0                  # <class 'int'>
        self.color = (0, 255, 0)         # <class 'tuple'>
        
        self.tracker = None
        self.tracker_blind_update = 0
        
        self.refined_by_lidar = False
        self.refined_by_radar = False
        
if __name__ == '__main__':
    obj = Object()
    print(obj.color)
    
    del obj
    print(obj.color)
