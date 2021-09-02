from matplotlib import cm
import numpy as np

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
