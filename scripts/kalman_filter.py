# -*- coding: UTF-8 -*-

import numpy as np
import math

class KalmanFilter2D:
    def __init__(self, time_interval, x, dx, sigma_ax=1, sigma_ox=1):
        # 功能：初始化KalmanFilter2D对象
        #      匀速直线运动模型，二维状态向量
        # 输入：time_interval <class 'float'> 时间间隔
        #      x <class 'float'> X坐标
        #      dx <class 'float'> X方向速度
        #      sigma_ax <class 'float'> 过程噪声标准差
        #      sigma_ox <class 'float'> 量测噪声标准差
        
        self.ti = time_interval
        
        # 过程噪声分布矩阵
        gg = np.array([[0.5 * self.ti ** 2],
                       [self.ti]])
        
        # 过程噪声、量测噪声
        self.noise_q = gg @ np.array([[sigma_ax ** 2]]) @ gg.T
        self.noise_r = np.array([[sigma_ox ** 2]])
        
        # 状态向量、状态协方差矩阵
        self.xx = np.array([[x],
                            [dx]])
        self.pp = np.array([[sigma_ox ** 2, 0],
                            [0, 0]])
        
    def predict(self):
        # 功能：计算离散卡尔曼滤波器预测方程
        
        # 状态转移矩阵
        ff = np.array([[1, self.ti],
                       [0, 1]])
        
        # 状态预测、状态协方差预测
        self.xx = ff @ self.xx
        self.pp = ff @ self.pp @ ff.T + self.noise_q
        
    def update(self, zx):
        # 功能：计算离散卡尔曼滤波器更新方程
        # 输入：zx <class 'float'> X坐标
        
        # 量测向量
        zs = np.array([[zx]])
        
        # 量测矩阵
        hh = np.array([[1, 0]])
        
        # 新息向量、新息协方差矩阵
        zz = zs - hh @ self.xx
        ss = hh @ self.pp @ hh.T + self.noise_r
        
        # 卡尔曼增益
        kk = self.pp @ hh.T @ np.linalg.inv(ss)
        
        # 状态更新、状态协方差更新
        self.xx = self.xx + kk @ zz
        self.pp = self.pp - kk @ hh @ self.pp

class KalmanFilter4D:
    def __init__(self, time_interval, x, dx, y, dy, sigma_ax=1, sigma_ay=1, sigma_ox=1, sigma_oy=1, gate_threshold=1000):
        # 功能：初始化KalmanFilter4D对象
        #      匀速直线运动模型，四维状态向量
        # 输入：time_interval <class 'float'> 时间间隔
        #      x <class 'float'> X坐标
        #      dx <class 'float'> X方向速度
        #      y <class 'float'> Y坐标
        #      dy <class 'float'> Y方向速度
        #      sigma_ax <class 'float'> 过程噪声标准差
        #      sigma_ay <class 'float'> 过程噪声标准差
        #      sigma_ox <class 'float'> 量测噪声标准差
        #      sigma_oy <class 'float'> 量测噪声标准差
        
        self.ti = time_interval
        
        # 过程噪声分布矩阵
        gg = np.array([[0.5 * self.ti ** 2, 0],
                       [self.ti, 0],
                       [0, 0.5 * self.ti ** 2],
                       [0, self.ti]])
        
        # 过程噪声、量测噪声
        self.noise_q = gg @ np.array([[sigma_ax ** 2, 0], [0, sigma_ay ** 2]]) @ gg.T
        self.noise_r = np.array([[sigma_ox ** 2, 0], [0, sigma_oy ** 2]])
        
        # 状态向量、状态协方差矩阵
        self.xx = np.array([[x],
                            [dx],
                            [y],
                            [dy]])
        self.pp = np.array([[sigma_ox ** 2, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, sigma_oy ** 2, 0],
                            [0, 0, 0, 0]])
        
        self.gate_threshold = gate_threshold
        
    def predict(self):
        # 功能：计算离散卡尔曼滤波器预测方程
        
        # 状态转移矩阵
        ff = np.array([[1, self.ti, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, self.ti],
                       [0, 0, 0, 1]])
        
        # 状态预测、状态协方差预测
        self.xx = ff @ self.xx
        self.pp = ff @ self.pp @ ff.T + self.noise_q
        
    def update(self, zx, zy):
        # 功能：计算离散卡尔曼滤波器更新方程
        # 输入：zx <class 'float'> X坐标
        #      zy <class 'float'> Y坐标
        
        # 量测向量
        zs = np.array([[zx],
                       [zy]])
        
        # 量测矩阵
        hh = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0]])
        
        # 新息向量、新息协方差矩阵
        zz = zs - hh @ self.xx
        ss = hh @ self.pp @ hh.T + self.noise_r
        
        # 卡尔曼增益
        kk = self.pp @ hh.T @ np.linalg.inv(ss)
        
        # 状态更新、状态协方差更新
        self.xx = self.xx + kk @ zz
        self.pp = self.pp - kk @ hh @ self.pp
        
    def compute_the_residual(self, zx, zy):
        # 功能：计算新息加权范数
        # 输入：zx <class 'float'> X坐标
        #      zy <class 'float'> Y坐标
        # 输出：dd <class 'float'> 新息加权范数
        
        # 量测向量
        zs = np.array([[zx],
                       [zy]])
        
        # 量测矩阵
        hh = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0]])
        
        # 新息向量、新息协方差矩阵
        zz = zs - hh @ self.xx
        ss = hh @ self.pp @ hh.T + self.noise_r
        
        # 新息加权范数
        dd = zz.T @ np.linalg.inv(ss) @ zz
        
        return dd
        
    def compute_association_gate(self, g):
        # 功能：计算跟踪门
        #      椭圆方程为x^2 / a^2 + y^2 / b^2 = 1
        # 输入：g <class 'float'> 跟踪门阈值
        # 输出：a <class 'float'> 跟踪门椭圆长轴长
        #      b <class 'float'> 跟踪门椭圆短轴长
        
        a = math.sqrt(g * (self.pp[0, 0] + self.noise_r[0, 0]))
        b = math.sqrt(g * (self.pp[2, 2] + self.noise_r[1, 1]))
        
        return a, b
