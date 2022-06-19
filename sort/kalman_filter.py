from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
from typing import Union


class KalmanFilter(object):
    """ 卡尔曼滤波器

    系统模型
        状态更新方程:   x(t) = F∙x(t-1) + B∙u(t) + w(t)
        观测方程:       z(t) = H∙x(t) + v(t)
        x为状态变量, z为观测变量, u为控制量, 如果没有则为0; 
        F为状态转移矩阵, H为观测矩阵, w为过程噪声, v为观测噪声
        w ~ N(0, Q), v ~ N(0, R), Q为过程噪声协方差矩阵, R为观测噪声协方差矩阵

    系统计算
        卡尔曼滤波器的操作包括两个阶段：预测与更新

        预测：滤波器使用上一状态的估计，做出对当前状态的估计
            状态先验估计:    x_(t) = F∙x(t-1) + B∙u(t)
            状态协方差预估:  P_(t) = F∙P(t-1)∙F^T + Q(t)
            x_为状态先验估计, P为状态协方差矩阵, P_为状态协方差先验估计

        更新: 利用当前时刻的观测修正先验估计得到后验估计
            观测余量:        y(t) = z(t) - H∙x_(t)
            观测的协方差:    S(t) = H∙P_(t)∙H^T + R(t)
            卡尔曼增益:      K(t) = P_(t)∙H^T∙S(t)^-1
                                = P_(t)∙H^T∙(H∙P_(t)∙H^T + R(t))^-1
            状态后验估计:    x(t) = x_(t) + K(t)∙y(t) 
                                = (I - K(t)∙H)∙x_(t) + K(t)∙z(t)
            状态协方差更新:  P(t) = (I - K(t)∙H)∙P_(t)
    """

    def __init__(self,
                 x: Union[tf.Tensor, np.ndarray],
                 F: Union[tf.Tensor, np.ndarray],
                 H: Union[tf.Tensor, np.ndarray],
                 B: Union[tf.Tensor, np.ndarray] = None,
                 P: Union[tf.Tensor, np.ndarray] = None,
                 Q: Union[tf.Tensor, np.ndarray] = None,
                 R: Union[tf.Tensor, np.ndarray] = None):

        # self.F, self.H, self.B 是原公式的转置，代表右乘变换
        self.x = tf.identity(x)
        self.F = tf.identity(F)
        self.H = tf.identity(H)

        assert self.x.shape[-1] == self.F.shape[-1]
        self.state_dim = self.x.shape[-1]
        self.measure_dim = self.H.shape[-1]

        if B is None:
            self.B = None
        else:
            self.B = tf.identity(B)

        if P is None:
            P = tf.eye(self.state_dim)
        if Q is None:
            Q = tf.eye(self.state_dim)
        if R is None:
            R = tf.eye(self.measure_dim)

        if tf.keras.backend.ndim(P) == 2:
            P = tf.expand_dims(P, 0)
        if tf.keras.backend.ndim(Q) == 2:
            Q = tf.expand_dims(Q, 0)
        if tf.keras.backend.ndim(R) == 2:
            R = tf.expand_dims(R, 0)

        self.P = tf.identity(P)
        self.Q = tf.identity(Q)
        self.R = tf.identity(R)

        assert self.Q.shape[-1] == self.state_dim
        assert self.R.shape[-1] == self.measure_dim

    def predict(self, u: tf.Tensor = None):
        x_ = tf.matmul(self.x, self.F)
        if u is not None:
            x_ += tf.matmul(u, self.B)
        # (P∙F^T)^T∙F^T = F∙P^T∙F^T = F∙P∙F^T
        P_ = tf.matmul(tf.matmul(self.P, self.F), self.F, transpose_a=True) + self.Q
        return x_, P_

    def update(self, z: tf.Tensor, x_: tf.Tensor, P_: tf.Tensor):
        y = z - tf.matmul(x_, self.H)
        # (P_∙H^T)^T∙H^T = H∙P^T∙H^T = H∙P∙H^T
        S = tf.matmul(tf.matmul(P_, self.H), self.H, transpose_a=True) + self.R
        K = backend.batch_dot(tf.matmul(P_, self.H), tf.linalg.inv(S))
        self.x = x_ + backend.batch_dot(K, y)
        I = tf.expand_dims(tf.eye(self.state_dim), 0)
        self.P = backend.batch_dot(I-tf.matmul(K, self.H, transpose_b=True), P_)

    def __call__(self, z: tf.Tensor, u: tf.Tensor = None):
        x_, P_ = self.predict(u)
        self.update(z, x_, P_)
