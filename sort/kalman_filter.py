""" 最优估计理论

信息融合：
假设 x1 ~ N(μ1, σ1^2), x2 ~ N(μ2, σ2^2), 则融合后的分布为:
x ~ N(σ2^2/σ1^2+σ2^2 * μ1 + σ1^2/σ1^2+σ2^2 * μ2, σ1^2*σ2^2/σ1^2+σ2^2)
"""

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
            方差之和:    S(t) = H∙P_(t)∙H^T + R(t)
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
                 R: Union[tf.Tensor, np.ndarray] = None,
                 format: str = "left_matmul"):

        self.x = tf.cast(tf.identity(x), tf.float32)
        self.F = tf.cast(tf.identity(F), tf.float32)
        self.H = tf.cast(tf.identity(H), tf.float32)

        if backend.ndim(self.x) == 1:
            self.x = tf.expand_dims(x, 0)
        if format == "left_matmul":
            # self.F, self.H, self.B 是原公式的转置，代表右乘变换
            self.F = tf.transpose(self.F)
            self.H = tf.transpose(self.H)

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

        if backend.ndim(P) == 2:
            P = tf.expand_dims(P, 0)
        if backend.ndim(Q) == 2:
            Q = tf.expand_dims(Q, 0)
        if backend.ndim(R) == 2:
            R = tf.expand_dims(R, 0)

        self.P = tf.cast(tf.identity(P), tf.float32)
        self.Q = tf.cast(tf.identity(Q), tf.float32)
        self.R = tf.cast(tf.identity(R), tf.float32)

        assert self.Q.shape[-1] == self.state_dim
        assert self.R.shape[-1] == self.measure_dim

        self.x_ = None
        self.P_ = None

    def predict(self, u: tf.Tensor = None):
        self.x_ = tf.matmul(self.x, self.F)
        if u is not None:
            u = tf.cast(u, tf.float32)
            self.x_ += tf.matmul(u, self.B)
        # (P∙F^T)^T∙F^T = F∙P^T∙F^T = F∙P∙F^T
        self.P_ = tf.matmul(tf.matmul(self.P, self.F), self.F, transpose_a=True) + self.Q

    def update(self, z: tf.Tensor):
        z = tf.cast(z, self.x.dtype)
        y = z - tf.matmul(self.x_, self.H)
        # (P_∙H^T)^T∙H^T = H∙P^T∙H^T = H∙P∙H^T
        S = tf.matmul(tf.matmul(self.P_, self.H), self.H, transpose_a=True) + self.R
        K = backend.batch_dot(tf.matmul(self.P_, self.H), tf.linalg.inv(S))
        self.x = self.x_ + backend.batch_dot(K, y)
        I = tf.expand_dims(tf.eye(self.state_dim), 0)
        self.P = backend.batch_dot(I - tf.matmul(K, self.H, transpose_b=True), self.P_)

    def __call__(self, z: tf.Tensor, u: tf.Tensor = None):
        self.predict(u)
        self.update(z)
        return self.x


if __name__ == "__main__":
    kalman = KalmanFilter(x=[0.5, 0.5, 0.5],
                          F=[[1, 0, 0], [0, 1, 1], [0, 0, 1]],
                          H=[[0.2, 0.5, 1], [1, 0.9, 0]])

    z = tf.constant([[0.7, 0.3]])
    x = kalman(z)
    print(x)
