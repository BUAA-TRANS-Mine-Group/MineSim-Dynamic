import numpy as np


class QuarticPolynomial:
    """四次多项式 插值曲线."""

    def __init__(self, xs: float, vxs: float, axs: float, vxe: float, axe: float, time: float):
        """calc coefficient of quartic polynomial.
            初始化四次多项式插值曲线的系数,该值(如s l)相对于time的曲线关系为四次多项式.

        Args:
            xs (float): 起始点的x坐标.
            vxs (float): 起始点的速度.
            axs (float): 起始点的加速度.
            vxe (float): 终点的速度.
            axe (float): 终点的加速度.
            time (float): 从起始点到终点的时间.
        """
        #

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time**2, 4 * time**3], [6 * time, 12 * time**2]])  # 2x2 的矩阵
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t: float):
        """计算在时间 t 的点值

        Args:
            t (_type_): 初始点为0.0s;该值是time轴 = t,求解对应曲线的 值.

        Returns:
            float: 求解对应曲线的 值.
        """
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t: float):
        """计算在时间 t 的一阶导数（速度）"""
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t: float):
        """计算在时间 t 的二阶导数（加速度）"""
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t: float):
        """计算在时间 t 的三阶导数"""
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class QuinticPolynomial:
    """五次多项式 插值曲线."""

    def __init__(self, xs: float, vxs: float, axs: float, xe: float, vxe: float, axe: float, time: float):
        """calc coefficient of quintic polynomial.
            初始化五次多项式插值曲线的系数,该值(如s l)相对于time的曲线关系为四次多项式.

        Args:
            xs (float): 起始点的x坐标.start
            vxs (float): 起始点的速度.
            axs (float): 起始点的加速度.
            xe (float): 终点的x坐标.end
            vxe (float): 终点的速度.
            axe (float): 终点的加速度.
            time (float): 从起始点到终点的时间.
        """
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time**3, time**4, time**5], [3 * time**2, 4 * time**3, 5 * time**4], [6 * time, 12 * time**2, 20 * time**3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time**2, vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t: float):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t: float):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t: float):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t: float):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt
