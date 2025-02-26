import math
import bisect
import numpy as np


class CubicSpline1D:
    """
    1D Cubic Spline class
    用于在一维空间中创建和处理三次样条曲线.三次样条曲线是数学中用于平滑地通过一系列数据点的一种曲线.
    该类实现了样条曲线的构建,并提供了计算给定 x 坐标处的 y 坐标,一阶导数和二阶导数的方法.
    https://zhuanlan.zhihu.com/p/598477502 ;
    https://zhuanlan.zhihu.com/p/604972619 ;三次样条曲线的 三,算法总结

    ## 三次样条曲线的组成:
    三次样条曲线由多个三次多项式段组成,每个段在定义的 x 倇值范围内有效.对于给定的数据点集,每个多项式的形式如下:
    公式1:  $$ S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3 $$
    其中,$S_i(x)$ 是第 i 段多项式,$a_i, b_i, c_i, d_i$ 是多项式系数,$x_i$ 是第 i 段多项式的起始 x 坐标.

    ## 初始化过程
    1. **验证输入:**
    - 确保输入的 x 坐标是按升序排列的.如果 x 坐标没有排序或有任何降序的情况,将抛出 `ValueError`.

    2. **计算系数:**
    - `h`:计算相邻 x 坐标之间的差异.这是通过对 x 坐标数组应用 `np.diff` 函数来实现的.
    - `a`:直接从 y 坐标数组获取.在三次样条曲线中,`a` 系数代表曲线的初始 y 值.
    - `c`:通过解线性方程组来计算.首先计算矩阵 A 和 B,然后使用 `np.linalg.solve` 解线性方程 A*c = B,得到 c 系数.
    - `b` 和 `d`:使用计算出的 `a` 和 `c` 系数,以及步长 `h`,按照公式计算出 b 和 d 系数.
    -------------------------------------------------------------------------
    公式2: \[ d = \frac{c_{i+1} - c_i}{3h_i} \]
    这个公式反映了 c 系数在相邻两个段之间的变化率,这个变化率影响了曲线的弯曲程度.其中:
    - \( c_{i+1} \) 和 \( c_i \) 是相邻两个段的 `c` 系数.
    - \( h_i \) 是第 i 段的宽度,即相邻 x 坐标之间的差值.

    -------------------------------------------------------------------------
    公式3: \[ b = \frac{1}{h_i} \left( a_{i+1} - a_i \right) - \frac{h_i}{3} \left( 2c_i + c_{i+1} \right) \]
    这个公式基于相邻两个数据点的 a 系数和 c 系数的差值,用于计算每个段的线性部分的斜率.其中:
    - \( a_{i+1} \) 和 \( a_i \) 是相邻两个段的 `a` 系数.
    - \( c_{i+1} \) 和 \( c_i \) 是相邻两个段的 `c` 系数.
    - \( h_i \) 是第 i 段的宽度.



    Parameters
    ----------
    x_list : list
        x coordinates for data points. This x coordinates must be sorted in ascending order.
        # X坐标必须按升序排序
        举例:[0, 15.482986863512455, 69.99999915294421, 72.13499753397986, 73.55662101510983, 74.97769778590721,
        76.39825403259374, 77.81891283645905, 79.23993906879963, 81.3703828738435, 83.50249530629749, 86.34730052796908,
        100.85900976106169, 117.33743712752857, ...]
    y_list : list
        y coordinates for data points
    """

    def __init__(self, x_list: np.array, y_list: np.array):
        """初始化 三次样条曲线 各参数"""

        h = np.diff(x_list)  # 计算相邻 x 坐标之间的差异.这是通过对 x 坐标数组应用 np.diff 函数来实现的.
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x_list = x_list
        self.y_list = y_list
        self.nx_list = len(x_list)  # dimension of x

        # calc coefficient a 直接从 y 坐标数组获取.在三次样条曲线中,`a` 系数代表曲线的初始 y 值.
        self.a = [iy for iy in y_list]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx_list - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)
        pass

    def calc_position(self, x: float):
        """
        Calc `y` position for given `x`.

        if `x` is outside the data point's `x_list` range, return None.

        Returns
        -------
        y : float
            y position for given x.
        """
        if x < self.x_list[0]:
            return None
        elif x > self.x_list[-1]:
            return None

        i = self.__search_index(x)
        if i >= self.nx_list - 1:  # 防止索引超出范围
            i = self.nx_list - 2
        # print('index:', i, 'abcd_lengths', len(self.a), len(self.b), len(self.c), len(self.d))
        dx = x - self.x_list[i]
        position = self.a[i] + self.b[i] * dx + self.c[i] * dx**2.0 + self.d[i] * dx**3.0

        return position

    def calc_first_derivative(self, x: float):
        """
        Calc first derivative at given x. 计算给定 x 坐标处的一阶导数（即斜率）

        if x is outside the input x_list, return None

        Returns
        -------
        dy : float
            first derivative for given x.
        """
        if x < self.x_list[0]:
            return None
        elif x > self.x_list[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x_list[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2.0

        return dy

    def calc_second_derivative(self, x: float):
        """
        Calc second derivative at given x. 计算给定 x 坐标处的二阶导数

        if x is outside the input x_list, return None

        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        if x < self.x_list[0]:
            return None
        elif x > self.x_list[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x_list[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def __search_index(self, x: float):
        """search data segment index
        在 x_list 坐标数组中搜索给定 x 值的索引,用于确定应用哪一段曲线的系数.
        """
        return bisect.bisect(self.x_list, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        用于计算三次样条曲线的 c 系数的线性方程组的系数矩阵 A 和 B.
        """
        A = np.zeros((self.nx_list, self.nx_list))
        A[0, 0] = 1.0
        for i in range(self.nx_list - 1):
            if i != (self.nx_list - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx_list - 1, self.nx_list - 2] = 0.0
        A[self.nx_list - 1, self.nx_list - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        """
        calc matrix B for spline coefficient c
        用于计算三次样条曲线的 c 系数的线性方程组的系数矩阵 A 和 B.
        """
        B = np.zeros(self.nx_list)
        for i in range(self.nx_list - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B


class CubicSpline2D:
    """
    Cubic CubicSpline2D class  用于表示和处理二维空间中的三次样条曲线.
    样条曲线是一种数学工具,用于创建通过一系列数据点的平滑曲线.
    这个类提供了计算样条曲线上特定点的位置,曲率和偏航角（yaw）的方法.

    Parameters
    ----------
    x_list : list
        x coordinates for data points.
    y_list : list
        y coordinates for data points.
    """

    def __init__(self, x_list: np.array, y_list: np.array):
        self.s_list = self.__calc_s(x_list, y_list)
        self.sx_list = CubicSpline1D(self.s_list, x_list)
        self.sy_list = CubicSpline1D(self.s_list, y_list)

    def __calc_s(self, x_list: np.array, y_list: np.array):
        """计算从曲线起点到每个数据点的累积距离.这是通过计算每对相邻点之间的欧几里得距离（使用 np.hypot）并累加这些距离来完成的."""
        dx_list = np.diff(x_list)
        dy_list = np.diff(y_list)
        self.ds_list = np.hypot(dx_list, dy_list)  # 计算每对相邻点之间的欧几里得距离
        s = [0]
        s.extend(np.cumsum(self.ds_list))
        return s

    def calc_position(self, s: float):
        """calc position
        沿曲线的距离 s 计算曲线上的 (x, y) 坐标.这是通过对两个独立的一维样条曲线（self.sx 和 self.sy）求值来实现的.

        Args:
            s (float): distance from the start point. if `s` is outside the data point's range, return None.

        Returns:
           (float,float) : (x ,y) position for given s.
                            超过 sx_list sx_list,返回(None,None).
        """
        x = self.sx_list.calc_position(s)
        y = self.sy_list.calc_position(s)

        return x, y

    def calc_curvature(self, s: float):
        """
        calc curvature
        计算曲线在给定 s 处的曲率 k.曲率是曲线在特定点处弯曲程度的度量.这里通过计算 x 和 y 坐标相对于 s 的一阶和二阶导数,并应用曲率公式来计算.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        dx = self.sx_list.calc_first_derivative(s)
        ddx = self.sx_list.calc_second_derivative(s)
        dy = self.sy_list.calc_first_derivative(s)
        ddy = self.sy_list.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
        return k

    def calc_yaw(self, s: float):
        """
        calc yaw
        计算曲线在给定 s 处的偏航角 yaw.偏航角是曲线在该点的切线方向.
        这通过计算 x 和 y 坐标相对于 s 的一阶导数（即切线的斜率）并使用 atan2 函数来确定yaw角度.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        dx = self.sx_list.calc_first_derivative(s)
        dy = self.sy_list.calc_first_derivative(s)
        # 返回从 x 轴到点 (x, y) 的角度,角度范围[-PI,PI),负值表示顺时针方向的角度,正值表示逆时针方向的角度.
        yaw = math.atan2(dy, dx)
        return yaw


def test_CubicSpline2D(centerline_pts: np.ndarray):
    #  centerline_pts (np.ndarray): 全局规划的参考路径 中心线, 稀疏的点序列[x, y, yaw , width]*i.
    # 三次样条
    cubic_spline = CubicSpline2D(centerline_pts[:, 0], centerline_pts[:, 1])

    import matplotlib.pyplot as plt

    # 生成样条曲线上的点
    num_points = 100  # 生成样条曲线上的点数
    s_values = np.linspace(0, cubic_spline.s_list[-1], num_points)
    x_values = []
    y_values = []
    for s in s_values:
        x, y = cubic_spline.calc_position(s)
        x_values.append(x)
        y_values.append(y)

    # 绘图
    plt.figure()
    plt.plot(centerline_pts[:, 0], centerline_pts[:, 1], "o", label="Original points")
    plt.plot(x_values, y_values, label="Cubic Spline")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Cubic Spline 2D")
    plt.savefig("cubic_spline_2d.png")

    # todo 绘图 self.sx = CubicSpline1D(self.s, x) 和 self.sy = CubicSpline1D(self.s, y)的点及样条曲线, png保存到本目录下
    # 绘制 sx 和 sy 的样条曲线
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(cubic_spline.s_list, centerline_pts[:, 0], "o", label="Original X points")
    plt.plot(s_values, x_values, label="Cubic Spline for X")
    plt.legend()
    plt.xlabel("S")
    plt.ylabel("X")
    plt.title("Cubic Spline 1D for X")

    plt.subplot(2, 1, 2)
    plt.plot(cubic_spline.s_list, centerline_pts[:, 1], "o", label="Original Y points")
    plt.plot(s_values, y_values, label="Cubic Spline for Y")
    plt.legend()
    plt.xlabel("S")
    plt.ylabel("Y")
    plt.title("Cubic Spline 1D for Y")

    plt.tight_layout()
    plt.savefig("cubic_spline_1d.png")
    pass


if __name__ == "__main__":
    # test
    centerline_pts = np.array(
        [
            [289.53814, -669.668355, -2.99432306, 3.49991322],
            [274.22275, -671.940295, -2.9798421, 3.49999574],
            [220.417355, -680.72005, -2.9028075, 3.49758871],
            [218.342935, -681.225025, -2.75489475, 3.49999531],
            [217.026285, -681.761165, -2.60090886, 3.50000156],
            [215.807915, -682.492625, -2.42439172, 3.49999897],
            [214.737315, -683.426325, -2.2312777, 3.4999988],
            [213.865745, -684.548215, -2.05633388, 3.50000303],
            [213.202575, -685.805005, -1.85465925, 3.49999721],
            [212.60591, -687.85019, -1.6739451, 3.50000281],
            [212.386375, -689.97097, -1.52895173, 3.49999487],
            [212.50538, -692.813285, -1.46093018, 3.49657444],
            [214.09652, -707.2375, -1.47329921, 3.50000182],
            [215.700575, -723.63767, -1.49953495, 3.49999281],
            [216.768845, -738.603155, -1.52210173, 3.49999902],
            [218.15795, -767.107485, -1.51020939, 3.49804829],
            [218.66056, -775.393015, -1.51004494, 3.49976263],
            [219.166585, -783.712205, -1.51004494, 3.50017813],
        ]
    )
    test_CubicSpline2D(centerline_pts)

    pass
