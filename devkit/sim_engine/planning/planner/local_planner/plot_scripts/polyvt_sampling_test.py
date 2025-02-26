import copy

import os, sys
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from tqdm import tqdm


# 添加必要的路径到sys.path
def add_path_to_sys(target: str):
    abs_path = os.path.abspath(target)
    if abs_path not in sys.path:
        sys.path.append(abs_path)


dir_current_file = os.path.dirname(__file__)
dir_parent_1 = os.path.dirname(dir_current_file)
dir_parent_2 = os.path.dirname(dir_parent_1)
add_path_to_sys(dir_current_file)
add_path_to_sys(dir_parent_1)
add_path_to_sys(dir_parent_2)

from devkit.common.coordinate_system.frenet import GoalSampledFrenetTrajectory
from devkit.common.coordinate_system.frenet import JerkSpaceSamplingFrenetTrajectory
from common.geometry.polynomial import QuarticPolynomial, QuinticPolynomial


def multiline(list_t, list_state, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    """
    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments_t_state = [np.column_stack([t[: len(state)], state]) for t, state in zip(list_t, list_state)]
    lc = LineCollection(segments_t_state, **kwargs)

    # set coloring of line segments
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    ax.add_collection(lc)
    ax.autoscale()

    return lc


def multiline_add_best(list_t, list_state, c, ax=None, best_traj_id=None, **kwargs):
    """
    Plot lines with different colorings, highlighting the best trajectory in bold green.

    Parameters
    ----------
    list_t : iterable container of x coordinates
    list_state : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax : optional, Axes to plot on.
    best_traj_id : int, the index of the best trajectory to highlight
    kwargs : optional, passed to LineCollection

    Notes
    -----
    len(xs) == len(ys) == len(c) is the number of line segments
    len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)
    """
    ax = plt.gca() if ax is None else ax

    # Create LineCollection for all trajectories
    # for i, (t, state) in enumerate(zip(list_t, list_state)):
    #     lc = LineCollection([np.column_stack([t, state])], array=[c[i]], **kwargs)
    #     ax.add_collection(lc)
    segments_t_state = [np.column_stack([t[: len(state)], state]) for t, state in zip(list_t, list_state)]
    lc = LineCollection(segments_t_state, **kwargs)

    # Set coloring of line segments
    lc.set_array(np.asarray(c))
    # Add lines to axes and rescale
    ax.add_collection(lc)
    ax.autoscale()

    # Highlight the best trajectory if specified
    if best_traj_id is not None and 0 <= best_traj_id < len(list_t):
        best_traj_segments = np.column_stack([list_t[best_traj_id][: len(list_state[best_traj_id])], list_state[best_traj_id]])
        lc_best = LineCollection([best_traj_segments], colors="black", linewidths=2, zorder=5)
        ax.add_collection(lc_best)
    # Highlight the best trajectory
    # best_t, best_state = list_t[best_traj_id], list_state[best_traj_id]
    # ax.plot(best_t, best_state, color="red", linewidth=2, label="Best Trajectory")
    # ax.legend()

    return lc


def plot_jerk_sampling(trajectories, dir_figure: str = "polyvt-sampling-.png"):
    """todo 绘制ST 图的采样可视化"""
    # plt.ion()

    fig = plt.figure(figsize=[30, 15])
    # plt.rcParams["font.size"] = 10
    # len_step = len(search_tree)
    # 创建一个具有两个子图的图表,左侧是主图,右侧是 colorbar
    # self.fig = plt.figure(figsize=(figsize_x + 2, figsize_y))  # figsize=(21, 10) 总宽度为21,包括20的主图和1的colorbar
    gs = GridSpec(2, 3, width_ratios=[20, 20, 1], figure=fig)  # 设置 gridspec_kw 参数中的 width_ratios 来控制两个子图的宽度比例.
    ax_st = fig.add_subplot(gs[0, 0])  # Top left
    ax_vt = fig.add_subplot(gs[0, 1])  # Top middle
    ax_at = fig.add_subplot(gs[1, 0])  # Bottom left
    ax_jt = fig.add_subplot(gs[1, 1])  # Bottom middle
    ax_cost_colorbar = fig.add_subplot(gs[:, 2])  # This will create a subplot that spans the 3rd column
    ax_cost_colorbar.set_title("traj cost", pad=10, fontsize=20, loc="center")
    ax_vt.set_ylabel("v [m/s]")
    ax_vt.set_title("v-t")

    ax_at.set_ylabel("a [m/s²]")
    ax_at.set_title("a-t")

    ax_jt.set_ylabel("jerk [m/s³]")
    ax_jt.set_title("jerk-t")

    list_t = []
    list_s = []
    list_s_d = []
    list_s_dd = []
    list_s_ddd = []
    costs = []
    cost = 0.0
    for ft in trajectories:  # 第 i 时间步的树搜索速度规划结果
        cost += 0.1
        list_t.append(ft.t)
        costs.append(ft.cost_total + cost)
        # list_s.append(ft.s[0:])
        list_s.append(ft.s)
        list_s_d.append(ft.s_d)
        list_s_dd.append(ft.s_dd)
        list_s_ddd.append(ft.s_ddd)

    # 绘制 s-t 图 (速度-时间)
    lc = multiline(list_t=list_t, list_state=list_s, c=costs, ax=ax_st, cmap="winter", lw=0.1)  # 多个规划结果绘制的函数
    colorbar = fig.colorbar(lc, cax=ax_cost_colorbar)
    ax_st.set_ylabel("s [m]")
    ax_st.set_title("s-t")

    # 绘制 v-t 图 (速度-时间)
    multiline(list_t=list_t, list_state=list_s_d, c=costs, ax=ax_vt, cmap="winter", lw=0.1)
    ax_vt.set_ylabel("v [m/s]")
    ax_vt.set_title("v-t")

    # 绘制 a-t 图 (加速度-时间)
    multiline(list_t=list_t, list_state=list_s_dd, c=costs, ax=ax_at, cmap="winter", lw=0.1)
    ax_at.set_ylabel("a [m/s^2]")
    ax_at.set_title("a-t")

    # 绘制 jerk-t 图 (加加速度-时间)
    multiline(list_t=list_t, list_state=list_s_ddd, c=costs, ax=ax_jt, cmap="winter", lw=0.1)
    ax_jt.set_ylabel("jerk [m/s^3]")
    ax_jt.set_title("jerk-t")

    plt.show()
    # pass
    # plt.savefig("polyvt-sampling-.png")
    plt.savefig(dir_figure)
    # pass
    plt.close()


def _set_axis_style(ax, labelsize=20):
    """设置给定轴的样式。"""
    ax.tick_params(axis="both", which="major", labelsize=labelsize)  # 设置刻度标签大小
    # self.axbg.tick_params(axis="x", which="major", labelsize=20)
    # self.axbg.tick_params(axis="y", which="major", labelsize=20)


def plot_jerk_sampling_of_best_traj(
    trajectories: list,
    best_traj_id: int,
    dir_figure: str = "polyvt-sampling-.png",
    dir_svg: str = "polyvt-sampling-.svg",
):
    """确保绘图中每条轨迹的cost_total较小的显示在上层，并且最佳轨迹（best_traj_id指定的轨迹）突出显示"""
    plt.ioff()  # 关闭交互模式
    fig = plt.figure(figsize=[30, 15])
    fig.tight_layout()
    gs = GridSpec(2, 3, width_ratios=[20, 20, 1], figure=fig)
    # fig.subplots_adjust(wspace=0)  # 设置子图之间的宽度空间为0
    ax_st = fig.add_subplot(gs[0, 0])
    ax_vt = fig.add_subplot(gs[0, 1])
    ax_at = fig.add_subplot(gs[1, 0])
    ax_jt = fig.add_subplot(gs[1, 1])
    ax_cost_colorbar = fig.add_subplot(gs[:, 2])

    _set_axis_style(ax=ax_st, labelsize=20)  # 设置坐标轴刻度字体大小
    _set_axis_style(ax=ax_vt, labelsize=20)  # 设置坐标轴刻度字体大小
    _set_axis_style(ax=ax_at, labelsize=20)  # 设置坐标轴刻度字体大小
    _set_axis_style(ax=ax_jt, labelsize=20)  # 设置坐标轴刻度字体大小
    _set_axis_style(ax=ax_cost_colorbar, labelsize=20)  # 设置坐标轴刻度字体大小

    ax_cost_colorbar.set_title("speed traj cost", pad=10, fontsize=20, loc="center")
    ax_st.set_ylabel("s [m]", fontsize=20)
    ax_st.set_xlabel("t [s]", fontsize=20)
    # ax_st.set_title("s-t")
    ax_vt.set_ylabel("v [m/s]", fontsize=20)
    ax_vt.set_xlabel("t [s]", fontsize=20)
    # ax_vt.set_title("v-t")
    ax_at.set_ylabel("a [m/s²]", fontsize=20)
    ax_at.set_xlabel("t [s]", fontsize=20)
    # ax_at.set_title("a-t")
    ax_jt.set_ylabel("jerk [m/s³]", fontsize=20)
    ax_jt.set_xlabel("t [s]", fontsize=20)
    # ax_jt.set_title("jerk-t")

    # 先提取最佳轨迹，然后从列表中移除它
    best_trajectory = trajectories.pop(best_traj_id)

    # 按成本对剩余轨迹进行降序排序，成本低的轨迹将被后绘制，因此会显示在上层
    trajectories = sorted(trajectories, key=lambda x: x.cost_total, reverse=True)

    # 将最佳轨迹添加回列表的末尾，确保它最后被绘制
    trajectories.append(best_trajectory)

    list_t = []
    list_s = []
    list_s_d = []
    list_s_dd = []
    list_s_ddd = []
    costs = []
    for ft in trajectories:
        list_t.append(ft.t)
        costs.append(ft.cost_total)
        list_s.append(ft.s)
        list_s_d.append(ft.s_d)
        list_s_dd.append(ft.s_dd)
        list_s_ddd.append(ft.s_ddd)

    # 绘制各图表 list_t=list_t, list_state=list_s_d, c=costs, ax=ax_vt
    # 22viridis  22rainbow  22winter
    lc = multiline_add_best(list_t=list_t, list_state=list_s, c=costs, ax=ax_st, cmap="winter", lw=0.2, best_traj_id=len(trajectories) - 1)
    colorbar = fig.colorbar(lc, cax=ax_cost_colorbar)
    multiline_add_best(list_t=list_t, list_state=list_s_d, c=costs, ax=ax_vt, cmap="winter", lw=0.2, best_traj_id=len(trajectories) - 1)
    multiline_add_best(list_t=list_t, list_state=list_s_dd, c=costs, ax=ax_at, cmap="winter", lw=0.2, best_traj_id=len(trajectories) - 1)
    multiline_add_best(list_t=list_t, list_state=list_s_ddd, c=costs, ax=ax_jt, cmap="winter", lw=0.2, best_traj_id=len(trajectories) - 1)

    num_trajs = len(trajectories)
    # text_title = f"One path sampled {num_trajs} speed curves"
    # fig.suptitle(text_title, fontsize=20)

    plt.savefig(dir_figure)
    plt.savefig(dir_svg)
    plt.close()
    print("##log##绘图完成.")


class PolyVTSampling(object):
    """基于线型设计的采样算法"""

    def __init__(self, total_time, delta_t, jerk_min, jerk_max, a_min, a_max, v_max, num_samples) -> None:
        """参数主要是自车的动力学最大性能限制."""
        self.total_time = total_time
        self.delta_t = delta_t
        self.jerk_min = jerk_min
        self.jerk_max = jerk_max
        self.a_min = a_min
        self.a_max = a_max
        self.v_max = v_max
        self.num_samples = num_samples
        self.time_step_max = int(round(self.total_time / self.delta_t))
        self.time_array = np.linspace(0.0, total_time, int(total_time / delta_t) + 1).tolist()
        # 为每个时间点生成jerk值
        self.jerk_values = np.linspace(jerk_min, jerk_max, num_samples)
        self.t1_array_delta = 0.2000
        self.t2_array_delta = 0.4000
        self.t3_array_delta = 0.8000
        self.t4_array_delta = 1.0000
        self.t1_array = np.arange(0.0, self.total_time + 1e-9, self.t1_array_delta)
        self.t2_array = np.concatenate((np.array([5.0]), np.arange(0.0, self.total_time + 1e-9, self.t2_array_delta)))
        # self.t3_array = np.concatenate((np.array([5.0]), np.arange(0.0, self.total_time + 1e-9, 0.5)))
        # self.t4_array = np.concatenate((np.array([0.1, 0.2, 0.3]), np.arange(0.0, self.total_time + 1e-9, 1.0)))
        # self.t2_array = np.arange(0.0, self.total_time + 1e-9, 0.4)
        self.t3_array = np.arange(0.0, self.total_time + 1e-9, self.t3_array_delta)
        self.t4_array = np.arange(0.0, self.total_time + 1e-9, self.t4_array_delta)

        self.jerk_min_3 = jerk_min * 0.8
        self.jerk_max_3 = jerk_max * 0.8
        self.jerk_min_5 = jerk_min * 0.4
        self.jerk_max_5 = jerk_max * 0.4
        a = 1

    @staticmethod
    def calculate_st_by_jt(s0, v0, a0, j, t):
        """计算jt采样结果对应时刻的位移s"""
        return s0 + v0 * t + 0.5 * a0 * t * t + (1 / 6.0) * j * t * t * t

    @staticmethod
    def calculate_st_by_jt_clamp(s0, v0, a0, j, t):
        s = s0 + v0 * t + 0.5 * a0 * t * t + (1 / 6.0) * j * t * t * t
        if s < s0:
            return s0
        else:
            return s

    @staticmethod
    def calculate_vt_by_jt(v0, a0, j, t):
        """计算jt采样结果对应时刻的速度v"""
        return v0 + a0 * t + 0.5 * j * t * t

    @staticmethod
    def calculate_vt_by_jt_clamp(v0, a0, j, t):
        v = v0 + a0 * t + 0.5 * j * t * t
        if v < 0.0:
            return 0.0
        else:
            return v

    @staticmethod
    def calculate_at_by_jt(a0, j, t):
        """计算jt采样结果对应时刻的加速度a"""
        return a0 + j * t

    def check_state(self, st, vt, at, s0, v0, a0):
        """检查终端状态是否超过动力学约束

        Args:
        s0, v0, a0:初始状态值;
        st, vt, at:t时间后的状态值;


        Returns:
            bool: _description_
        """

        # check acceleration
        ACC_MAX = self.a_max - 1e-9
        if at > ACC_MAX or at < -ACC_MAX:
            return False

        # check velocity
        VEL_MAX = self.v_max - 1e-9
        # if vt > VEL_MAX or vt < 0.0 + 1e-9:
        if vt > VEL_MAX or vt < 0.0 - 1e-9:  # 加减影响不大；加 num =3737;减 num =3720
            return False

        if st < s0 - 1e-9:
            # if st < s0 + 1e-9: #加减影响非常大，减少0.8的轨迹；
            return False

        return True

    def check_state_stop(self, s2, v2, a2, s1):

        if a2 > 0.0 or a2 < -1.0:
            return False

        # clamp 一个范围[-0.5,0.0]
        if v2 > 0.0 or v2 < -0.5:  # 加减影响不大；加 num =3737;减 num =3720
            return False

        if s2 < s1 - 0.1:
            return False

        return True

    def smpling_quintic_polynomial(self, s0, v0, a0, num_sampling):
        frenet_trajectories = []
        traj_per_timestep = []
        ft = GoalSamplingFrenetTrajectory()

        lat_qp = QuinticPolynomial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.total_time)
        ft.t = [t for t in np.arange(0.0, self.total_time, self.delta_t)]
        ft.d = [lat_qp.calc_point(t) for t in ft.t]
        ft.d_d = [lat_qp.calc_first_derivative(t) for t in ft.t]
        ft.d_dd = [lat_qp.calc_second_derivative(t) for t in ft.t]
        ft.d_ddd = [lat_qp.calc_third_derivative(t) for t in ft.t]

        #! longitudinal sampling
        for index, tv in enumerate(np.linspace(0.0, self.v_max, num_sampling)):
            tft = copy.deepcopy(ft)  #
            lon_qp = QuarticPolynomial(s0, v0, a0, tv, 0.0, self.total_time)
            tft.s = [lon_qp.calc_point(t) for t in ft.t]
            tft.s_d = [lon_qp.calc_first_derivative(t) for t in ft.t]
            tft.s_dd = [lon_qp.calc_second_derivative(t) for t in ft.t]
            tft.s_ddd = [lon_qp.calc_third_derivative(t) for t in ft.t]

            # Compute the final cost
            tft.cost_total = index * 0.001
            frenet_trajectories.append(tft)
            traj_per_timestep.append(tft)

        return frenet_trajectories

    def calc_global_paths(self, ftlist: list[GoalSamplingFrenetTrajectory]) -> list:
        """将一系列 GoalSamplingFrenetTrajectory 对象（代表在 Frenet 坐标系中的路径）转换成全局坐标系中的路径.
        具体来说,它会计算每个 GoalSamplingFrenetTrajectory 上的全局 x, y 坐标,偏航角,曲率等参数.

        Args:
            ftlist (list[GoalSamplingFrenetTrajectory]): GoalSamplingFrenetTrajectory 的列表.其实每次只传入一个值

        Returns:
            list:ftlist, 返回本身的地址, 将一个列表作为参数传递给函数时,您实际上是传递了这个列表的引用（或地址）
        """
        passed_ftlist = []
        for ft in ftlist:
            # Calculate global positions for each point in the  Frenet trajectory ft
            for i in range(len(ft.s)):
                ix, iy = self.cubic_spline.calc_position(ft.s[i])  # 沿参考曲线的距离 s 计算曲线上的 (x, y) 坐标
                if ix is None:  # !超过了参考路径终点s距离
                    break
                # Calculate global yaw (orientation) from Frenet s coordinate
                i_yaw = self.cubic_spline.calc_yaw(ft.s[i])  # 沿参考曲线的距离 s 计算曲线上的 yaw
                di = ft.d[i]
                # Calculate the final x, y coordinates considering lateral offset (d)
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                ft.x.append(fx)
                ft.y.append(fy)

            if len(ft.x) >= 2:
                #  Convert lists to numpy arrays for easier mathematical operations
                # calc yaw and ds
                ft.x = np.array(ft.x)
                ft.y = np.array(ft.y)
                # # Calculate differences between consecutive x and y coordinates
                x_d = np.diff(ft.x)
                y_d = np.diff(ft.y)
                # Calculate yaw angles and their differences
                # ! 涉及角度的计算都考虑了角度的周期性:使用 np.arctan2(y_d, x_d) 函数计算偏航角时,结果的范围[-PI,PI),使用弧度作为单位.
                ft.yaw = np.arctan2(y_d, x_d)
                ft.ds = np.hypot(x_d, y_d)
                ft.yaw = np.append(ft.yaw, ft.yaw[-1])
                # Calculate curvature (c), its first derivative (c_d), and second derivative (c_dd)
                dt = self.settings.delta_t
                # ! 计算曲率
                # 通过偏航角的差分除以路径长度的差分计算曲率.
                # 这里使用 np.diff 函数来计算偏航角 yaw 的差分,即每段路径的偏航角变化量,然后除以相应的路径长度变化量 ft.ds.
                ft.c = np.divide(np.diff(ft.yaw), ft.ds)  # ! 确保 yaw 为-pi 到 pi.
                ft.c_d = np.divide(np.diff(ft.c), dt)
                ft.c_dd = np.divide(np.diff(ft.c_d), dt)

                # Append the trajectory with calculated global parameters to the list
                passed_ftlist.append(ft)
        return ftlist

    # class QuinticPolynomialSampler():
    # def

    def sampling(self, s0, v0, a0) -> list[tuple]:
        """基于a-t空间线形设计的j-t空间采样策略.
        短规划时域(如5秒):短规划时域内,好的纵向规划是不进行频繁的加速、减速模式切换的;
        设计5段at直线拼接,对应着不同的驾驶模式:
        1)a曲线上升;2)a曲线恒定不变;3)a曲线下降\反方向上升; 4)a曲线再次恒定不变;5)a曲线反方向下降;
        对应的每段a-t直线的jerk值及维持时间:j1, t1, j2, t2, j3, t3, j4, t4, j5, t5

        Args:
            s0, v0, a0: Initial value.

        Returns:
            list[tuple]: 由(j1, t1, j2, t2, j3, t3, j4, t4, j5, t5)组成的不同at线形直线拼接采样结果种子.
        """
        seeds = []
        TINTER_MIN = 0.55
        for j1 in tqdm(np.linspace(self.jerk_min, self.jerk_max, self.num_samples)):
            # for t1 in np.linspace(0, self.total_time, self.time_step_max + 1):
            for t1 in self.t1_array:
                # j1固定数量采样,t1固定数量采样
                if t1 < TINTER_MIN:
                    continue

                s1 = self.calculate_st_by_jt(s0, v0, a0, j1, t1)
                v1 = self.calculate_vt_by_jt(v0, a0, j1, t1)
                a1 = self.calculate_at_by_jt(a0, j1, t1)

                if not self.check_state(s1, v1, a1, s0, v0, a0):
                    continue

                j2 = 0.0
                # for t2 in np.linspace(0, self.total_time - t1, int((self.total_time - t1) / (self.delta_t * 2))):
                # len_2 =len(np.linspace(0, self.total_time - t1, int((self.total_time - t1) / (self.delta_t * 2))))#不要大于22
                for t2 in self.t2_array:

                    # j2=0,t2采样数量是t1计算方式/2
                    if t1 + t2 >= self.total_time - self.t2_array_delta * 3:
                        continue

                    s2 = self.calculate_st_by_jt(s1, v1, a1, j2, t2)
                    v2 = self.calculate_vt_by_jt(v1, a1, j2, t2)
                    a2 = self.calculate_at_by_jt(a1, j2, t2)

                    if not self.check_state(s2, v2, a2, s1, v1, a1):
                        continue

                    if t1 + t2 > self.total_time + 1e-9:
                        continue

                    for j3 in np.linspace(self.jerk_min_3, self.jerk_max_3, int(self.num_samples / 4)):
                        if j3 * j1 > 0:  # 线型设计，不允许只有同向加速度变化
                            continue

                        # for t3 in np.linspace(0, self.total_time - t1 - t2, int((self.total_time - t1 - t2) / (self.delta_t * 4))):
                        # len_3 =len(np.linspace(0, self.total_time - t1 - t2, int((self.total_time - t1 - t2) / (self.delta_t * 4))))#不要大于11
                        for t3 in self.t3_array:
                            # j3=j1采样数量*0.25,t3采样数量是t1计算方/4
                            if t1 + t2 + t3 > self.total_time - self.t3_array_delta * 2:
                                continue

                            s3 = self.calculate_st_by_jt(s2, v2, a2, j3, t3)
                            v3 = self.calculate_vt_by_jt(v2, a2, j3, t3)
                            a3 = self.calculate_at_by_jt(a2, j3, t3)

                            if not self.check_state(s3, v3, a3, s2, v2, a2):
                                continue

                            if a3 * a1 > 0:  # 确保第三段模式已经变为相反的纵向运动模型
                                continue

                            if a3 * a2 < 0:  #! 检查一下[a3 ,0.0 ,a2]的极点情况
                                t2_5 = (0.0 - a2) / j3
                                v2_5 = self.calculate_vt_by_jt(v2, a2, j3, t2_5)
                                if v2_5 > self.v_max or v2_5 < 0.0 + 1e-9:
                                    continue

                            j4 = 0.0
                            # for t4 in np.linspace(0, self.total_time - t1 - t2 - t3, int((self.total_time - t1 - t2 - t3) / (self.delta_t * 8))):
                            # len_4=len(np.linspace(0, self.total_time - t1 - t2, int((self.total_time - t1 - t2) / (self.delta_t * 8))))#不要大于5
                            for t4 in self.t4_array:
                                # aaaa = np.linspace(0, self.total_time - t1 - t2 - t3, int((self.total_time - t1 - t2 - t3) / (self.delta_t * 9)))
                                # j4=0,t4采样数量是t1计算方式/8
                                if t1 + t2 + t3 + t4 > self.total_time - self.t4_array_delta:
                                    continue

                                s4 = self.calculate_st_by_jt(s3, v3, a3, j4, t4)
                                v4 = self.calculate_vt_by_jt(v3, a3, j4, t4)
                                a4 = self.calculate_at_by_jt(a3, j4, t4)

                                if not self.check_state(s4, v4, a4, s3, v3, a3):
                                    continue

                                # j5=保证加速度到0
                                t5 = self.total_time - t1 - t2 - t3 - t4
                                j5 = (0.0 - a4) / t5
                                if j5 > self.jerk_max_5 or j5 < self.jerk_min_5:
                                    continue

                                s5 = self.calculate_st_by_jt(s4, v4, a4, j5, t5)
                                v5 = self.calculate_vt_by_jt(v4, a4, j5, t5)
                                a5 = self.calculate_at_by_jt(a4, j5, t5)
                                if not self.check_state(s5, v5, a5, s4, v4, a4):
                                    continue

                                seeds.append((j1, t1, j2, t2, j3, t3, j4, t4, j5, t5))
        return seeds

    def sampling_stop_seeds_supplement(self, s0, v0, a0):
        """Supplements the existing seeds with emergency stop trajectories.
        补充急停的采样轨迹,一直减速到停车.
        机动设计:a减小，再增加到0;speed较小到0
        """
        seeds_emer_stop = []
        j3, j4, j5 = 0.0, 0.0, 0.0
        t4, t5 = 0.0, 0.0
        for j1 in tqdm(np.linspace(self.jerk_min, self.jerk_max, self.num_samples * 2)):
            # for t1 in np.linspace(0, self.total_time, self.time_step_max + 1):
            if j1 > 0:
                continue

            for t1 in np.arange(0.0, self.total_time + 1e-9, 0.1):
                s1 = self.calculate_st_by_jt(s0, v0, a0, j1, t1)
                v1 = self.calculate_vt_by_jt(v0, a0, j1, t1)
                a1 = self.calculate_at_by_jt(a0, j1, t1)

                if not self.check_state(s1, v1, a1, s0, v0, a0):
                    continue

                for j2 in np.linspace(self.jerk_min, self.jerk_max, self.num_samples * 2):
                    if j2 < 0:
                        continue

                    for t2 in np.arange(0.0, self.total_time + 1e-9, 0.1):
                        if t1 + t2 >= self.total_time:
                            continue
                        a2 = self.calculate_at_by_jt(a1, j2, t2)
                        if a2 > 0.0 or a2 < -0.3:  # clamp 一个范围[-0.5,0.0]
                            continue
                        v2 = self.calculate_vt_by_jt(v1, a1, j2, t2)
                        if v2 > 0.0 or v2 < -0.3:  # clamp 一个范围[-0.5,0.0]
                            continue
                        s2 = self.calculate_st_by_jt(s1, v1, a1, j2, t2)
                        if s2 < s1 - 0.1:
                            continue

                        t3 = self.total_time - t1 - t2

                        seeds_emer_stop.append((j1, t1, j2, t2, j3, t3, j4, t4, j5, t5))
        return seeds_emer_stop

    def get_all_frenet_trajectory(self, s0, a0, v0, seeds, ft: JerkSpaceSamplingFrenetTrajectory = None):
        """批量生成速度轨迹"""
        if ft is None:
            ft = JerkSpaceSamplingFrenetTrajectory()
            ft.cost_total = 0.1

        # return [self.get_one_frenet_trajectory(s0=s0, v0=v0, a0=a0, seed=seed, jssft=ft) for seed in seeds]
        return [self.get_one_frenet_trajectory(s0=s0, v0=v0, a0=a0, seed=seed, jssft=ft, id=id) for id, seed in enumerate(seeds)]

    def get_one_frenet_trajectory(self, s0, v0, a0, seed, jssft: JerkSpaceSamplingFrenetTrajectory, id: int) -> JerkSpaceSamplingFrenetTrajectory:
        """根据jerk-t空间采样结果生成一条速度轨迹"""
        ft = copy.deepcopy(jssft)
        ft.cost_total = ft.cost_total + id * 0.001
        j1, t1, j2, t2, j3, t3, j4, t4, j5, t5 = seed

        s1 = self.calculate_st_by_jt(s0, v0, a0, j1, t1)
        v1 = self.calculate_vt_by_jt(v0, a0, j1, t1)
        a1 = self.calculate_at_by_jt(a0, j1, t1)

        s2 = self.calculate_st_by_jt(s1, v1, a1, j2, t2)
        v2 = self.calculate_vt_by_jt(v1, a1, j2, t2)
        a2 = self.calculate_at_by_jt(a1, j2, t2)

        s3 = self.calculate_st_by_jt(s2, v2, a2, j3, t3)
        v3 = self.calculate_vt_by_jt(v2, a2, j3, t3)
        a3 = self.calculate_at_by_jt(a2, j3, t3)

        s4 = self.calculate_st_by_jt(s3, v3, a3, j4, t4)
        v4 = self.calculate_vt_by_jt(v3, a3, j4, t4)
        a4 = self.calculate_at_by_jt(a3, j4, t4)

        for t in self.time_array:
            dt = 0
            if t < t1 - 1e-9:
                ft.t.append(t)

                dt = t
                ft.s_ddd.append(j1)
                ft.s_dd.append(self.calculate_at_by_jt(a0, j1, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v0, a0, j1, dt))
                ft.s.append(self.calculate_st_by_jt(s0, v0, a0, j1, dt))

            elif t < t1 + t2 + 1e-9:
                ft.t.append(t)

                dt = t - t1
                ft.s_ddd.append(j2)
                ft.s_dd.append(self.calculate_at_by_jt(a1, j2, dt))
                # if self.calculate_vt_by_jt(v1, a1, j2, dt) < 0.0:
                #     aaa = 1
                ft.s_d.append(self.calculate_vt_by_jt(v1, a1, j2, dt))
                ft.s.append(self.calculate_st_by_jt(s1, v1, a1, j2, dt))

            elif t < t1 + t2 + t3 + 1e-9:
                ft.t.append(t)

                dt = t - t1 - t2
                ft.s_ddd.append(j3)
                ft.s_dd.append(self.calculate_at_by_jt(a2, j3, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v2, a2, j3, dt))
                # if self.calculate_vt_by_jt(v2, a2, j3, dt) < 0.0:
                #     aaa = 1
                ft.s.append(self.calculate_st_by_jt(s2, v2, a2, j3, dt))

            elif t < t1 + t2 + t3 + t4 + 1e-9:
                ft.t.append(t)

                dt = t - t1 - t2 - t3
                ft.s_ddd.append(j4)
                ft.s_dd.append(self.calculate_at_by_jt(a3, j4, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v3, a3, j4, dt))
                # if self.calculate_vt_by_jt(v3, a3, j4, dt) < 0.0:
                #     aa = 1
                ft.s.append(self.calculate_st_by_jt(s3, v3, a3, j4, dt))

            else:
                ft.t.append(t)

                dt = t - t1 - t2 - t3 - t4
                ft.s_ddd.append(j5)
                ft.s_dd.append(self.calculate_at_by_jt(a4, j5, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v4, a4, j5, dt))
                # if self.calculate_vt_by_jt(v4, a4, j5, dt) < 0.0:
                #     aa = 1
                ft.s.append(self.calculate_st_by_jt(s4, v4, a4, j5, dt))

        return ft

    def get_all_frenet_trajectory_stop(self, s0, a0, v0, seeds, ft: JerkSpaceSamplingFrenetTrajectory = None, cost_total_max=0.1):
        """批量生成停车速度轨迹"""
        if ft is None:
            ft = JerkSpaceSamplingFrenetTrajectory()
            ft.cost_total = cost_total_max

        # return [self.get_one_frenet_trajectory(s0=s0, v0=v0, a0=a0, seed=seed, jssft=ft) for seed in seeds]
        return [self.get_one_frenet_trajectory_stop(s0=s0, v0=v0, a0=a0, seed=seed, jssft=ft, id=id) for id, seed in enumerate(seeds)]

    def get_one_frenet_trajectory_stop(
        self, s0, v0, a0, seed, jssft: JerkSpaceSamplingFrenetTrajectory, id: int
    ) -> JerkSpaceSamplingFrenetTrajectory:
        """根据jerk-t空间采样结果生成一条停车速度轨迹"""
        ft = copy.deepcopy(jssft)
        ft.cost_total = ft.cost_total + id * 0.01
        j1, t1, j2, t2, j3, t3, j4, t4, j5, t5 = seed
        # j1, t1, j2, t2, 0.0, t3, 0.0, 0.0, 0.0, 0.0 = seed

        s1 = self.calculate_st_by_jt(s0, v0, a0, j1, t1)
        v1 = self.calculate_vt_by_jt(v0, a0, j1, t1)
        a1 = self.calculate_at_by_jt(a0, j1, t1)

        s2 = self.calculate_st_by_jt_clamp(s1, v1, a1, j2, t2)
        # v2 = self.calculate_vt_by_jt_clamp(v1, a1, j2, t2)
        # a2 = self.calculate_at_by_jt(a1, j2, t2)

        # s3 = s2
        # v3 = 0.0
        # a3 = 0.0

        for t in self.time_array:
            dt = 0
            if t < t1 + 1e-9:
                ft.t.append(t)

                dt = t
                ft.s_ddd.append(j1)
                ft.s_dd.append(self.calculate_at_by_jt(a0, j1, dt))
                ft.s_d.append(self.calculate_vt_by_jt(v0, a0, j1, dt))
                ft.s.append(self.calculate_st_by_jt(s0, v0, a0, j1, dt))

            elif t < t1 + t2 + 1e-9:
                ft.t.append(t)

                dt = t - t1
                ft.s_ddd.append(j2)
                ft.s_dd.append(self.calculate_at_by_jt(a1, j2, dt))
                ft.s_d.append(self.calculate_vt_by_jt_clamp(v1, a1, j2, dt))
                ft.s.append(self.calculate_st_by_jt_clamp(s1, v1, a1, j2, dt))

            else:
                ft.t.append(t)

                dt = t - t1 - t2
                ft.s_ddd.append(0.0)
                ft.s_dd.append(0.0)
                ft.s_d.append(0.0)
                ft.s.append(s2)

        return ft


def test_jssp():
    def check_dir(target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    OUTPUT_DIR = os.path.join(dir_current_file, "cache_figures")
    check_dir(OUTPUT_DIR)
    fig_path = os.path.join(OUTPUT_DIR, "{fig_name}.svg".format(fig_name="polyvt-sampling-"))

    s0, v0, a0 = 100.0, 5.0, 0.0
    pvt_sampler = PolyVTSampling(total_time=5.0, delta_t=0.1, jerk_min=-10.0, jerk_max=10.0, a_min=-8.0, a_max=8.0, v_max=18.0, num_samples=21)
    seeds_t = []
    seeds = pvt_sampler.sampling(s0, v0, a0)  # 不同at线形直线拼接采样种子.
    all_traj = pvt_sampler.get_all_frenet_trajectory(s0=s0, v0=v0, a0=a0, seeds=seeds)
    seeds_stop = pvt_sampler.sampling_stop_seeds_supplement(s0=s0, v0=v0, a0=a0)
    all_traj_stop = pvt_sampler.get_all_frenet_trajectory_stop(s0=s0, v0=v0, a0=a0, seeds=seeds_stop, cost_total_max=all_traj[-1].cost_total)
    all_traj_concat = all_traj + all_traj_stop
    # plot_jerk_sampling(all_traj)
    plot_jerk_sampling_of_best_traj(all_traj_concat, best_traj_id=33, dir_figure=fig_path)
    # pass  polyvt-sampling-去掉极点-无补充种子采样.png


def test_jssp():
    def check_dir(target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    OUTPUT_DIR = os.path.join(dir_current_file, "cache_figures")
    check_dir(OUTPUT_DIR)
    fig_path_png = os.path.join(OUTPUT_DIR, "{fig_name}.png".format(fig_name="polyvt-sampling-"))
    fig_path_svg = os.path.join(OUTPUT_DIR, "{fig_name}.svg".format(fig_name="polyvt-sampling-"))

    s0, v0, a0 = 100.0, 5.0, 0.0
    pvt_sampler = PolyVTSampling(total_time=5.0, delta_t=0.1, jerk_min=-10.0, jerk_max=10.0, a_min=-8.0, a_max=8.0, v_max=18.0, num_samples=15)

    start_time = time.time()

    seeds = pvt_sampler.sampling(s0, v0, a0)  # 不同at线形直线拼接采样种子.
    all_traj = pvt_sampler.get_all_frenet_trajectory(s0=s0, v0=v0, a0=a0, seeds=seeds)
    end_time_1 = time.time()
    print(f"###log### jssp 用时:{end_time_1 - start_time}\n")

    seeds_stop = pvt_sampler.sampling_stop_seeds_supplement(s0=s0, v0=v0, a0=a0)
    all_traj_stop = pvt_sampler.get_all_frenet_trajectory_stop(s0=s0, v0=v0, a0=a0, seeds=seeds_stop, cost_total_max=all_traj[-1].cost_total)
    all_traj_concat = all_traj + all_traj_stop
    print(f"##log## all_traj num ={len(all_traj)}.")
    print(f"##log## all_traj_stop num ={len(all_traj_stop)}.")
    # plot_jerk_sampling(all_traj)
    plot_jerk_sampling_of_best_traj(all_traj_concat, best_traj_id=33, dir_figure=fig_path_png, dir_svg=fig_path_svg)
    # pass  polyvt-sampling-去掉极点-无补充种子采样.png


def test_quintic_polynomial():
    def check_dir(target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    OUTPUT_DIR = os.path.join(dir_current_file, "cache_figures")
    check_dir(OUTPUT_DIR)
    fig_path_png = os.path.join(OUTPUT_DIR, "{fig_name}.png".format(fig_name="smpling_quintic_polynomial-"))
    fig_path_svg = os.path.join(OUTPUT_DIR, "{fig_name}.svg".format(fig_name="smpling_quintic_polynomial-"))

    s0, v0, a0 = 100.0, 5.2, 0.5
    pvt_sampler = PolyVTSampling(total_time=5.0, delta_t=0.1, jerk_min=-10.0, jerk_max=10.0, a_min=-8.0, a_max=8.0, v_max=18.0, num_samples=21)

    start_time = time.time()
    frenet_trajectories = pvt_sampler.smpling_quintic_polynomial(s0, v0, a0, num_sampling=880)  # 不同at线形直线拼接采样种子.
    end_time_1 = time.time()
    print(f"###log### test_quintic_polynomial 用时:{end_time_1 - start_time}\n")
    plot_jerk_sampling_of_best_traj(frenet_trajectories, best_traj_id=33, dir_figure=fig_path_png, dir_svg=fig_path_svg)

    # pass  polyvt-sampling-去掉极点-无补充种子采样.png


if __name__ == "__main__":
    # test
    test_jssp()
    # test_quintic_polynomial()
