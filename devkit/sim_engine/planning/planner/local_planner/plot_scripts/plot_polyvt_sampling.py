import copy

import os, sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection


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

from devkit.common.coordinate_system.frenet import JerkSpaceSamplingFrenetTrajectory
from devkit.sim_engine.planning.planner.local_planner.utils.polyvt_sampling import PolyVTSampling


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

    return lc


def plot_jerk_sampling_of_best_traj(trajectories: list, best_traj_id: int, dir_png: str = "polyvt-sampling-.png"):
    """确保绘图中每条轨迹的cost_total较小的显示在上层，并且最佳轨迹（best_traj_id指定的轨迹）突出显示"""
    plt.ioff()  # 关闭交互模式
    fig = plt.figure(figsize=[30, 15])
    fig.tight_layout()
    gs = GridSpec(2, 3, width_ratios=[20, 20, 1], figure=fig)

    ax_st = fig.add_subplot(gs[0, 0])
    ax_vt = fig.add_subplot(gs[0, 1])
    ax_at = fig.add_subplot(gs[1, 0])
    ax_jt = fig.add_subplot(gs[1, 1])
    ax_cost_colorbar = fig.add_subplot(gs[:, 2])

    ax_st.set_ylabel("s [m]")
    ax_st.set_title("s-t")
    ax_vt.set_ylabel("v [m/s]")
    ax_vt.set_title("v-t")
    ax_at.set_ylabel("a [m/s²]")
    ax_at.set_title("a-t")
    ax_jt.set_ylabel("jerk [m/s³]")
    ax_jt.set_title("jerk-t")
    ax_cost_colorbar.set_title("spd traj cost", pad=10, fontsize=10, loc="center")

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
    lc = multiline_add_best(list_t=list_t, list_state=list_s, c=costs, ax=ax_st, cmap="rainbow", lw=0.2, best_traj_id=len(trajectories) - 1)
    colorbar = fig.colorbar(lc, cax=ax_cost_colorbar)
    multiline_add_best(list_t=list_t, list_state=list_s_d, c=costs, ax=ax_vt, cmap="rainbow", lw=0.2, best_traj_id=len(trajectories) - 1)
    multiline_add_best(list_t=list_t, list_state=list_s_dd, c=costs, ax=ax_at, cmap="rainbow", lw=0.2, best_traj_id=len(trajectories) - 1)
    multiline_add_best(list_t=list_t, list_state=list_s_ddd, c=costs, ax=ax_jt, cmap="rainbow", lw=0.2, best_traj_id=len(trajectories) - 1)

    num_trajs = len(trajectories)
    text_title = f"One path sampled {num_trajs} speed curves"
    fig.suptitle(text_title, fontsize=16)

    plt.savefig(dir_png)
    plt.close()
    print("##log##绘图完成.")


if __name__ == "__main__":
    pass
    # test

    s0, v0, a0 = 100.0, 8.3, 2.4

    pvt_sampler = PolyVTSampling(total_time=5.0, delta_t=0.1, jerk_min=-12, jerk_max=12, a_min=-8, a_max=8, v_max=20, num_samples=25)
    seeds = pvt_sampler.sampling(s0, v0, a0)  # 不同at线形直线拼接采样种子.
    all_traj = pvt_sampler.get_all_frenet_trajectory(s0, v0, a0, seeds)
