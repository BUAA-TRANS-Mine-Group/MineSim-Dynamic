# Python library
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

# Third-party library
import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.path import Path as MatplotlibPath
from matplotlib.path import Path
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib.ticker import FuncFormatter

# Local library
from devkit.common.actor_state.state_representation import StateVector2D
from devkit.common.actor_state.ego_state import EgoState
from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMapExplorer
from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner
from devkit.sim_engine.history.simulation_history import SimulationHistory
from devkit.configuration.sim_engine_conf import SimConfig as sim_config
from devkit.visualization.plot_data import AgentStatePlot
from devkit.visualization.plot_data import SimulationIterationTimeSequence

logger = logging.getLogger(__name__)


def check_dir(target_dir: str) -> None:
    """Check and create the directory if it doesn't exist.

    Parameters:
    - target_dir (str):The directory path to be checked.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)  # If directory already exists,don't raise an error.


def multiline(xs, ys, c, ax=None, **kwargs):
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

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    # ax.autoscale()

    return lc


def multiline_add_best(list_x, list_state, c, ax=None, best_traj_id=None, zorder_trajs=2, zorder_bset_traj=3, **kwargs):
    """
    Plot lines with different colorings, highlighting the best trajectory in bold green.

    Parameters
    ----------
    list_x : iterable container of x coordinates
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

    segments_t_state = [np.column_stack([t[: len(state)], state]) for t, state in zip(list_x, list_state)]
    lc = LineCollection(segments_t_state, zorder=zorder_trajs, **kwargs)

    # Set coloring of line segments
    lc.set_array(np.asarray(c))
    # Add lines to axes and rescale
    ax.add_collection(lc)
    # ax.autoscale()

    # Highlight the best trajectory if specified
    if best_traj_id is not None and 0 <= best_traj_id < len(list_x):
        best_traj_segments = np.column_stack([list_x[best_traj_id][: len(list_state[best_traj_id])], list_state[best_traj_id]])
        lc_best = LineCollection([best_traj_segments], colors="purple", linewidths=0.6, zorder=zorder_bset_traj)
        ax.add_collection(lc_best)

        # Draw a small circle every 10 points along the best trajectory
        for i in range(0, len(best_traj_segments), 10):
            ax.scatter(best_traj_segments[i][0], best_traj_segments[i][1], color="purple", s=10, zorder=zorder_bset_traj + 1)

    return lc


class PlanVisualizer2D:
    """
    - PlanVisualizer2D : 参考 OnSite 1.0 的可视化方案;
    - Visualizer2D : 参考 OnSite 2.0 的可视化方案;
    """

    def __init__(self, bitmap_type="bitmap_mask"):
        """
        Ref:
        - RGB Calculator:  https://www.w3schools.com/colors/colors_rgb.asp
        - https://colordrop.io/  https://colordrop.io/flat/
        """
        self.bitmap_type = bitmap_type

        self.xy_margin = 5.0
        self.color_map = {
            # ########## 地图元素 ##########
            "bitmap": None,  # !标记为None,表示后续会自动配置
            "semantic_map_polygon": None,
            "semantic_map_road_block": None,
            "semantic_map_refpath": None,
            "semantic_map_borderline": "#003366",
            "semantic_map_dubins_pose": None,
            # ########## 交通参与者 agent ##########
            "ego_box": "#26ae60",
            "ego_yaw_triangle": "#155a32",  # "darkgreen",
            "veh_box": "#3398da",  # "cornflowerblue",
            "veh_yaw_triangle": "#154361",  # "darkblue",
            "static_obstacle": "#6A0573",
            # ########## 轨迹信息 ##########
            "veh_history_traj": "#244ab4",
            "ego_history_traj": "#22551a",
            "veh_predict_trajs": None,
            "veh_predict_trajs_points": None,
            "veh_plan_trajs": None,
            "veh_plan_best_traj": None,
            # ########## other ##########
            "ego_start_pose": "#f39c13",
            "ego_goal": "#FF6347",  # goal="tomato",  # "#B03060",
            "ego_goal_pose": "#f39c13",
            "warning_signal": "#e60000",
        }
        self.zorders = {
            # ########## 地图元素 ##########
            "bitmap": 2,
            "semantic_map_polygon": 3,
            "semantic_map_road_block": 4,
            "semantic_map_refpath": 4,
            "semantic_map_borderline": 5,
            "semantic_map_dubins_pose": 6,
            # ########## 交通参与者 agent ##########
            "ego_box": 10,
            "ego_yaw_triangle": 11,
            "veh_box": 10,
            "veh_yaw_triangle": 11,
            "static_obstacle": 10,
            "veh_id": 31,
            "static_obstacles_id": 31,
            # ########## 轨迹信息 ##########
            "veh_history_traj": 20,
            "veh_predict_trajs": 20,
            "veh_predict_trajs_points": 21,
            "veh_plan_trajs": 23,
            "veh_plan_best_traj": 24,
            # ########## other ##########
            "ego_start_pose": 7,
            "ego_goal": 7,
            "ego_goal_pose": 8,
            "warning_signal": 40,
            "axbg_tick_params": 41,  # Ensure this is higher than any other zorder in use
            "ax_text": 999,  # 创建 ax_text：位于最上方zorder，并可用于显示文本
        }
        self.fontsize = {
            "axcolorbar_title": 20,
            "axcolorbar_label": 20,  # 坐标轴xy标签 字体大小
            "axcolorbar_tick": 20,  # 坐标轴刻度字体大小
            "colorbar_tick": 20,  # colorbar刻度字体大小
            "axbg_title": 20,
            "axbg_label": 20,  # 坐标轴 标签字体大小
            "axbg_tick": 20,  # 坐标轴刻度字体大小
            "table_font": 18,
            "text_testinfo": 20,
            "static_obstacles_id": 22,
            "veh_id": 22,
        }
        self.alphas = {
            "map_polygon_layer": 0.6,  # 面图层的透明度
            "map_refpath": 0.8,  # 线的透明度
            "map_borderline": 1.0,  # 坐标轴刻度字体大小
        }
        self.linewidths = {
            "map_polygon_layer": 2.0,  # 面图层的边
            "map_refpath": 2.2,
            "map_borderline": 2.0,
            "vehicle_box": 2.0,
            "ego_goal_box": 3.0,
            "veh_predict_trajs": 3.5,
            "multiline_plan_sampling": 3.5,
        }

        self.resampling_interval_connector_path = 20
        self.resampling_interval_base_path = 5
        self.resampling_interval_edge_borderline = 30
        self.resampling_interval_inner_borderline = 5
        self.history_smooth = 2

        self.flag_visilize = True  # 是\否进行可视化绘图
        self.flag_plot_refpaths = True  # 是否可视化all参考路径
        self.flag_plot_borderline = False  # ! rgb_mask默认要绘制
        self.flag_plot_road_block = True
        self.flag_plot_route_refpath = True
        self.flag_plot_vehicle_history = True
        self.flag_save_fig_whitout_show = True  # 绘图保存,但是不显示
        # self.flag_plot_prediction = True  # 是否可视化轨迹预测结果
        self.flag_close_axcolorbar = True

        self.flag_plot_hdmaps = False  # ! hdmaps已经被绘图(仅绘制一次即可)标志位

        self.now_t = 0.0
        self.now_t_str: str = "0.0"
        self.now_t_step: int = 0

        self.fig = None
        self.axbg = None
        self.axveh = None
        self.ax_plan_traj = None
        self.axcolorbar = None
        self.colorbar = None
        self.ax_text = None

        self.result_path_png = None
        self.result_path_svg = None

        # self.predictor = None  # 预测器信息
        self.local_planner = None  # 规划器信息

    def init(self, scenario: AbstractScenario, planner: AbstractPlanner, dynamic_part_log_name: str):
        """
        创建 坐标轴i:
        - fig, axbg(主轴,带坐标刻度)，图上唯一可见的刻度就是主轴 self.axbg 的 x、y 轴;
        - axveh, ax_plan_traj(两个twiny,隐藏刻度,透明背景)，默认的 Axes facecolor（通常是白色）;
        - ax_text(最高zorder,可绘制文本)。
        """
        # 测试信息
        self.scenario_name = scenario.scenario_name
        self.scenario_type = scenario.scenario_type
        self.scenario = scenario

        # ------ 1) 计算绘图区域 & 初始化 figure ------
        self._get_xy_max_min(scenario=scenario)
        figsize_x = 20
        aspect_ratio = (self.y_max - self.y_min) / (self.x_max - self.x_min)  # 动态计算figsize_y以保持纵横比
        figsize_y = figsize_x * aspect_ratio

        # 创建一个具有两个子图的图表,左侧是主图,右侧是 colorbar; width_ratios:总宽度为21,包括20的主图和1的colorbar
        self.fig = plt.figure(figsize=(figsize_x + 1.0, figsize_y))
        gs = GridSpec(1, 2, width_ratios=[20, 1], figure=self.fig)

        # ------ 2) 主轴：axbg ------
        self.axbg = self.fig.add_subplot(gs[0])
        self.axbg.set_xlim(self.x_min, self.x_max)
        self.axbg.set_ylim(self.y_min, self.y_max)
        self.axbg.set_aspect("equal", adjustable="datalim")
        self._ensure_ticks_labels_visible(self.axbg)  # 给主轴保留刻度

        # ------ 3) 隐藏或保留 colorbar 轴 ------
        if not self.flag_close_axcolorbar:
            self.axcolorbar = self.fig.add_subplot(gs[1])
            # 如果需要设置色条标题
            self.axcolorbar.set_title("traj cost", pad=10, loc="center", fontdict={"size": f"{self.fontsize['axcolorbar_title']}"})
        else:
            self.axcolorbar = None

        # ------ 4) 创建 axveh、ax_plan_traj: twiny + 背景透明 + 无刻度 ------
        self.axveh = self.axbg.twiny()  # 共享 y 轴
        self.ax_plan_traj = self.axbg.twiny()  # 共享 y 轴
        for ax_sub in [self.axveh, self.ax_plan_traj]:
            self._sync_axes(ax_sub)
            self._hide_scale_labels_sub_axis(ax_sub)

        # ------ 5) 创建 ax_text ：位于最上方zorder，并可用于显示文本 ------
        self.ax_text = self.axbg.twiny()  # 共享 y 轴
        self._sync_axes(self.ax_text)
        self.ax_text.set_axis_off()  # 如果只想当文本绘制层，不需要坐标轴
        self.ax_text.set_facecolor((0, 0, 0, 0))  # 让背景透明

        # ------ 6) 处理交互模式 & 输出路径 ------
        if self.flag_save_fig_whitout_show:
            plt.ioff()
        else:
            plt.ion()

        self._prepare_output_directories(planner=planner, scenario=scenario, dynamic_part_log_name=dynamic_part_log_name)

    def _prepare_output_directories(self, planner: AbstractPlanner, scenario: AbstractScenario, dynamic_part_log_name: str):
        """
        对 result_path_png/result_path_svg 做的文件夹检查
        """
        dir_outputs_figure = sim_config["directory_conf"]["dir_outputs_figure"]
        self.scenario_name = scenario.scenario_name
        from devkit.visualization.configuration.visualizer_conf import log_number

        self.result_path_png = os.path.join(
            dir_outputs_figure,
            "gif_cache_png",
            planner.name(),
            self.scenario_name,
            f"{dynamic_part_log_name}-number-{log_number}",
        )
        self.result_path_svg = os.path.join(
            dir_outputs_figure,
            "gif_cache_svg",
            planner.name(),
            self.scenario_name,
            f"{dynamic_part_log_name}-number-{log_number}",
        )

        for dpath in [self.result_path_png, self.result_path_svg]:
            if not os.path.exists(dpath):
                os.makedirs(dpath)
                print(f"###log### Target directory: {dpath} Created...")

    def _ensure_ticks_labels_visible(self, ax):
        """
        设置给定轴 ax 的刻度、刻度大小及可见性。只对 axbg 这样需要显式刻度的轴用。
        """
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=6,
            width=2,
            colors="black",
            labelsize=self.fontsize["axbg_label"],
        )

    def _sync_axes(self, ax):
        """将子图 ax 的 xlim, ylim, aspect 与主轴 axbg 同步."""
        if ax is None:
            return
        ax.set_xlim(self.axbg.get_xlim())
        ax.set_ylim(self.axbg.get_ylim())
        ax.set_aspect("equal", adjustable="datalim")

    def _hide_scale_labels_sub_axis(self, ax):
        """
        隐藏子轴 ax 的刻度、边框、使其成为透明背景。
        """
        ax.set_frame_on(False)  # 直接把这条轴的边框,刻度关掉
        ax.set_facecolor((0, 0, 0, 0))
        ax.set_xticks([])  # 移除x轴刻度
        ax.set_yticks([])  # 移除y轴刻度
        ax.set_xticklabels([])  # 移除x轴刻度标签
        ax.set_yticklabels([])  # 移除y轴刻度标签

    def _sync_all_axes_sets(self):
        """
        若在某些阶段(比如每帧绘制后)需要统一同步一下:
        """
        for ax_sub in [self.axveh, self.ax_plan_traj]:
            self._sync_axes(ax_sub)
        # ax_text 如果只用于文本层且 set_axis_off()，也不必同步
        # 若想要 ax_text 显示坐标，就在这里再 _sync_axes(self.ax_text)

    def _get_xy_max_min(self, scenario: AbstractScenario):
        self.x_target = list(scenario.scenario_info.test_setting["goal"]["x"])
        self.y_target = list(scenario.scenario_info.test_setting["goal"]["y"])
        x_start = [
            scenario.scenario_info.test_setting["start_ego_info"]["x"] - scenario.scenario_info.test_setting["start_ego_info"]["shape"]["length"],
            scenario.scenario_info.test_setting["start_ego_info"]["x"] + scenario.scenario_info.test_setting["start_ego_info"]["shape"]["length"],
        ]
        y_start = [
            scenario.scenario_info.test_setting["start_ego_info"]["y"] - scenario.scenario_info.test_setting["start_ego_info"]["shape"]["length"],
            scenario.scenario_info.test_setting["start_ego_info"]["y"] + scenario.scenario_info.test_setting["start_ego_info"]["shape"]["length"],
        ]

        xs = self.x_target + x_start + [scenario.scenario_info.test_setting["x_max"], scenario.scenario_info.test_setting["x_min"]]
        ys = self.y_target + y_start + [scenario.scenario_info.test_setting["y_max"], scenario.scenario_info.test_setting["y_min"]]

        # 获得该场景绘图的地理边界
        self.x_max = max(xs) + self.xy_margin
        self.x_min = min(xs) - self.xy_margin
        self.y_max = max(ys) + self.xy_margin
        self.y_min = min(ys) - self.xy_margin

    def plot_scenario(self, scenario: AbstractScenario, planner: AbstractPlanner, flag_plot_route_refpath: bool = False):
        self.flag_plot_route_refpath = flag_plot_route_refpath
        # draw reference path with dots
        if self.flag_plot_route_refpath:
            if planner.global_route_planner:
                self.plot_reference_path(planner=planner)

        self._plot_hdmaps(scenario=scenario)
        self.flag_plot_hdmaps = True

        self._plot_goal_box(scenario=scenario)
        self._plot_start_pose_and_end_pose(scenario=scenario)

    def plot_reference_path(self, planner: AbstractPlanner):
        """axbg轴，一次即可。"""
        route_planner = planner.global_route_planner
        x_list = route_planner.refline_smooth[:, 0]
        y_list = route_planner.refline_smooth[:, 1]
        temp_color = self.color_map["route_reference_path"]
        self.axbg.plot(x_list, y_list, color=temp_color, linestyle="--", alpha=1.0, lw=self.linewidths["map_refpath"])

    @classmethod
    def get_simulatiion_all_frames(self, scenario: AbstractScenario, planner: AbstractPlanner, simulation_history: SimulationHistory):
        """重新整理来自 simulation_log 中的记录数据"""
        self.simulation_history = simulation_history
        self.sim_times = SimulationIterationTimeSequence(sim_step_num=len(simulation_history.extract_ego_state))
        self.agents_plot = AgentStatePlot(sim_times=self.sim_times)
        self.agents_plot.update_data_sources(history=simulation_history)

    def update_base_info(self, scenario: AbstractScenario, simulation_history: SimulationHistory, index: int) -> None:
        """在 axveh 轴子图中 进行绘制.
        - cla(self.axveh)，在上面绘制车辆;
        - 仅需一次对 axveh 同步，直接 self._sync_axes(self.axveh)
        """
        self.agents_plot.update_now_index(now_index=index)
        self.now_t_step = index
        self.now_t = round(scenario.scenario_info.test_setting["dt"] * self.now_t_step, 1)
        self.now_t_str = str(self.now_t)

        # 画布属性设置--------------------------
        # "axes.unicode_minus"：这是rcParams字典中的一个键，它控制是否使用Unicode负号而不是标准的ASCII负号。
        # 在某些情况下，特别是当使用中文或其他非ASCII字符集时，使用标准的ASCII负号可能会导致显示问题或乱码。
        plt.rcParams["axes.unicode_minus"] = False
        self.axveh.cla()  # 清除当前子图中的所有绘图内容
        self._sync_axes(self.axveh)

        # ----------------绘制所有的车辆-----------------------
        self._plot_other_vehicles(agents_plots=self.agents_plot)
        self._plot_ego_vehicle(scenario=scenario, simulation_history=simulation_history, index=index)
        self.plot_ego_veh_history_traj()
        # self._plot_static_obstacle(observation)  # axveh轴,可以忽然消失

        # --------若发生碰撞,则绘制碰撞警告标志--------
        # if scenario.scenario_info.test_setting["end"] in [2, 3, 4]:
        #     self._plot_warning_signal(scenario=scenario)

        # GIF显示各类信息:显示测试相关信息 --------------------------
        ts_text = "Time_stamp: " + self.now_t_str + "s"
        name_text = "Test_scenario: " + str(self.scenario_name)
        type_text = "Scenario_type: " + str(self.scenario_type)

        self.ax_text.cla()  # 清空文本层
        self.ax_text.set_axis_off()  # 只画文本
        self.ax_text.text(
            0.02, 0.97, name_text, transform=self.axveh.transAxes, fontdict={"size": f"{self.fontsize['text_testinfo']}", "color": "black"}
        )
        self.ax_text.text(
            0.02, 0.94, type_text, transform=self.axveh.transAxes, fontdict={"size": f"{self.fontsize['text_testinfo']}", "color": "black"}
        )
        self.ax_text.text(
            0.02, 0.91, ts_text, transform=self.axveh.transAxes, fontdict={"size": f"{self.fontsize['text_testinfo']}", "color": "black"}
        )

        if not self.flag_close_axcolorbar:
            bar_title_text = "traj cost:"
            self.ax_text.text(
                0.90, 0.95, bar_title_text, transform=self.axveh.transAxes, fontdict={"size": f"{self.fontsize['text_testinfo']}", "color": "black"}
            )

        # GIF显示各类信息:显示所有车辆运行信息 --------------------------
        self._display_vehicle_table(scenario=scenario, simulation_history=simulation_history, index=index)

    def update_all_local_planning_info(
        self, scenario: AbstractScenario, planner: AbstractPlanner, simulation_history: SimulationHistory, index: int
    ) -> None:
        self.planner_name = planner.name()
        if planner.name() == "SimplePlanner":
            pass
        elif planner.name() == "IDMPlanner":
            pass
        elif planner.name() == "FrenetOptimalPlanner":
            # --------- 渲染 局部轨迹规划结果 ---------
            self.update_local_planning_frenet(scenario=scenario, planner=planner)
        elif planner.name() == "PredefinedManeuverModeSamplingPlanner":
            self.update_local_planning_SPPMM(scenario=scenario, planner=planner)
        else:
            pass
        

    def update_local_planning_frenet(self, scenario: AbstractScenario, planner: AbstractPlanner) -> None:
        """
        note:应该将 LineCollection 添加到主图（self.ax_plan_traj）,然后创建 colorbar 并将其关联到 colorbar 的轴（self.axcolorbar）.
        """
        self.ax_plan_traj.cla()
        self._sync_axes(self.ax_plan_traj)

        self.plot_sampling_traj_and_best_traj(scenario=scenario, planner=planner)

        # plt.title(
        #     "{method} sampling total num:{len_total}, pass check num:{len_pass}".format(
        #         method=local_planner.method,
        #         len_total=len(local_planner.fplanner.all_path_sampling_trajs[self.now_t_step]),
        #         len_pass=len(local_planner.fplanner.all_trajs[self.now_t_step]),
        #     ),
        #     fontdict={"size": f"{self.fontsize['axbg_title']}", "color": "black"},
        # )  # 上标题
        # self.axbg.set_title("{method} ".format(method=planner.name()), fontdict={"size": f"{self.fontsize['axbg_title']}", "color": "black"})

        # self.axbg.set_title(
        #     "{method} sampling total num:{len_total}, pass check num:{len_pass}".format(
        #         method=local_planner.method,
        #         len_total=len(local_planner.fplanner.all_trajs_before_check[self.now_t_step]),
        #         len_pass=len(local_planner.fplanner.all_trajs[self.now_t_step]),
        #     ),
        #     fontdict={"size": f"{self.fontsize['axbg_title']}", "color": "black"},
        # )

    def update_local_planning_SPPMM(self, scenario: AbstractScenario, local_planner) -> None:
        # TODO
        pass

    def plot_other_veh_history_traj(self, scenario: AbstractScenario, id_veh):
        t_keys = [
            str(round(t, 2))
            for t in np.arange(
                0.0, scenario.scenario_info.test_setting["t"] - scenario.scenario_info.test_setting["dt"], scenario.scenario_info.test_setting["dt"]
            )
        ]
        # 绘制其他车的历史轨迹
        if id_veh in self.predictor.vehi_traj_GT.keys():
            vehi_traj_GT = self.predictor.vehi_traj_GT[id_veh]
            trajs_history_x = []
            trajs_history_y = []
            for t_key in t_keys:
                if t_key in vehi_traj_GT.keys():
                    trajs_history_x.append(vehi_traj_GT[t_key]["x"])
                    trajs_history_y.append(vehi_traj_GT[t_key]["y"])
            if len(trajs_history_y) > 5:
                self.axveh.scatter(
                    trajs_history_x,
                    trajs_history_y,
                    color=self.color_map["veh_history_traj"],
                    alpha=0.6,
                    zorder=self.zorders["veh_history_traj"],
                    s=30,
                )

    def plot_ego_veh_history_traj(self):
        ego_history: List[EgoState] = self.simulation_history.extract_ego_state[: self.now_t_step]

        if len(ego_history) > 5:
            trajs_history_x = []
            trajs_history_y = []
            for ego_state in ego_history:
                trajs_history_x.append(ego_state.rear_axle.x)
                trajs_history_y.append(ego_state.rear_axle.y)
            self.axveh.scatter(
                trajs_history_x,
                trajs_history_y,
                color=self.color_map["ego_history_traj"],
                alpha=0.6,
                zorder=self.zorders["veh_history_traj"],
                s=30,
            )

    def plot_single_veh_future_traj_GT(self, scenario: AbstractScenario, id_veh, point_mode="END_POINT"):
        """绘制其他车的future_traj_GT
        note：用于轨迹预测器算法的可视化。
        """
        if id_veh in self.predictor.vehi_traj_GT.keys():
            t_keys = [
                str(round(t, 2))
                for t in np.arange(
                    scenario.scenario_info.test_setting["t"],
                    scenario.scenario_info.test_setting["t"] + self.predictor.lane_prob_inferrer.time_horizon,
                    scenario.scenario_info.test_setting["dt"],
                )
            ]

            vehi_traj_GT = self.predictor.vehi_traj_GT[id_veh]
            trajs_GT_x = []
            trajs_GT_y = []
            color = (0.0, 0.55, 0.37, 1.0)
            for t_key in t_keys:
                if t_key in vehi_traj_GT.keys():
                    trajs_GT_x.append(vehi_traj_GT[t_key]["x"])
                    trajs_GT_y.append(vehi_traj_GT[t_key]["y"])
            #  绘制轨迹线段
            self.ax_predict_traj.plot(
                trajs_GT_x,
                trajs_GT_y,
                color=color,
                alpha=0.8,
                zorder=self.zorders["veh_predict_trajs"],
                linewidth=self.linewidths["veh_predict_trajs"],
            )

            if point_mode == "ONE_SECOND_POINT":  # 绘制圆点，每隔10个点,1秒画一个
                # 计算需要画多少个点（根据是否整除来避免在最后重复画一个点）
                num_points_to_scatter = len(trajs_GT_x)
                if num_points_to_scatter > 10:
                    num_points_to_scatter -= num_points_to_scatter % 10
                for i in range(0, num_points_to_scatter + 2, 10):
                    # 绘制圆点，使用比线段略宽的线宽
                    self.ax_predict_traj.scatter(
                        trajs_GT_x[i], trajs_GT_y[i], marker="d", color=color, alpha=1.0, zorder=self.zorders["veh_predict_trajs_points"] + 2, s=65
                    )
            elif point_mode == "END_POINT":  # 绘制最后一个点
                # 绘制圆点，使用比线段略宽的线宽
                self.ax_predict_traj.scatter(
                    trajs_GT_x[-1], trajs_GT_y[-1], marker="d", color=color, alpha=1.0, zorder=self.zorders["veh_predict_trajs_points"] + 2, s=65
                )

        pass

    def plot_sampling_traj_and_best_traj(self, scenario: AbstractScenario, planner: AbstractPlanner):
        # 先提取最佳轨迹，然后从列表中移除它
        # best_trajectory = trajectories.pop(best_traj_id)
        trajectories = planner.all_trajs[self.now_t_step]

        # 按成本对剩余轨迹进行降序排序，成本低的轨迹将被后绘制，因此会显示在上层
        trajectories = sorted(trajectories, key=lambda x: x.cost_total, reverse=True)

        # 将最佳轨迹添加回列表的末尾，确保它最后被绘制
        # trajectories.append(best_trajectory)

        costs = []
        xs = []
        ys = []

        for ft in trajectories:  # 第 now_t_step 时间步的规划结果
            costs.append(ft.cost_total)
            xs.append(ft.x[1:])
            ys.append(ft.y[1:])
        # 将 LineCollection 添加到主图 (self.ax_plan_traj)
        # lc = multiline(xs, ys, costs, ax=self.ax_plan_traj, cmap="RdYlGn_r", lw=0.2, zorder=20)  # 多个规划结果绘制的函数
        # 多个规划结果绘制的函数
        lc = multiline_add_best(
            list_x=xs,
            list_state=ys,
            c=costs,
            ax=self.ax_plan_traj,
            best_traj_id=len(trajectories) - 1,
            zorder_trajs=self.zorders["veh_plan_trajs"],
            zorder_bset_traj=self.zorders["veh_plan_best_traj"],
            # cmap="RdYlGn_r",#plasma
            cmap="plasma",  # plasma
            lw=self.linewidths["multiline_plan_sampling"],
        )

        if not self.flag_close_axcolorbar:
            # 创建新的 colorbar best_traj_id
            self.colorbar = self.fig.colorbar(lc, cax=self.axcolorbar)

            # 使用 FuncFormatter 定制刻度显示格式
            def format_one_decimal(x, _):
                return f"{x:.1f}"

            formatter = FuncFormatter(format_one_decimal)
            self.colorbar.ax.yaxis.set_major_formatter(formatter)

            # 保持原有字体大小设置（放在格式化之后）
            for label in self.colorbar.ax.get_yticklabels():
                label.set_fontsize(self.fontsize["colorbar_tick"])

    def adjust_layout_ensure_all_visible(self):
        """调整布局以确保所有内容都可见,
        show 或者save前操作.
        """
        self.fig.tight_layout()  # 或者使用subplots_adjust

    def save_figure_as_png(self):
        # self.adjust_layout_ensure_all_visible()

        fig_path = os.path.join(self.result_path_png, "{time_step}.png".format(time_step=self.now_t_step))
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        print("###log### png saved to:", fig_path)

    def save_figure_as_svg(self):
        fig_path = os.path.join(self.result_path_svg, "{time_step}.svg".format(time_step=self.now_t_step))
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print("###log### svg saved to:", fig_path)

    # !############################## 以下为onsite原先的 ########################################
    def plot_end_status(self, scenario: AbstractScenario, c="red"):
        """绘制场景结束状态.
        1)主车碰撞时的提醒标志;车与背景车是否发生碰撞,status=2
        2)主车已经到达目标区域,status=1.5,
        3)已到达场景终止时间max_t,status=1
        4)正常状态,status=-1
        """
        if scenario.scenario_info.test_setting["end"] == 2:
            for key, values in observation["vehicle_info"].items():
                if key == "ego":
                    # text_status = "End:collision occurred!"
                    text_status = "End:collision!"
                    x, y = [float(values[i]) for i in ["x", "y"]]
                    self.axveh.text(
                        x=x + 5,
                        y=y + 5,
                        s=text_status,
                        ha="left",
                        va="bottom",
                        color="black",
                        fontsize=20,
                    )
                    self.axveh.scatter(x, y, s=60, c=c, alpha=1.0, marker=(8, 1, 30), zorder=4)
        elif scenario.scenario_info.test_setting["end"] == 1.5:
            for key, values in observation["vehicle_info"].items():
                if key == "ego":
                    text_status = "End:reach the goal!"
                    x, y = [float(values[i]) for i in ["x", "y"]]
                    self.axveh.text(
                        x=x + 5,
                        y=y + 5,
                        s=text_status,
                        ha="left",
                        va="bottom",
                        color="black",
                        fontsize=20,
                    )
        elif scenario.scenario_info.test_setting["end"] == 1:
            for key, values in observation["vehicle_info"].items():
                if key == "ego":
                    text_status = "End:scene Max time exceeded!"
                    x, y = [float(values[i]) for i in ["x", "y"]]
                    self.axveh.text(
                        x=x + 5,
                        y=y + 5,
                        s=text_status,
                        ha="left",
                        va="bottom",
                        color="black",
                        fontsize=20,
                    )
        elif scenario.scenario_info.test_setting["end"] == -1:
            pass
        else:
            end_s = scenario.scenario_info.test_setting["end"]
            raise ValueError(f"#log# end ={end_s}")

    def _display_vehicle_table(self, scenario: AbstractScenario, simulation_history: SimulationHistory, index: int):
        colLabels = []
        v_list = []
        a_list = []
        now_time_str = self.agents_plot.sim_times.t_str_s[self.agents_plot.now_index]

        # ego vehicle
        colLabels.append("ego")
        v_list.append(round(simulation_history.extract_ego_state[index].dynamic_car_state.rear_axle_velocity_2d.x, 4))
        a_list.append(round(simulation_history.extract_ego_state[index].dynamic_car_state.acceleration, 4))

        # other vehicle
        for track_id_minesim in self.agents_plot.times_link_track_ids_minesim[index]:
            colLabels.append(track_id_minesim)
            v_list.append(round(self.agents_plot.data_sources[track_id_minesim][now_time_str].velocity.x, 4))
            if self.agents_plot.data_sources[track_id_minesim][now_time_str].acceleration:
                a_list.append(round(self.agents_plot.data_sources[track_id_minesim][now_time_str].acceleration.x, 4))
            else:
                a_list.append(round(-0.1111, 4))
                logger.warning("#log# agent acceleration filed is ERROR!")

        v = np.array(v_list).reshape(1, -1)
        a = np.array(a_list).reshape(1, -1)

        rowLabels = ["v(m/s)", "a(m/s2)"]
        cellTexts = np.vstack((v, a))

        # 创建表格
        info_table = self.ax_text.table(
            cellText=cellTexts,
            colLabels=colLabels,
            rowLabels=rowLabels,
            loc="bottom",  # 表格位置
            cellLoc="center",  # 单元格内文字位置
            # colWidths=[0.1] * len(colLabels)  # 设置每列的宽度
        )

        # 设置字体大小
        info_table.auto_set_font_size(False)
        info_table.set_fontsize(self.fontsize["table_font"])
        # 调整表格行高，不改变列宽
        info_table.scale(1, 1.7)  # 只增加行高，不变更列宽

        # 调整布局以避免表格超出图像边缘 被截断或挤压
        # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

    def _display_vehicle_table(self, scenario: AbstractScenario, simulation_history: SimulationHistory, index: int):
        colLabels = []
        v_list = []
        a_list = []
        now_time_str = self.agents_plot.sim_times.t_str_s[self.agents_plot.now_index]

        # ego vehicle
        colLabels.append("ego")
        ego_velocity = simulation_history.extract_ego_state[index].dynamic_car_state.rear_axle_velocity_2d.x
        ego_acceleration = simulation_history.extract_ego_state[index].dynamic_car_state.acceleration

        #  异常检查：如果速度或加速度为None或不合理，替换为NaN
        if ego_velocity is None or np.isnan(ego_velocity):
            v_list.append("NaN")
            logger.warning("#log# ego vehicle velocity is ERROR!")
        else:
            v_list.append(f"{round(ego_velocity, 2):.2f}")

        if ego_acceleration is None or np.isnan(ego_acceleration):
            a_list.append("NaN")
            logger.warning("#log# ego vehicle acceleration is ERROR!")
        else:
            a_list.append(f"{round(ego_acceleration, 2):.2f}")

        # other vehicle
        for track_id_minesim in self.agents_plot.times_link_track_ids_minesim[index]:
            colLabels.append(track_id_minesim)
            velocity_2d: StateVector2D = self.agents_plot.data_sources[track_id_minesim][now_time_str].velocity
            acceleration_2d: StateVector2D = self.agents_plot.data_sources[track_id_minesim][now_time_str].acceleration

            # 异常检查：如果速度或加速度为None或不合理，替换为NaN
            if velocity_2d is None:
                v_list.append("NaN")
                logger.warning(f"#log# {track_id_minesim} velocity is ERROR!")
            else:
                v_list.append(f"{round(velocity_2d.x, 2):.2f}")

            if acceleration_2d is None:
                a_list.append("NaN")
                logger.warning(f"#log# {track_id_minesim} acceleration is ERROR!")
            else:
                a_list.append(f"{round(acceleration_2d.x, 2):.2f}")

        # 将速度和加速度列表转换为numpy数组，并重新整形
        v = np.array(v_list).reshape(1, -1)
        a = np.array(a_list).reshape(1, -1)

        rowLabels = ["v(m/s)", "a(m/s2)"]
        cellTexts = np.vstack((v, a))

        # 创建表格
        info_table = self.ax_text.table(
            cellText=cellTexts,
            colLabels=colLabels,
            rowLabels=rowLabels,
            loc="bottom",  # 表格位置
            cellLoc="center",  # 单元格内文字位置
            # colWidths=[0.1] * len(colLabels)  # 设置每列的宽度
        )

        # 设置字体大小
        info_table.auto_set_font_size(False)
        info_table.set_fontsize(self.fontsize["table_font"])
        # 调整表格行高，不改变列宽
        info_table.scale(1, 1.7)  # 只增加行高，不变更列宽

        # 调整布局以避免表格超出图像边缘 被截断或挤压
        # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

    def _plot_other_vehicles(self, agents_plots: AgentStatePlot) -> None:
        for car_track_id in self.agents_plot.times_link_track_ids_minesim[self.agents_plot.now_index]:
            car_track_id_int = int(car_track_id)
            shape_info = self.scenario.scenario_info.vehicle_traj[car_track_id_int]["shape"]

            if self.flag_plot_vehicle_history:
                # 生成从当前时间到3秒前的时间序列，间隔为0.1秒
                time_list = [self.now_t - i * 0.1 for i in range(31) if self.now_t - i * 0.1 >= 0]
                # 格式化时间列表中的所有时间点为字符串，保留一位小数
                time_list_str = ["{:.1f}".format(t) for t in sorted(time_list)]
                self._plot_single_vehicle_with_history(car_track_id_int, time_list_str, shape_info)
            else:
                now_time_str = self.agents_plot.sim_times.t_str_s[self.agents_plot.now_index]
                agent = self.agents_plot.data_sources[car_track_id][now_time_str]
                self._plot_single_vehicle(
                    key=car_track_id,
                    x=agent.center.x,
                    y=agent.center.y,
                    yaw=agent.center.heading,
                    veh_shape=shape_info,
                    color=self.color_map["veh_box"],
                    plot_yaw_triangle=True,
                    plot_box_edge=False,
                    plot_vehicle_id=True,
                )

    def _plot_single_vehicle_with_history(self, car_track_id_int: int, time_list_str: list, shape_info: dict):
        """Draw the other vehicle's trajectory (no  ego vehicle) with  gradient colors."""

        target_vehicle_log = self.agents_plot.data_sources[str(car_track_id_int)]
        time_str_sub = [time_str for time_str in time_list_str if time_str in target_vehicle_log]

        import matplotlib.colors as mcolors

        base_color = np.array(mcolors.hex2color(self.color_map["veh_box"]))
        for i, time_str in enumerate(time_str_sub):
            # Reverse the index to start drawing from the oldest history to the newest
            reverse_index = len(time_str_sub) - i - 1

            # Skip some frames based on history_smooth value to smooth the trajectory
            if self.history_smooth != 0 and (reverse_index % self.history_smooth != 0):
                continue

            if time_str in target_vehicle_log:
                agent = target_vehicle_log[time_str]
            else:
                continue

            # The blending ratio between the whole base color and white achieves the color gradient
            alpha_f = min(reverse_index / len(time_str_sub), 0.75)
            color = base_color * (1 - alpha_f) + np.array([1, 1, 1]) * alpha_f

            if reverse_index == 0:
                plot_yaw_triangle, plot_box_edge, plot_vehicle_id = (True, True, True)
            else:
                plot_yaw_triangle, plot_box_edge, plot_vehicle_id = (False, False, False)

            self._plot_single_vehicle(
                key=car_track_id_int,
                x=agent.center.x,
                y=agent.center.y,
                yaw=agent.center.heading,
                veh_shape=shape_info,
                color=color,
                plot_yaw_triangle=plot_yaw_triangle,
                plot_box_edge=plot_box_edge,
                plot_vehicle_id=plot_vehicle_id,
            )

    def _plot_single_vehicle(
        self,
        key: str,
        x: float,
        y: float,
        yaw: float,
        veh_shape: dict,
        color: None,
        plot_yaw_triangle: bool = False,
        plot_box_edge: bool = False,
        plot_vehicle_id: bool = False,
    ):
        """plot single obstacle vehicle.定位点处增加箭头表示航向"""

        x_A3 = x - veh_shape["locationPoint2Rear"] * np.cos(yaw) + 0.5 * veh_shape["width"] * np.sin(yaw)
        y_A3 = y - veh_shape["locationPoint2Rear"] * np.sin(yaw) - 0.5 * veh_shape["width"] * np.cos(yaw)
        width_x = veh_shape["length"]
        height_y = veh_shape["width"]

        rect = patches.Rectangle(
            xy=(x_A3, y_A3),
            width=width_x,
            height=height_y,
            angle=yaw / np.pi * 180,
            facecolor=color,
            fill=True,
            edgecolor="black" if plot_box_edge else None,
            linewidth=self.linewidths["vehicle_box"] if plot_box_edge else 0.0,
            zorder=self.zorders["veh_box"],
        )
        self.axveh.add_patch(rect)

        # 绘制三角形，三角形尖部指向航向
        if plot_yaw_triangle:
            triangle_length = height_y  # 三角形的边长
            triangle_height = triangle_length * np.sqrt(3) / 2  # 三角形的高
            top_point = (x + triangle_height * np.cos(yaw), y + triangle_height * np.sin(yaw))
            left_point = (x - triangle_length * np.sin(yaw) / 2, y + triangle_length * np.cos(yaw) / 2)
            right_point = (x + triangle_length * np.sin(yaw) / 2, y - triangle_length * np.cos(yaw) / 2)
            triangle_points = [top_point, left_point, right_point]
            self.axveh.add_patch(
                patches.Polygon(triangle_points, closed=True, color=self.color_map["veh_yaw_triangle"], zorder=self.zorders["veh_yaw_triangle"])
            )

        if plot_vehicle_id:
            self.axveh.annotate(key, (x + 2.0, y + 2.0), fontsize=self.fontsize["veh_id"], zorder=self.zorders["veh_id"])

    def _plot_ego_vehicle(self, scenario: AbstractScenario, simulation_history: SimulationHistory, index: int):
        """利用 matplotlib 和 patches 绘制小汽车,以 x 轴为行驶方向
        定位点处增加箭头表示航向
        """
        x = simulation_history.extract_ego_state[index].rear_axle.x
        y = simulation_history.extract_ego_state[index].rear_axle.y
        yaw = simulation_history.extract_ego_state[index].rear_axle.heading
        veh_shape = self.scenario.ego_vehicle_parameters.shape

        x_A3 = x - veh_shape["locationPoint2Rear"] * np.cos(yaw) + 0.5 * veh_shape["width"] * np.sin(yaw)
        y_A3 = y - veh_shape["locationPoint2Rear"] * np.sin(yaw) - 0.5 * veh_shape["width"] * np.cos(yaw)
        width_x = veh_shape["length"]
        height_y = veh_shape["width"]

        self.axveh.add_patch(
            patches.Rectangle(
                xy=(x_A3, y_A3),
                width=width_x,
                height=height_y,
                angle=yaw / np.pi * 180,
                color=self.color_map["ego_box"],
                fill=True,
                zorder=self.zorders["veh_box"],
            )
        )

        # 绘制三角形，三角形尖部指向航向
        triangle_length = height_y  # 三角形的边长
        triangle_height = triangle_length * np.sqrt(3) / 2  # 三角形的高

        top_point = (x + triangle_height * np.cos(yaw), y + triangle_height * np.sin(yaw))
        left_point = (x - triangle_length * np.sin(yaw) / 2, y + triangle_length * np.cos(yaw) / 2)
        right_point = (x + triangle_length * np.sin(yaw) / 2, y - triangle_length * np.cos(yaw) / 2)

        triangle_points = [top_point, left_point, right_point]

        self.axveh.add_patch(
            patches.Polygon(
                triangle_points,
                closed=True,
                color=self.color_map["ego_yaw_triangle"],
                zorder=self.zorders["veh_yaw_triangle"],
            )
        )

    def _plot_warning_signal(self, scenario: AbstractScenario, c="red"):
        """绘制主车碰撞时的提醒标志"""
        for key, values in observation["vehicle_info"].items():
            if key == "ego":
                x, y = [float(values[i]) for i in ["x", "y"]]
                self.axveh.scatter(x, y, s=60, c=c, alpha=1.0, marker=(8, 1, 30), zorder=4)

    def _plot_static_obstacle(self, scenario: AbstractScenario):
        for obstacle in observation["static_obstacles"]:
            obstacle_id = obstacle["type"] + "-" + str(obstacle["id"])
            obstacle_vertices = obstacle["vertices"]

            # 计算障碍物的几何中心
            center_x = sum(vertex[0] for vertex in obstacle_vertices) / len(obstacle_vertices)
            center_y = sum(vertex[1] for vertex in obstacle_vertices) / len(obstacle_vertices)

            # 创建多边形的顶点坐标
            polygon_vertices = [(x, y) for x, y in obstacle_vertices]

            # 创建Polygon对象，并设置颜色为深蓝色
            obstacle_polygon = Polygon(
                polygon_vertices,
                closed=True,
                facecolor=self.color_map["static_obstacle"],
                edgecolor=self.color_map["static_obstacle"],
                alpha=0.6,
            )

            # 将Polygon对象添加到坐标轴上
            self.axveh.add_patch(obstacle_polygon)

            # 计算标注文本的偏移量，确保文本在障碍物内部
            offset = 2.0  # 可以根据障碍物的大小调整这个值
            text_x, text_y = center_x + offset, center_y + offset

            # 在障碍物的几何中心标注obstacle_id
            self.axveh.text(center_x, center_y, obstacle_id, color="black", ha="center", va="center", fontsize=self.fontsize["static_obstacles_id"])

    def _plot_start_pose_and_end_pose(self, scenario: AbstractScenario):
        """绘制开始和目标的车辆pose"""
        start_ego_info = scenario.scenario_info.test_setting["start_ego_info"]
        arrow_length = scenario.scenario_info.test_setting["start_ego_info"]["shape"]["length"] * 1.3
        arrow_width = scenario.scenario_info.test_setting["start_ego_info"]["shape"]["width"]

        self._plot_start_pose(start_ego_info=start_ego_info, arrow_length=arrow_length, arrow_width=arrow_width)

    def _plot_start_pose(self, start_ego_info, arrow_length, arrow_width):
        """开始位置的绘制（点+箭头）"""
        start_arrow = patches.FancyArrow(
            start_ego_info["x"],
            start_ego_info["y"],
            arrow_length * np.cos(start_ego_info["yaw_rad"]),
            arrow_length * np.sin(start_ego_info["yaw_rad"]),
            width=arrow_width * 0.2,
            head_width=arrow_width * 0.6,
            head_length=arrow_width * 0.8,
            color=self.color_map["ego_start_pose"],
            zorder=self.zorders["ego_start_pose"],
        )

        # 创建一个圆作为开始箭头的原点
        start_arrow_origin = patches.Circle(
            (start_ego_info["x"], start_ego_info["y"]),  # 使用开始位置的坐标作为圆心
            radius=arrow_width * 0.3,  # 圆的半径
            facecolor=self.color_map["ego_start_pose"],  # 圆的填充颜色
            edgecolor=self.color_map["ego_start_pose"],  # 圆的边缘颜色
            zorder=self.zorders["ego_start_pose"],
        )
        self.axbg.add_patch(start_arrow)
        self.axbg.add_patch(start_arrow_origin)

    def _plot_end_pose_ego_goal_arrow(self, goal_pose, arrow_length, arrow_width):
        """目标位置的绘制,箭头"""
        ego_goal_pose = patches.FancyArrow(
            goal_pose[0],
            goal_pose[1],
            arrow_length * np.cos(goal_pose[2]),
            arrow_length * np.sin(goal_pose[2]),
            width=arrow_width * 0.2,
            head_width=arrow_width * 0.6,
            head_length=arrow_width * 0.8,
            color=self.color_map["ego_goal_pose"],
            zorder=self.zorders["ego_goal_pose"],
        )
        # 创建一个圆作为箭头的原点
        goal_arrow_origin = patches.Circle(
            (goal_pose[0], goal_pose[1]),  # 圆心坐标
            radius=arrow_width * 0.3,  # 圆的半径
            facecolor=self.color_map["ego_goal_pose"],  # 圆的填充颜色
            edgecolor=self.color_map["ego_goal_pose"],  # 圆的边缘颜色
            zorder=self.zorders["ego_goal_pose"],
        )
        self.axbg.add_patch(ego_goal_pose)
        self.axbg.add_patch(goal_arrow_origin)

    def _plot_goal_box(self, scenario: AbstractScenario):
        """绘制该场景预设目标区域."""
        if self.x_target and self.y_target:
            x, y = self.x_target, self.y_target
            codes_box = [MatplotlibPath.MOVETO] + [MatplotlibPath.LINETO] * 3 + [MatplotlibPath.CLOSEPOLY]
            vertices_box = [
                (self.x_target[0], self.y_target[0]),
                (self.x_target[1], self.y_target[1]),
                (self.x_target[2], self.y_target[2]),
                (self.x_target[3], self.y_target[3]),
                (0, 0),
            ]  # 4 points of polygon
            path_box = MatplotlibPath(vertices_box, codes_box)  # 定义对应Path
            # 创建一个PathPatch对象，用于在图表上绘制这个Path。这里只设置了边颜色，没有设置填充颜色，所以只绘制边
            # facecolor 参数用来设置填充颜色，由于没有被设置或者注释掉了，所以多边形内部不会有填充
            pathpatch_box = patches.PathPatch(
                path_box,
                edgecolor=self.color_map["ego_goal"],
                facecolor="none",
                linewidth=self.linewidths["ego_goal_box"],
                zorder=self.zorders["ego_goal"],
            )
            # 将PathPatch对象添加到背景轴（axbg）上，这样它就会被绘制出来
            # pathpatch_box = patches.PathPatch(path_box, facecolor="tomato", edgecolor="orangered", zorder=self.zorders['ego_goal'])

            self.axbg.add_patch(pathpatch_box)

    def _plot_hdmaps(self, scenario: AbstractScenario) -> None:
        if not scenario.map_api:
            return

        layer_names = ["road", "intersection", "loading_area", "unloading_area"]
        if self.flag_plot_road_block:
            layer_names += ["road_block"]

        self._plot_hdmaps_bitmap_patch(scenario=scenario, ax=self.axbg, zorder=self.zorders["bitmap"])
        self._plot_hdmaps_semantic_map_patch(tgsc_map_explorer=scenario.map_api.semantic_map.explorer, layer_names=layer_names, ax=self.axbg)

    # scenario.map_api.semantic_map.explorer
    def _plot_hdmaps_semantic_map_patch(
        self,
        tgsc_map_explorer: TgScenesMapExplorer = None,
        box_coords: Tuple[float, float, float, float] = (0, 0, 1, 1),
        layer_names: List[str] = None,
        ax: Axes = None,
    ) -> None:
        for layer_name in layer_names:  # 渲染各图层
            tgsc_map_explorer._render_layer(
                ax=ax,
                layer_name=layer_name,
                alpha=self.alphas["map_polygon_layer"],
                zorder=self.zorders["semantic_map_polygon"],
                linewidth=self.linewidths["map_polygon_layer"],
            )

        if self.flag_plot_refpaths:
            #  渲染intersection的reference path,lane
            tgsc_map_explorer.render_connector_path_centerlines(
                ax=ax,
                alpha=self.alphas["map_refpath"],
                resampling_interval=self.resampling_interval_connector_path,
                zorder=self.zorders["semantic_map_refpath"],
                linewidth=self.linewidths["map_refpath"],
            )
            #  渲染road的reference path,lane
            tgsc_map_explorer.render_base_path_centerlines(
                ax=ax,
                alpha=self.alphas["map_refpath"],
                resampling_interval=self.resampling_interval_base_path,
                zorder=self.zorders["semantic_map_refpath"],
                linewidth=self.linewidths["map_refpath"],
            )

        if self.flag_plot_borderline:
            tgsc_map_explorer.render_edge_borderlines(
                ax=ax,
                alpha=self.alphas["map_borderline"],
                resampling_interval=20,
                zorder=self.zorders["semantic_map_borderline"],
                linewidth=self.linewidths["map_refpath"],
            )
            tgsc_map_explorer.render_inner_borderlines(
                ax=ax,
                alpha=self.alphas["map_borderline"],
                resampling_interval=20,
                zorder=self.zorders["semantic_map_borderline"],
                linewidth=self.linewidths["map_refpath"],
            )

    def _plot_hdmaps_bitmap_patch(self, scenario: AbstractScenario, ax: Axes = None, zorder: int = 2) -> None:
        if self.bitmap_type == "bitmap_mask":
            bitmap = scenario.map_api.raster_bit_map
            bitmap.render_mask_map_using_image_ndarray_local(ax, window_size=5, gray_flag=False, zorder=zorder)  #!mask图降采样绘图
        elif self.bitmap_type == "bitmap_rgb":
            bitmap = scenario.map_api.raster_bit_map
            bitmap.render_rgb_map(ax, zorder=zorder)
        else:
            raise Exception("###Exception### 非法的 bitmap type:%s" % self.bitmap_type)  # 自定义异常
