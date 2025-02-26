# Python library
import sys
import os
import time

# Third-party library
import matplotlib
import matplotlib.pyplot as plt
import argparse

from typing import Dict, List, Tuple, Optional, Union, Any
from matplotlib.patches import Rectangle, Arrow
from matplotlib.axes import Axes
from PIL import Image
import io


# Local library
from devkit.sim_engine.map_manager.map_expansion.bit_map import BitMap
from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMap
from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMapExplorer


class GlobalMapVisualizer:

    def __init__(self, dataroot: str = "onsite_mine/datasets", location: str = "jiangxi_jiangtong"):
        self.flag_visilize = True  # 是\否进行可视化绘图
        self.flag_hdmaps_visilized = False  # hdmaps已经被绘图(仅绘制一次即可)标志位
        self.flag_visilize_ref_path = True  # 是否可视化参考路径
        self.flag_visilize_prediction = True  # 是否可视化轨迹预测结果
        self.flag_save_fig_whitout_show = False  # 绘图保存,但是不显示

        self.dataroot = dataroot
        self.location = location

        self.hdmaps = {}
        self.hdmaps = self._parse_hdmaps()
        self.color_map = dict(
            road="#b2df8a",
            road_block="#2A2A2A",
            intersection="#fb9a99",
            loading_area="#6a3d9a",
            unloading_area="#7e772e",
            connector_path="#1f78b4",
            base_path="#1f78b4",
            dubins_pose="#064b7a",
            highlight="#FF0000",
        )

    def _parse_hdmaps(self):
        """解析高清地图文件.
        功能:
        1)获取可行驶区域的mask图信息.
        2)加载路网结构.
        """

        # 获取mask图信息并确定最大方形区域
        self.hdmaps["image_mask"] = BitMap(self.dataroot, self.location, "bitmap_mask")
        self.x_min = self.hdmaps["image_mask"].bitmap_info["bitmap_mask_PNG"]["UTM_info"]["point_southwest"][0]
        self.x_max = self.hdmaps["image_mask"].bitmap_info["bitmap_mask_PNG"]["UTM_info"]["point_northeast"][0]
        self.y_min = self.hdmaps["image_mask"].bitmap_info["bitmap_mask_PNG"]["UTM_info"]["point_southwest"][1]
        self.y_max = self.hdmaps["image_mask"].bitmap_info["bitmap_mask_PNG"]["UTM_info"]["point_northeast"][1]

        # 加载路网信息
        self.hdmaps["tgsc_map"] = TgScenesMap(self.dataroot, self.location)
        self.hdmaps = {"image_mask": self.hdmaps["image_mask"], "tgsc_map": self.hdmaps["tgsc_map"]}

        return self.hdmaps

    def init(self, args, dpi: int = 200):
        """绘图

        Args:
            args (_type_): 终端输入的参数
            dpi (int, optional): （dots per inch）是一个用于描述图像或打印输出分辨率的单位,每英寸的像素点数量; Defaults to 200.;取值范围设定为[100,1000];dpi越小图像越精细;
        """
        # 初始化画布
        if self.flag_visilize:
            if self.flag_save_fig_whitout_show == True:
                plt.ioff()
            else:
                plt.ion()
            pixel_size = self.hdmaps["image_mask"].bitmap_info["bitmap_mask_PNG"]["canvas_edge_pixel"]  # width,height
            # width_normalize, height_normalize = width / max(width, height), height / max(width, height)

            # 系统没有足够的内存来分配给图像缓冲区。这个问题通常发生在尝试创建非常大的图像时，比如当图像尺寸或分辨率特别高时。
            fig_size_auto = [pixel_size[0] / dpi, pixel_size[1] / dpi]  # 以英寸为单位的画布大小

            # figsize_calculate = [width_normalize* fig_size ,   height_normalize* fig_size ]  # 获取比例关系
            self.fig = plt.figure(figsize=fig_size_auto)
            self.axbg = self.fig.add_subplot()  # 多个x轴,共用y轴
            # 设置xy坐标刻度的字体大小
            labelsize_auto = int(10000 / dpi)
            self.axbg.tick_params(axis="x", labelsize=labelsize_auto)  # 设置x轴刻度字体大小
            self.axbg.tick_params(axis="y", labelsize=labelsize_auto)  # 设置y轴刻度字体大小
            self.axbg.set_xlabel("x[m]")
            self.axbg.set_ylabel("y[m]")

            # 添加网格，并设置网格线的粗细
            def dpi_to_linewidth_auto(dpi):
                # DPI 的范围 [100, 1000] 映射到 suto 的范围 [5, 1]
                dpi_min, dpi_max = 100, 1000
                suto_min, suto_max = 5, 1

                # 使用线性插值来计算 suto 值
                suto = suto_min + (dpi - dpi_min) * (suto_max - suto_min) / (dpi_max - dpi_min)
                return suto

            linewidth_auto = dpi_to_linewidth_auto(dpi=dpi)
            plt.grid(True, linewidth=linewidth_auto)  # 将网格线粗细设置为2

        else:
            plt.ioff()
            return

        self.flag_hdmaps_visilized = False  # 本次绘制地图
        if self.hdmaps:
            self._plot_hdmaps(args)

    def _plot_hdmaps(self, args, padding: float = 20):
        # 1) 绘制地图个图层; 2)标记 各元素的token;
        if not self.hdmaps:
            return
        # plot mask图,plot 道路片段、交叉口多边形划分;
        my_patch = (self.x_min - padding, self.y_min - padding, self.x_max + padding, self.y_max + padding)  #  (x_min,y_min,x_max,y_max).
        # layer_names = ['intersection','road']
        layer_names = ["road", "intersection", "loading_area", "unloading_area", "road_block"]
        self._plot_hdmaps_render_map_patch(
            tgsc_map_explorer=TgScenesMapExplorer(self.hdmaps["tgsc_map"], color_map=self.color_map),
            box_coords=my_patch,
            layer_names=layer_names,
            alpha=0.7,
            bitmap=self.hdmaps["image_mask"],
            ax=self.axbg,
            args=args,
        )

    def _plot_hdmaps_render_map_patch(
        self,
        args,
        tgsc_map_explorer: TgScenesMapExplorer = None,
        box_coords: Tuple[float, float, float, float] = (0, 0, 1, 1),
        layer_names: List[str] = None,
        alpha: float = 0.5,
        bitmap: Optional[BitMap] = None,
        ax: Axes = None,
    ) -> None:
        """渲染一个矩形图框,指定矩形框的xy坐标范围.By default renders all layers."""

        if bitmap is not None:
            if bitmap.bitmap_type == "bitmap_mask":
                bitmap.render_mask_map(ax, gray_flag=False)

        for layer_name in layer_names:  # 渲染各图层
            tgsc_map_explorer._render_layer(ax, layer_name, alpha)

        #  渲染intersection的reference path,lane
        if self.flag_visilize_ref_path == True:
            tgsc_map_explorer.render_connector_path_centerlines(ax, alpha, resampling_interval=20)

        #  渲染road的reference path,lane
        if self.flag_visilize_ref_path == True:
            tgsc_map_explorer.render_base_path_centerlines(ax, alpha, resampling_interval=5)

        # 渲染特定dubins、path、polygon
        # 初始化
        dubins_pose = []
        polygon = []
        reference_path = []
        # 透明度
        alpha = 1.0
        if args.dubins_pose:
            for dubins_pose_token in args.dubins_pose:
                dubins_pose.append("dubins_pose-" + str(dubins_pose_token))
            tgsc_map_explorer.render_dubins_poses(self.axbg, alpha, dubinspose_tokens=dubins_pose, highlight=True)

        if args.polygon:
            for polygon_token in args.polygon:
                polygon.append("polygon-" + str(polygon_token))
            tgsc_map_explorer._render_polygon_tokens_layer(self.axbg, alpha, tokens=polygon)

        if args.reference_path:
            for reference_path_token in args.reference_path:
                reference_path.append("path-" + str(reference_path_token))
            tgsc_map_explorer._render_reference_path_centerlines(self.axbg, alpha, resampling_rate=0.05, tokens=reference_path)


def save_fig(filename):
    # 保存图像
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    img.save(filename)
    buf.close()


def view_global_map_demo(args):
    """示例,查看and保存全局地图."""
    dir_datasets = os.path.abspath(os.path.join(dir_parent_3, "datasets"))  # 指定数据集根目录
    location = "jiangxi_jiangtong"  #  指定矿区,索引对应地图
    # location = "guangdong_dapai"  #  指定矿区,索引对应地图
    # 1 解析地图
    global_map_visualizer = GlobalMapVisualizer(dataroot=dir_datasets, location=location)

    # 2 初始化 图幅,应该根据全局地图的尺寸自动初始化一个长宽;
    global_map_visualizer.init(args, dpi=100)

    # plt.show()
    plt.subplots_adjust()

    if 1:  # 绘图保存

        def check_dir(target_dir):
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

        OUTPUT_DIR = os.path.join(dir_parent_3, "cache_figures")
        check_dir(OUTPUT_DIR)
        fig_name = location + "_semantic_map_add_tokens"
        fig_path = os.path.join(OUTPUT_DIR, "{fig_name}.png".format(fig_name=fig_name))
        save_fig(fig_path)
        print(f"##log## 地图绘制完成.dir={fig_path}")

    print("##log## 地图绘制完成.")


if __name__ == "__main__":
    # Parse and view the global map
    parser = argparse.ArgumentParser()
    parser.add_argument("--polygon", metavar="N", type=int, nargs="*", help="List of integers for polygon", required=False)
    parser.add_argument("--dubins_pose", metavar="N", type=int, nargs="*", help="List of integers for dubins-pose", required=False)
    parser.add_argument("--reference_path", metavar="N", type=int, nargs="*", help="List of integers for path", required=False)
    args = parser.parse_args()
    if args.polygon:
        print("polygon:", args.polygon)
    if args.dubins_pose:
        print("dubins_pose:", args.dubins_pose)
    if args.reference_path:
        print("reference_path:", args.reference_path)
    view_global_map_demo(args)
