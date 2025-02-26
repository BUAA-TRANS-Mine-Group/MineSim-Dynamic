# Python library
import sys
import os
import time
import io

# Third-party library
import matplotlib
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional, Union, Any
from matplotlib.patches import Rectangle, Arrow
from matplotlib.axes import Axes
from PIL import Image


# Local library
from devkit.sim_engine.map_manager.map_expansion.bit_map import BitMap
from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMap
from devkit.sim_engine.map_manager.map_expansion.map_api import TgScenesMapExplorer


class GlobalMapVisualizer:

    def __init__(self, dataroot: str = "onsite_mine/datasets", location: str = "jiangxi_jiangtong"):
        self.flag_visilize_ref_path = True  # 是否可视化参考路径

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

    def init(self, dpi: int, local_range_x: Tuple[float, float], local_range_y: Tuple[float, float]):
        """绘图
        dpi:(dots per inch)是一个用于描述图像或打印输出分辨率的单位,每英寸的像素点数量;取值范围设定为[100,1000];dpi越小图像越精细;
        """
        if local_range_x[1] < local_range_x[0]:
            Exception("local_range_x error")
        if local_range_y[1] < local_range_y[0]:
            Exception("local_range_y error")

        x_range = local_range_x[1] - local_range_x[0]
        y_range = local_range_y[1] - local_range_y[0]
        # 根据坐标确定最大的方形框
        max_range = max(x_range, y_range)
        x_center, y_center = (local_range_x[0] + local_range_x[1]) / 2, (local_range_y[0] + local_range_y[1]) / 2
        utm_local_range = (x_center - max_range / 2, y_center - max_range / 2, x_center + max_range / 2, y_center + max_range / 2)
        self.hdmaps["image_mask"].load_bitmap_using_utm_local_range(utm_local_range, 1.0, 1.0)

        # 初始化画布
        plt.ion()
        self.fig = plt.figure(figsize=[20, 20])
        self.axbg = self.fig.add_subplot()  # 多个x轴,共用y轴

        # 设置xy坐标刻度的字体大小
        labelsize_auto = int(10000 / dpi)
        # self.axbg.tick_params(axis="x", labelsize=labelsize_auto)  # 设置x轴刻度字体大小
        # self.axbg.tick_params(axis="y", labelsize=labelsize_auto)  # 设置y轴刻度字体大小
        self.axbg.set_xlabel("x[m]")
        self.axbg.set_ylabel("y[m]")
        self.axbg.set_xlim(xmin=local_range_x[0], xmax=local_range_x[1])
        self.axbg.set_ylim(ymin=local_range_y[0], ymax=local_range_y[1])

        # linewidth_auto = dpi_to_linewidth_auto(dpi=dpi)
        linewidth_auto = 1.2
        plt.grid(True, linewidth=linewidth_auto)  # 将网格线粗细设置为2
        plt.show()

        if self.hdmaps:
            my_patch = utm_local_range
            layer_names = ["road", "intersection", "loading_area", "unloading_area", "road_block"]
            self._plot_hdmaps_render_map_patch(
                tgsc_map_explorer=TgScenesMapExplorer(self.hdmaps["tgsc_map"]),
                box_coords=my_patch,
                layer_names=layer_names,
                alpha=0.7,
                bitmap=self.hdmaps["image_mask"],
                ax=self.axbg,
            )

    def _plot_hdmaps_render_map_patch(
        self,
        tgsc_map_explorer: TgScenesMapExplorer = None,
        box_coords: Tuple[float, float, float, float] = (0, 0, 1, 1),
        layer_names: List[str] = None,
        alpha: float = 0.5,
        render_egoposes_range: bool = False,
        render_legend: bool = False,
        bitmap: Optional[BitMap] = None,
        ax: Axes = None,
    ) -> None:
        """渲染一个矩形图框,指定矩形框的xy坐标范围.By default renders all layers."""
        x_min, y_min, x_max, y_max = box_coords
        local_width = x_max - x_min
        local_height = y_max - y_min

        if bitmap is not None:
            if bitmap.bitmap_type == "bitmap_mask":
                bitmap.render_mask_map_using_image_ndarray_local(ax, window_size=5, gray_flag=False)  #!mask图降采样绘图
            elif bitmap.bitmap_type == "bitmap_rgb":
                bitmap.render_rgb_map(ax)
            else:
                raise Exception("###Exception### 非法的 bitmap type:%s" % self.bitmap_type)  # 自定义异常

        for layer_name in layer_names:  # 渲染各图层
            tgsc_map_explorer._render_layer(ax, layer_name, alpha)

        if self.flag_visilize_ref_path == True:
            #  渲染intersection的reference path,lane
            tgsc_map_explorer.render_connector_path_centerlines(ax, alpha, resampling_interval=20)
            #  渲染road的reference path,lane
            tgsc_map_explorer.render_base_path_centerlines(ax, alpha, resampling_interval=5)


def save_fig(filename):
    # 保存图像
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    img.save(filename)
    buf.close()


def view_map_by_local_range(location: str, dpi: int, local_range_x: Tuple[float, float], local_range_y: Tuple[float, float]):
    """局部矩形区域的地图.

    Args:
        location (str): 指定矿区,索引对应地图,可选"guangdong_dapai" , "jiangxi_jiangtong"
        dpi (int): (dots per inch)是一个用于描述图像或打印输出分辨率的单位,每英寸的像素点数量;取值范围设定为[100,1000];dpi越小图像越精细;
        local_range_x (Tuple[float, float]): _description_
        local_range_y (Tuple[float, float]): _description_
    """

    dir_datasets = os.path.abspath(os.path.join(dir_parent_3, "datasets"))  # 指定数据集根目录
    # 1 解析地图
    global_map_visualizer = GlobalMapVisualizer(dataroot=dir_datasets, location=location)
    # 2 初始化 图幅 由dpi确定图片的绘制分辨率
    global_map_visualizer.init(dpi=dpi, local_range_x=local_range_x, local_range_y=local_range_y)

    plt.subplots_adjust()

    # aaaaaa = 1
    # pass  # temp
    if 0:  # 绘图保存

        def check_dir(target_dir):
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

        OUTPUT_DIR = os.path.join(dir_parent_3, "cache_figures")
        check_dir(OUTPUT_DIR)
        fig_name = location + "_semantic_map"
        fig_path = os.path.join(OUTPUT_DIR, "{fig_name}.png".format(fig_name=fig_name))
        save_fig(fig_path)
        print(f"##log## 地图绘制完成.dir={fig_path}")

    return global_map_visualizer


if __name__ == "__main__":
    # Parse and view the global map
    # location = "guangdong_dapai"  #  指定矿区,索引对应地图
    # location = "jiangxi_jiangtong"  #  指定矿区,索引对应地图
    local_range_x = (2000, 2250)
    local_range_y = (600, 1000)
    view_map_by_local_range(dpi=10, location="guangdong_dapai", local_range_x=local_range_x, local_range_y=local_range_y)
    # view_global_map_demo(dpi=100, location="jiangxi_jiangtong")
