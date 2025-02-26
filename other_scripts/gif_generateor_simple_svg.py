# Python Standard Library Imports
import os
from pathlib import Path as PathlibPath
import re
import time
import io

# Third-Party Imports
from cairosvg import svg2png
from PIL import Image


# 日志记录类
class Logger:
    def __init__(self):
        self.prefix = "[GIF Creator] "

    def info(self, msg):
        print(f"{self.prefix}{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

    def error(self, msg):
        print(f"{self.prefix}ERROR! {msg}")


LOGGER = Logger()

# 全局常量
BASE_DIR_SVG = PathlibPath("/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/MineSim-Dynamic-Dev/outputs/outputs_figure/gif_cache_svg")
IMAGE_DIR_NAME = "FrenetOptimalPlanner/jiangtong_intersection_9_3_2/log_2025-02-25_16-31-12_5d5b295d-number-11"


SVG_SOURCE_DIR = BASE_DIR_SVG / IMAGE_DIR_NAME
GIF_FILE_PATH = BASE_DIR_SVG / f"{IMAGE_DIR_NAME.replace('/', '_')}.gif"
IMAGE_TEMPLATE = f"{{}}.svg"


def generate_gif_from_images():
    LOGGER.info("开始生成GIF动画")
    tic = time.time()
    image_list = []

    # 目录验证
    if not SVG_SOURCE_DIR.exists():
        LOGGER.error(f"源目录不存在: {SVG_SOURCE_DIR}")
        return

    # 文件筛选与排序
    pattern = re.compile(r"^(\d+)\.svg$")
    svg_files = []
    for file in SVG_SOURCE_DIR.glob("*.svg"):
        match = pattern.match(file.name)
        if match:
            svg_files.append((int(match.group(1)), file))

    if not svg_files:
        LOGGER.error("未找到有效SVG文件")
        return

    sorted_files = sorted(svg_files, key=lambda x: x[0])
    max_index = sorted_files[-1][0]
    LOGGER.info(f"找到 {len(sorted_files)} 个文件，最大索引: {max_index}")

    # 图像处理参数
    target_size = None
    color_mode = "RGB"

    for idx in range(max_index + 1):
        image_path = SVG_SOURCE_DIR / IMAGE_TEMPLATE.format(idx)
        if not image_path.exists():
            LOGGER.warning(f"跳过缺失文件: {image_path}")
            continue

        try:
            # 修正：强制转换为字符串路径
            png_bytes = svg2png(url=str(image_path))
            with Image.open(io.BytesIO(png_bytes)) as img:
                # 自动适应尺寸
                if target_size is None:
                    target_size = img.size
                else:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                # 统一颜色模式
                if img.mode != color_mode:
                    img = img.convert(color_mode)
                image_list.append(img)
        except Exception as e:
            LOGGER.error(f"处理文件 {image_path} 失败: {e}")
            continue  # 跳过无效图像

    # 最终校验
    if not image_list:
        LOGGER.error("无有效图像可生成GIF")
        return

    # 图像一致性检查
    first = image_list[0]
    for img in image_list[1:]:
        if img.size != first.size or img.mode != first.mode:
            LOGGER.warning("图像尺寸/模式不一致，已调整")
            img = img.resize(first.size, Image.Resampling.LANCZOS)
            img = img.convert(first.mode)
            image_list.append(img)

    # 保存GIF
    try:
        with image_list[0] as first_img:
            first_img.save(GIF_FILE_PATH, save_all=True, append_images=image_list[1:], duration=100, loop=0, optimize=True)
        LOGGER.info(f"GIF已生成: {GIF_FILE_PATH}")
    except Exception as e:
        LOGGER.error(f"保存GIF失败: {e}")

    toc = time.time()
    LOGGER.info(f"总耗时: {toc - tic:.2f} 秒")


if __name__ == "__main__":
    generate_gif_from_images()
