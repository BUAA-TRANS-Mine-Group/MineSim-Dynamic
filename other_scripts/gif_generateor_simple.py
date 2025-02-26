# Python Standard Library Imports
import os
import time
from pathlib import Path as PathlibPath
from os.path import isfile, join

# Third-Party Imports
import numpy as np
from PIL import Image
import imageio

# !Note：该脚本复制到'MineSim-Dynamic/outputs/outputs_figure/gif_cache_png/IDMPlanner/jiangtong_intersection_9_3_2/log_2025-02-25_11-53-12_945fa6d3-number-10'使用
# Constant
IMAGE_DIR_NAME = "dapai_intersection_1_3_4"
IMAGE_DIR_NAME = "dapai_intersection_2_2_5"
# IMAGE_DIR_NAME = "dapai_intersection_2_3_6"
# IMAGE_DIR_NAME = "jiangtong_intersection_9_3_2"
# IMAGE_DIR_NAME = "8_3_4_565"

method = "frenet"
# method = "JSSP"
BASE_DIR = PathlibPath(__file__).parent
PNGS_DIR = BASE_DIR / method / IMAGE_DIR_NAME
GIF_FILE_PATH = BASE_DIR / f"{method}-{IMAGE_DIR_NAME}.gif"
IMAGE_TEMPLATE = f"{{}}.png"


def generate_gif_from_images():
    # 记录gif生成时间,用于评估效率,没有特殊用途
    tic = time.time()

    # 创建一个图像列表,用于存储所有图像
    image_list = []

    # 获取目录下PNG图像文件的列表
    if not os.path.exists(PNGS_DIR):
        print(f"路径 {PNGS_DIR} 不存在!")

    files_in_directory = [file for file in PNGS_DIR.iterdir() if file.is_file() and file.name.endswith(".png")]

    # 找到文件夹中存在的最大编号
    max_image_index = -1
    for file in files_in_directory:
        try:
            index = int(file.stem.split("_")[-1])
            max_image_index = max(max_image_index, index)
        except ValueError:
            pass
    # 初始化变量来跟踪图像尺寸和颜色模式
    expected_dimensions = None
    expected_mode = None

    # for file in files_in_directory:
    #     img = Image.open(file)
    #     if expected_dimensions is None:
    #         expected_dimensions = img.size
    #         expected_mode = img.mode
    #     # elif img.size == expected_dimensions and img.mode == expected_mode:
    #     #     image_list.append(img)
    #     # else:
    #     #     print(f"Skipping image {file.name} due to size or mode mismatch.")
    #     elif img.size != expected_dimensions or img.mode != expected_mode:
    #         print(f"Warning: Image {file.name} has different dimensions or color mode. Adjusting...")
    #         # Adjust image to match the expected dimensions and color mode
    #         img = img.resize(expected_dimensions)
    #         # if img.mode != expected_mode:
    #         #     img = img.convert(expected_mode)
    #     image_list.append(img)

    # 循环读取图片并添加到列表中
    for i in range(max_image_index + 1):
        image_path = PNGS_DIR / IMAGE_TEMPLATE.format(i)
        if image_path.exists():
            img = Image.open(image_path)
            if expected_dimensions is None:
                expected_dimensions = img.size
                expected_mode = img.mode
            elif img.size != expected_dimensions or img.mode != expected_mode:
                print(f"Warning: Image {file.name} has different dimensions or color mode. Adjusting...")
                # Adjust image to match the expected dimensions and color mode
                img = img.resize(expected_dimensions)
                # if img.mode != expected_mode:
                #     img = img.convert(expected_mode)
            image_list.append(img)

    # 保存GIF动画
    if image_list:
        if len(image_list) > 100:
            step = int(len(image_list) / 100) + 1
            image_list = image_list[::step]
        image_list[0].save(GIF_FILE_PATH, save_all=True, append_images=image_list[1:], duration=100, loop=0)
        print(f"GIF文件保存在 {GIF_FILE_PATH}")

    print(f"###log### GIF文件保存在 {GIF_FILE_PATH}")
    toc = time.time()
    print(f"###log### GIF generated in {toc - tic:.2f} seconds.")


if __name__ == "__main__":
    generate_gif_from_images()
