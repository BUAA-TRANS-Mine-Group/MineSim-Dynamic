import sys, os

dir_current_file = os.path.dirname(__file__)
dir_parent_1 = os.path.dirname(dir_current_file)
dir_parent_2 = os.path.dirname(dir_parent_1)
dir_parent_3 = os.path.dirname(dir_parent_2)


def get_config_directory(dir_name="dir_maps", specified_dir=None):
    """Specify absolute directory
    默认为 minesim 的上级目录下: minesim/devkit/../..
    """
    if specified_dir is None:  # default dir
        if dir_name == "dir_datasets":
            return os.path.abspath(os.path.join(dir_parent_3, "datasets"))
        elif dir_name == "dir_maps":
            return os.path.abspath(os.path.join(dir_parent_3, "maps"))
        elif dir_name == "dir_inputs":
            return os.path.abspath(os.path.join(dir_parent_2, "inputs"))
        elif dir_name == "dir_outputs":
            return os.path.abspath(os.path.join(dir_parent_2, "outputs"))
        elif dir_name == "dir_outputs_log":
            return os.path.abspath(os.path.join(dir_parent_2, "outputs", "outputs_log"))
        elif dir_name == "dir_outputs_figure":
            return os.path.abspath(os.path.join(dir_parent_2, "outputs", "outputs_figure"))
        else:
            raise ValueError("dir_name Error.")
    if not os.path.exists(specified_dir):
        raise ValueError("dir_path Error!")
    else:
        return specified_dir


SimConfig = {
    "BASE_DEVKIT_PATH": "/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-vscode/MineSim-Dynamic/devkit",
    "other_conf": {
        "is_visualize": True,
        "is_save_fig": True,
        "is_record_sim_log": True,
        "is_skip_exist_scene": False,  # 是否跳过outputs中已有记录的场景(可以避免程序异常中断后重新测试之前已经完成的场景)
    },
    "directory_conf": {
        "dir_datasets": get_config_directory(
            dir_name="dir_datasets",
            specified_dir="/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-vscode/datasets",
        ),
        "dir_maps": get_config_directory(
            dir_name="dir_maps",
            specified_dir="/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-vscode/datasets/maps",
        ),
        "dir_inputs": get_config_directory(dir_name="dir_inputs"),
        "dir_outputs": get_config_directory(dir_name="dir_outputs"),
        "dir_outputs_log": get_config_directory(dir_name="dir_outputs_log"),
        "dir_outputs_figure": get_config_directory(dir_name="dir_outputs_figure"),
    },
    "skip_exist_scene": True,
}


def validate_sim_engine_conf(SimConfig) -> None:
    # todo
    pass
