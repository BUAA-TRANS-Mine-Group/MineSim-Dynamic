import os
import csv
from datetime import datetime
from devkit.configuration.sim_engine_conf import SimConfig as sim_config


def record_log_info(log_file_path):
    """
    记录log文件位置到本地文件
    在dir_outputs目录下创建 log_file_list.csv;
    第 1 列为序号 number,
    第 2 列为日期 2025-01-06 20:28:10,695,
    第 3 列为 log 文件路径;
    """
    # 获取输出目录
    dir_outputs = sim_config["directory_conf"]["dir_outputs"]

    # 确保输出目录存在
    os.makedirs(dir_outputs, exist_ok=True)

    # log_file_list.csv 的路径
    csv_file = os.path.join(dir_outputs, "log_file_list.csv")

    # 获取当前时间，格式为 "2025-01-06 20:28:10,695"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

    # 检查 csv 文件是否存在
    file_exists = os.path.isfile(csv_file)

    # 如果文件存在，读取最后一行的第一列 (序号)
    if file_exists:
        with open(csv_file, mode="r", encoding="utf-8") as fr:
            reader = csv.reader(fr)
            # 跳过表头
            next(reader, None)
            last_number = 0
            for row in reader:
                if row:
                    # row[0] 为序号
                    try:
                        last_number = int(row[0])
                    except ValueError:
                        # 如果序号非数字（数据异常），可做相应处理，这里简单地当作 0 处理
                        last_number = 0
            # 新条目的序号在最后一行序号上 +1
            serial_number = last_number + 1
    else:
        # 如果 csv 文件不存在，则从 1 开始记录
        serial_number = 1

    # 以追加模式打开 csv 文件
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(["number", "date", "log_file_path"])
        # 写入日志信息
        writer.writerow([serial_number, current_time, log_file_path])
