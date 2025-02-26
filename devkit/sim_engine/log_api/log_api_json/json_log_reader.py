import json
import gzip
from datetime import datetime


class JsonLogReader:
    def __init__(self, file_path, compress=False):
        self.file_path = file_path
        self.open_func = gzip.open if compress else open

    def read_logs(self):
        logs = []
        with self.open_func(self.file_path, "rt") as file:
            for line in file:
                logs.append(json.loads(line))
        return logs


if __name__ == "__main__":
    # 示例
    file = "/home/czf/project_czf/20240401-静态障碍物避障轨迹规划场景库/project_static_obs/outputs/dapai_staticObstacleScene_1_1_2024-05-10T17:45:18.872212.json.gz"
    # file = "/home/czf/project_czf/20240401-静态障碍物避障轨迹规划场景库/project_static_obs/outputs/dapai_staticObstacleScene_1_1_2024-05-10T17:45:18.872212.json"
    reader = JsonLogReader(file, compress=True)
    all_logs = reader.read_logs()
