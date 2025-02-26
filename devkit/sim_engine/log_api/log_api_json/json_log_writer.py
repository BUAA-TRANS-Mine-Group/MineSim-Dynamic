import json
import gzip
from datetime import datetime
from threading import Timer


class JsonLogWriter:
    def __init__(
        self,
        file_path: str = None,
        float_precision: int = 2,
        compress: bool = False,
        batch_size: int = 10,
        flush_interval: int = 10,
    ):
        """
        初始化日志写入器类。

        参数:
            file_path (str, optional): 日志文件存储路径。默认为 None。
            float_precision (int, optional): 浮点数精确到小数点后多少位。默认为 2。
            compress (bool, optional): 是否压缩日志文件。默认为 False。
            batch_size (int, optional): 触发写入操作前缓存的日志条目数量。默认为 10。
            flush_interval (int, optional): 自动写入磁盘的时间间隔（秒）。默认为 10。
        """
        self.file_path = file_path  # 文件路径
        self.file_path_single_msg = None
        self.float_precision = float_precision  # 浮点数精度
        self.compress = compress  # 是否压缩文件
        self.batch_size = batch_size  # 批处理大小
        self.flush_interval = flush_interval  # 刷新间隔
        self.file_mode = "at" if not compress else "at"  # 文件打开模式
        self.open_func = open if not compress else gzip.open  # 打开文件的函数
        self.log_entries = []  # 日志条目缓存
        self.timer = None  # 定时器

    def _format_floats(self, obj):
        """递归格式化字典或列表中的浮点数。"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self._format_floats(v)
        elif isinstance(obj, list):
            return [self._format_floats(item) for item in obj]
        elif isinstance(obj, float):
            return round(obj, self.float_precision)
        return obj

    def log_single_message(self, data):
        log_entry = self._format_floats(data.copy())
        log_entry["timestamp"] = datetime.now().isoformat()
        with self.open_func(self.file_path_single_msg, self.file_mode) as file:
            if self.compress:
                file.write((json.dumps(log_entry) + "\n").encode('utf-8'))
            else:
                json.dump(log_entry, file)
                file.write("\n")
        pass

    def log(self, data):
        """记录日志数据，加上时间戳，处理浮点数精度，并管理日志缓存。"""
        log_entry = self._format_floats(data.copy())
        log_entry["timestamp"] = datetime.now().isoformat()  # 实时添加当前时间戳
        self.log_entries.append(log_entry)
        if len(self.log_entries) >= self.batch_size:
            self.flush()
        if not self.timer:
            self.timer = Timer(self.flush_interval, self.flush)
            self.timer.start()

    def flush(self):
        """将缓存的日志条目刷新到磁盘。"""
        if self.log_entries:
            with self.open_func(self.file_path, self.file_mode) as file:
                for entry in self.log_entries:
                    if self.compress:
                        file.write((json.dumps(entry) + "\n").encode('utf-8'))
                    else:
                        json.dump(entry, file)
                        file.write("\n")
            self.log_entries = []  # 清空缓存
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def __del__(self):
        """确保在对象销毁前刷新缓存到磁盘。"""
        self.flush()


if __name__ == "__main__":
    # 示例使用
    timestamp = datetime.now().isoformat()  # 当前时间戳
    file_name = "autodrive_log_" + timestamp + ".json.gz"
    file_path = file_name
    logger = JsonLogWriter(file_path, compress=True, batch_size=5, flush_interval=30)
    for i in range(20):
        logger.log({"event": "obstacle_detected", "details": {"size": "large", "distance": 15.125 + i}})
