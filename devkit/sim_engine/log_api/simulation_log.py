from __future__ import annotations
import logging
import lzma  # 用于支持 LZMA 格式的压缩和解压缩
import pickle  # Python 内置的序列化库
from dataclasses import dataclass  # 用于简化类的定义和数据存储
from pathlib import Path  # 用于文件路径管理
from typing import Any  # 通用类型
import msgpack  # 高效的二进制序列化库

from devkit.scenario_builder.abstract_scenario import AbstractScenario
from devkit.sim_engine.history.simulation_history import SimulationHistory
from devkit.sim_engine.planning.planner.abstract_planner import AbstractPlanner

logger = logging.getLogger(__name__)


@dataclass
class SimulationLog:
    """Simulation log.
    这是一个数据类(dataclass)，用于存储仿真日志的相关信息
    """

    # 需要记录的数据放到这里;
    file_path: Path
    scenario: AbstractScenario
    planner: AbstractPlanner
    simulation_history: SimulationHistory

    def _dump_to_pickle(self) -> None:
        """Dump file into compressed pickle."""
        with lzma.open(self.file_path, "wb", preset=0) as f:  # 压缩后写入文件
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)  # 使用最高协议将对象序列化为二进制

    def _dump_to_msgpack(self) -> None:
        """Dump file into compressed msgpack"""
        # Serialize to a pickle object
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)  # 先使用 pickle 序列化为二进制
        with lzma.open(self.file_path, "wb", preset=0) as f:  # 压缩后保存到文件
            f.write(msgpack.packb(pickle_object))  # 再用 msgpack 对序列化后的二进制进行打包

    def save_to_file(self) -> None:
        """Dump simulation log into file."""
        serialization_type = self.simulation_log_type(self.file_path)  # 根据后缀判断要使用何种序列化格式
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if serialization_type == "pickle":
            self._dump_to_pickle()  # 若是 pickle，就调用 pickle 序列化方法
        elif serialization_type == "msgpack":
            self._dump_to_msgpack()  # 若是 msgpack，就调用 msgpack 序列化方法
        else:
            raise ValueError(f"Unknown option: {serialization_type}")  # 若类型都不匹配，则抛出错误

        logger.info(f"#log# Saved simulation log files to {self.file_path}")
        # 记录log文件位置到本地文件
        from devkit.common.utils.record_log_info import record_log_info
        record_log_info(log_file_path=self.file_path)
        

    @staticmethod
    def simulation_log_type(file_path: Path) -> str:
        """
        Deduce the simulation log type.
        :param file_path: File path.
        :return: one from ["msgpack", "pickle", "json"].
        """
        msg_pack = file_path.suffixes == [".msgpack", ".xz"]
        msg_pickle = file_path.suffixes == [".pkl", ".xz"]
        number_of_available_types = int(msg_pack) + int(msg_pickle)

        # We can handle only conclusive serialization type
        if number_of_available_types != 1:
            raise RuntimeError(f"Inconclusive file type: {file_path}!")

        if msg_pickle:
            return "pickle"
        elif msg_pack:
            return "msgpack"
        else:
            raise RuntimeError("Unknown condition!")

    @classmethod
    def load_data(cls, file_path: Path) -> Any:
        """Load simulation log."""
        simulation_log_type = SimulationLog.simulation_log_type(file_path=file_path)
        if simulation_log_type == "msgpack":
            with lzma.open(str(file_path), "rb") as f:
                data = msgpack.unpackb(f.read())  # 先使用 msgpack 解包二进制数据
                data = pickle.loads(data)  # 再用 pickle 还原为 Python 对象
        elif simulation_log_type == "pickle":
            with lzma.open(str(file_path), "rb") as f:
                data = pickle.load(f)  # 直接用 pickle.load 读取并还原
        else:
            raise ValueError(f"Unknown serialization type: {simulation_log_type}!")

        return data  # 返回还原后的对象
