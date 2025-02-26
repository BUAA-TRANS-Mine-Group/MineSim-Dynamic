# Python library
import copy
import time
import sys
import os
from typing import Dict, List, Union, Tuple

# Third-party library
import matplotlib.pyplot as plt
import numpy as np


# Local library
from devkit.sim_engine.prediction.constant_velocity_yawrate_predictor import ConstantVelocityYawRatePredictor


class Predictor:
    """周围障碍物车辆 轨迹预测器
    背景车轨迹信息 vehicle_traj
    """

    def __init__(self, time_horizon: float = 5.0) -> None:
        """默认构造 Predictor
        time_horizon : 预测时域,默认为5s
        """
        self.time_horizon = time_horizon
        self.max_t = None
        self.dt = 0.1

        self.now_t: float = 0.0
        self.now_t_str: str = "0.0"
        self.now_t_step: int = 0

        self.all_veh_traj_predict = {}
        self.all_veh_traj_GT = {}

        self.veh_ids_predict = []

    def update_now_time(self, observation):
        self.now_t = observation["test_setting"]["t"]
        self.now_t_str = str(round(self.now_t, 1))
        self.now_t_step = int(round(self.now_t / self.dt, 1))

    def predict_all_vehicle_traj(self, observation):
        """不同车辆使用不同类型预测方法."""
        self.vehi_traj_predict.clear()
        for id_veh in observation["vehicle_info"].keys():
            if id_veh != "ego":
                vehi_traj_predict = self.CVCYR_predictor.predict_veh_traj(
                    observation=observation, id_veh=id_veh, vehi_traj_GT=self.vehi_traj_GT[id_veh]
                )
                self.vehi_traj_predict[id_veh] = copy.deepcopy(vehi_traj_predict)
        pass

    def get_all_vehicle_GT(self, traj_GT: dict, observation: dict):
        for id_veh in observation["vehicle_info"].keys():
            if id_veh != "ego":
                self.all_veh_traj_GT[id_veh] = traj_GT[id_veh]

        return self.all_veh_traj_GT

    def predict(self, traj_GT: dict, observation: dict, no_predict: bool = False):
        """目标车辆轨迹预测."""
        self.update_now_time(observation=observation)

        if no_predict:
            return self.get_all_vehicle_GT(traj_GT=traj_GT, observation=observation)

        # ######################### 不同的背景车可以使用不同的预测器 #########################
        # 不同车辆使用不同的轨迹预测方法
        for id_veh in self.vehi_traj_GT.keys():
            traj_future_ = self.CVCYR_predictor.predict(traj_GT[id_veh], self.now_t)  # 调用单个车辆预测器
            for key, value in traj_future_.items():  # 遍历所有键值对
                self.vehi_traj_predict[id_veh][key] = value
            self.vehi_traj_predict[id_veh][-1] = {}
            self.vehi_traj_predict[id_veh]["shape"] = traj_GT[id_veh]["shape"]

        return self.all_veh_traj_predict


if __name__ == "__main__":
    import time
