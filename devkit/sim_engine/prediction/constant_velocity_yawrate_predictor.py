import numpy as np
import common.tool as tool


# class ConstantVelocityYawRatePredictor:
class ConstantVelocityYawRatePredictor:
    """恒速度假设,constant Velocity and constant yaw rate
    基于车辆运动学模型,利用线速度和角速度从当前位置推算出一条轨迹
    extrapolates a curved line from current position using linear and angular velocity

    其它同族方法:
    - 恒角度假设;
    - 恒加速度假设;
    """

    def __init__(self, interval=0.1, max_t=18.0, time_horizon=5.0) -> None:
        """默认构造 获取每个场景后,进行初始化
        默认时间间隔0.1 , 还有一些时间间隔是0.04 ;
        """
        self.time_horizon = time_horizon
        self.max_t = max_t
        self.dt = interval

        self.now_t: float = 0.0
        self.now_t_str: str = "0.0"
        self.now_t_step: int = 0

        self.predicted_trajs = {}

    def update_now_time(self, observation):
        self.now_t = observation["test_setting"]["t"]
        self.now_t_str = str(round(self.now_t, 1))
        self.now_t_step = int(round(self.now_t / self.dt, 1))

    def predict_veh_traj(self, observation: dict, id_veh: int, vehi_traj_GT: dict):
        """predict id_veh traj."""
        veh_info = vehi_traj_GT[self.now_t_str]
        predicted_trajs = self.predict_veh_traj_by_CACYR(veh_info=veh_info, vehi_traj_GT=vehi_traj_GT)

        return predicted_trajs

    def get_vhe_history_state(self, observation: dict, id_veh: int, vehi_traj_GT: dict, t_step_num: int = 10):
        # todo
        pass

    def predict_veh_traj_by_CACYR_old(self, veh_info: dict, vehi_traj_GT: dict):
        """有bug，模型不太对"""
        x = veh_info["x"]
        y = veh_info["y"]
        v = veh_info["v"]
        yaw = veh_info["yaw"]
        # acc = veh_info["a"]
        acc = 0.0

        t_last1_str = str(round(self.now_t - self.dt, 2))
        t_pre2_str = str(round(self.now_t - 2 * self.dt, 2))
        if (t_last1_str in vehi_traj_GT) and (t_pre2_str in vehi_traj_GT):
            yaw_rate = (vehi_traj_GT[self.now_t_str]["yaw_rate"] + vehi_traj_GT[t_last1_str]["yaw_rate"] + vehi_traj_GT[t_pre2_str]["yaw_rate"]) / 3
        else:
            yaw_rate = vehi_traj_GT[self.now_t_str]["yaw_rate"]

        t_list = [t for t in np.arange(self.now_t + self.dt, min(self.now_t + self.time_horizon + self.dt, self.max_t), self.dt)]
        t_str_list = [str(round(t, 2)) for t in t_list]

        self.predicted_trajs[self.now_t_str] = {"x": round(x, 2), "y": round(y, 2), "v": round(v, 2), "a": round(acc, 2), "yaw": round(yaw, 3)}
        for index, t_str in enumerate(t_str_list):
            t = t_list[index]
            x += self.dt * (v * np.cos(yaw) + acc * np.cos(yaw) * 0.5 * self.dt + yaw_rate * v * np.sin(yaw) * 0.5 * self.dt)  # 确定yaw定义
            y += self.dt * (v * np.sin(yaw) + acc * np.sin(yaw) * 0.5 * self.dt + yaw_rate * v * np.cos(yaw) * 0.5 * self.dt)
            yaw += self.dt * yaw_rate
            v += self.dt * acc

            self.predicted_trajs[t_str] = {"x": round(x, 2), "y": round(y, 2), "v": round(v, 2), "a": round(acc, 2), "yaw": round(yaw, 3)}

        return self.predicted_trajs

    def predict_veh_traj_by_CACYR(self, veh_info: dict, vehi_traj_GT: dict):
        """简单运动学模型"""
        x = veh_info["x"]
        y = veh_info["y"]
        v = veh_info["v"]
        yaw = veh_info["yaw"]
        # acc = veh_info["a"]
        acc = 0.0

        t_last1_str = str(round(self.now_t - self.dt, 2))
        t_pre2_str = str(round(self.now_t - 2 * self.dt, 2))
        if (t_last1_str in vehi_traj_GT) and (t_pre2_str in vehi_traj_GT):
            yaw_rate = (vehi_traj_GT[self.now_t_str]["yaw_rate"] + vehi_traj_GT[t_last1_str]["yaw_rate"] + vehi_traj_GT[t_pre2_str]["yaw_rate"]) / 3
        else:
            yaw_rate = vehi_traj_GT[self.now_t_str]["yaw_rate"]

        t_list = [t for t in np.arange(self.now_t + self.dt, min(self.now_t + self.time_horizon + self.dt, self.max_t), self.dt)]
        t_str_list = [str(round(t, 2)) for t in t_list]

        self.predicted_trajs[self.now_t_str] = {"x": round(x, 2), "y": round(y, 2), "v": round(v, 2), "a": round(acc, 2), "yaw": round(yaw, 3)}
        for index, t_str in enumerate(t_str_list):
            t = t_list[index]
            x += v * np.cos(yaw) * self.dt
            y += v * np.sin(yaw) * self.dt
            yaw += yaw_rate * self.dt
            v += acc * self.dt
            self.predicted_trajs[t_str] = {"x": round(x, 2), "y": round(y, 2), "v": round(v, 2), "a": round(acc, 2), "yaw": round(yaw, 3)}

        return self.predicted_trajs

    def predict_veh_traj_by_CACY(self, veh_info: dict, now_t: float):
        # todo 无 yaw rate ,仅使用 yaw
        pass
