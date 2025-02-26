import numpy as np


class ScenarioInfo:
    """用于存储回放测试中用以控制背景车辆的所有数据
    背景车轨迹信息 vehicle_traj
    vehicle_traj = {
        "vehicle_id_0":{
            "shape":{
                "vehicle_type":"PickupTruck",
                "length":5.416,
                "width":1.947,
                "height":1.886,
                "locationPoint2Head":2.708,
                "locationPoint2Rear":2.708
            },
            "t_0":{
                "x":0,
                "y":0,
                "v_mps":0,
                "acc_mpss":0,
                "yaw_rad":0,
                "yawrate_radps":0,
            },
            "t_1":{...},
            ...
        },
        "vehicle_id_1":{...},
        ...
    }
    主车轨迹信息,只包含当前帧信息
    ego_info = {
        "shape":{
                    "vehicle_type":"MineTruck_NTE200",
                    "length":13.4,
                    "width":6.7,
                    "height":6.9,
                    "min_turn_radius":14.2,
                    "locationPoint2Head":2.708,
                    "locationPoint2Rear":2.708
                },
        "x":0,
        "y":0,
        "v_mps":0,
        "acc_mpss":0,
        "yaw_rad":0,
        "yawrate_radps":0
    }
    地图相关信息,具体介绍地图解析工作的教程 markdown 文档待编写
    road_info = {}
    测试环境相关信息 test_setting
    test_setting = {
        "t":,
        "dt":,
        "max_t",
        "goal":{
            "x":[-1,-1,-1,-1],
            "y":[-1,-1,-1,-1]
        },
        "end":,
        "scenario_type":,
        "scenario_name":,
        "map_type":,
        "start_ego_info"
    }

    """

    def __init__(self):
        self.vehicle_traj = {}
        self.ego_info = {
            "shape": {
                "vehicle_type": "MineTruck_NTE200",
                "length": 13.4,
                "width": 6.7,
                "height": 6.9,
                "min_turn_radius": 14.2,
                "locationPoint2Head": 2.708,
                "locationPoint2Rear": 2.708,
            },
            "x": 0,
            "y": 0,
            "v_mps": 0,
            "acc_mpss": 0,
            "yaw_rad": 0,
            "yawrate_radps": 0,
        }
        self.hdmaps = {}
        self.test_setting = {
            "t": 0,
            "dt": 0.1,
            "max_t": 10,
            "goal": {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]},  # goal box:4 points [x1,x2,x3,x4],[y1,xy,y3,y4]
            "end": -1,
            "scenario_name": None,
            "scenario_type": None,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "start_ego_info": None,
        }  # 同 Observation.test_setting

    def add_vehicle_shape(self, id, t, traj_info=None):
        """
        该函数实现向vehicle_trajectiry中添加背景车轨迹信息的功-增加车辆形状
        """
        # a = 1
        if id == "ego":
            # self._add_vehicle_ego(x,y,v,a,yaw,length,width)
            self._add_vehicle_ego(traj_info)
        else:
            if id not in self.vehicle_traj.keys():
                self.vehicle_traj[id] = {}
                self.vehicle_traj[id]["shape"] = {}
            if t not in self.vehicle_traj[id].keys():
                self.vehicle_traj[id][t] = {}
            self.vehicle_traj[id]["shape"] = traj_info["VehicleShapeInfo"]

    def add_vehicle_traj(self, id, t, traj_info=None):
        """
        该函数实现向vehicle_trajectiry中添加背景车轨迹信息的功能-增加车辆状态、轨迹
        """
        self.vehicle_traj[id]["shape"] = traj_info["VehicleShapeInfo"]

        for index, _ in enumerate(traj_info["states"]["x"]):
            t = traj_info["StartTimeInScene"] + index * self.test_setting["dt"]
            str_t = str(round(float(t), 2))
            if str_t not in self.vehicle_traj[id].keys():
                self.vehicle_traj[id][str_t] = {}
            for key, value in zip(
                ["x", "y", "yaw_rad", "v_mps", "yawrate_radps", "acc_mpss"],
                [
                    traj_info["states"]["x"][index][0],
                    traj_info["states"]["y"][index][0],
                    traj_info["states"]["yaw_rad"][index][0],
                    traj_info["states"]["v_mps"][index][0],
                    traj_info["states"]["yawrate_radps"][index][0],
                    traj_info["states"]["acc_mpss"][index][0],
                ],
            ):
                if value is not None:
                    self.vehicle_traj[id][str_t][key] = np.around(value, 5)  # 保留5位小数

    def add_ego_info(self, ego_info):
        """
        该函数实现向test_setting中添加自车初始状态信息
        """
        self.ego_info["x"] = ego_info["states"]["x"]
        self.ego_info["y"] = ego_info["states"]["y"]
        self.ego_info["yaw_rad"] = ego_info["states"]["yaw_rad"]
        self.ego_info["v_mps"] = ego_info["states"]["v_mps"]
        self.ego_info["yawrate_radps"] = ego_info["states"]["yawrate_radps"]
        self.ego_info["acc_mpss"] = ego_info["states"]["acc_mpss"]
        self.ego_info["shape"] = ego_info["VehicleShapeInfo"]

    def add_settings(self, scenario_name=None, scenario_type=None, dt=None, max_t=None, goal_x=None, goal_y=None):
        """
        该函数实现向test_setting中添加测试环境相关信息
        """
        for key, value in zip(["scenario_name", "scenario_type", "dt", "max_t"], [scenario_name, scenario_type, dt, max_t]):
            if value is not None:
                self.test_setting[key] = value
        for key, value in zip(["x", "y"], [goal_x, goal_y]):
            if value is not None:
                self.test_setting["goal"][key] = value

    def _init_vehicle_ego_info(self, one_scenario=None):
        """
        该函数实现向ego_info中增加主车信息的功能
        注意:ego_info中只含有主车当前帧的信息
        """
        self.ego_info["shape"] = one_scenario["ego_info"]["VehicleShapeInfo"]
        self.ego_info["x"] = one_scenario["ego_info"]["start_states"]["x"]
        self.ego_info["y"] = one_scenario["ego_info"]["start_states"]["y"]
        self.ego_info["yaw_rad"] = one_scenario["ego_info"]["start_states"]["yaw_rad"]
        self.ego_info["v_mps"] = one_scenario["ego_info"]["start_states"]["v_mps"]
        self.ego_info["acc_mpss"] = one_scenario["ego_info"]["start_states"]["acc_mpss"]
        self.ego_info["yawrate_radps"] = one_scenario["ego_info"]["start_states"]["yawrate_radps"]

        self.test_setting["start_ego_info"] = self.ego_info

    def _get_dt_maxt(self, one_scenario=None):
        """
        该函数实现得到最大仿真时长阈值以及采样率的功能 ,步长,最大时间,最大最小xy范围
        """
        self.test_setting["max_t"] = one_scenario["max_t"]
        self.test_setting["dt"] = one_scenario["dt"]
        self.test_setting["x_min"] = one_scenario["x_min"]
        self.test_setting["x_max"] = one_scenario["x_max"]
        self.test_setting["y_min"] = one_scenario["y_min"]
        self.test_setting["y_max"] = one_scenario["y_max"]
