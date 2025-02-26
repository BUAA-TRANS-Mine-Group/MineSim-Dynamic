
# 参数配置：路径跟踪控制器
paras_control = {
    "controller_type": "pure_pursuit",  # only one
    "pure_pursuit": {
        # ld = kv * self.ego_state.v + ld0
        # "kv_ld0": [0.25, 2.8],
        "kv_ld0": [0.35, 4.8],
        # 轴距计算系数 # l = self.ego_vehicle.l / 1.7
        # "coff_wheel_base": 2.0,
        "coff_wheel_base": 1.7,
    },
    # 参数 kv= 0.6, ld0= 5.0, 系数= 1.7,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.5, ld0= 4.8, 系数= 1.5,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.5, ld0= 4.8, 系数= 1.7,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.5, ld0= 5.8, 系数= 1.7,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.1, ld0= 5.8, 系数= 1.7,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.25, ld0= 5.8, 系数= 1.7,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.25, ld0= 5.8, 系数= 2.0,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.25, ld0= 2.8, 系数= 2.0,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.25, ld0= 2.8, 系数= 2.0,find_lookahead_point_by_curve_distance,跟踪参考路径
    # 参数 kv= 0.25, ld0= 2.8, 系数= 2.0,find_lookahead_point_by_curve_distance,跟踪规划路径
    # 参数 kv= 0.25, ld0= 3.8, 系数= 2.0,find_lookahead_point_by_linear_distance,跟踪参考路径
    # 参数 kv= 0.25, ld0= 3.8, 系数= 2.0,find_lookahead_point_by_curve_distance,跟踪规划路径
    # 参数 kv= 0.45, ld0= 1.8, 系数= 2.0,find_lookahead_point_by_curve_distance,跟踪规划路径
    # 参数 kv= 0.25, ld0= 3.8, 系数= 2.0,find_lookahead_point_by_curve_distance,跟踪参考路径
}


# 参数配置：局部路径规划器
paras_planner = {
    "is_visualize_plan": "True",
    "planner_type": "idm",  #!NOTE: "frenet";"JSSP"
    "planner_frenet": {
        "num_width": 5,
        "num_speed": 5,
        "num_t": 7,  # 5.0 6.0 7.0 8.0
        "min_t": 5.0,
        "max_t": 8.0,
        # "cost_wights": [1.0, 5.4/65, 10.0, 0.9, 1.0, 3.0, 4.0, 0.5],
        # "cost_wights": [0.15, 55.4/65, 3.0, 0.9, 1.0, 3.0, 4.0, 0.5],
        "cost_wights": [1.0, 0.0, 39.4, 0.2, 0.1, 0.8, 0.4, 10.0],
        # w_time ;w_distance;w_laneoffset;w_lateral_accel;w_lateral_jerk;w_longitudinal_a;w_longitudinal_jerk;w_longitudinal_v
        "max_distance": 92.0,
    },
    "planner_JSSP": {
        "num_width": 3,
        "num_jerk": 9,
        "plan_horizon": 5.0,
        "cost_wights": [0.0, 0.2, 0.4, 0.2, 0.1, 0.8, 0.4, 0.5],
        # w_time ;w_distance;w_laneoffset;w_lateral_accel;w_lateral_jerk;w_longitudinal_a;w_longitudinal_jerk;w_longitudinal_v
        # ! w_time权重没有
        "max_distance": 86.0,
        "traj_num_max": 50,
    },
}
