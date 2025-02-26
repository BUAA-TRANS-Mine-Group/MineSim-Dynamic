# vehicle configuration parameters
vehicle_conf = {
    "ego_name": {
        "current_name": "AUTO_CONF",
        "names": ["XG90G", "NTE200", "AUTO_CONF"],
    },
    "XG90G": {
        "vehicle_type":"wide-body-dump-truck",
        "shape": {
            "length": 9.0,
            "width": 4.0,
            "locationPoint2Head": 6.5,
            "locationPoint2Rear": 2.5,
            "wheelbase": 5.3,
             "height":3.5,
        },
        "constraints": {
            "min_steering_angle": -1.8,  # rad
            "max_steering_angle": 1.8,
            "max_steering velocity": 0.3,  # rad/s
            "max_longitudinal velocity": 16.7,  # m/s
            "min_longitudinal_acceleration": -15.5,  # m/s^2
            "max_longitudinal_acceleration": 16.8,
            "max_centripetal_acceleration": 1.5,
            "max_lateral_acceleration": 1.2,
            "min_turning_radius": 5.44,  # m,R = L / sin(max_steering_angle)
        },
    },
    "NTE200": {
        "vehicle_type":"electric-drive-mining-truck",
        "shape": {
            "length": 13.0,
            "width": 6.7,
            "locationPoint2Head": 9.2,
            "locationPoint2Rear": 3.8,
            "wheelbase": 9.6,
            "height":6.9,
            
        },
        "constraints": {
            "min_steering_angle": -1.5,  # rad
            "max_steering_angle": 1.5,
            "max_steering velocity": 0.2,
            "max_longitudinal velocity": 12.5,
            "min_longitudinal_acceleration": -8.9,
            "max_longitudinal_acceleration": 10.2,
            "max_centripetal_acceleration": 1.5,
            "max_lateral_acceleration": 1.2,
            "min_turning_radius": 9.62,  # m
        },
    },
}
