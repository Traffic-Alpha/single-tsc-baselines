'''
@Author: WANG Maonan
@Description: 可插拔的 observation 构造函数集合
所有 obs 函数签名统一: (lane_dynamic_features_seq, static_lane_features, lane_order, num_phases) -> np.ndarray
lane_dynamic_features_seq: List[Dict]，每个元素为一个子步的 lane 动态特征
默认使用最后一帧（seq[-1]）；如需利用完整序列，可自行实现
每个 obs 函数需要提供 get_space(num_phases, num_lanes) 方法返回 observation space
'''
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, List


# 归一化参数
MAX_VEHICLES = 50.0
MAX_SPEED = 15.0       # m/s (约 54 km/h)
MAX_WAITING_TIME = 300.0  # 秒
SPEED_THRESHOLD = 1.0  # m/s，低于此速度视为等待


def lane_aggregate_obs(
    lane_dynamic_features_seq: List[Dict[str, Any]],
    static_lane_features: Dict[str, Any],
    lane_order: List[str],
    num_phases: int,
) -> np.ndarray:
    """将 lane 的 cell 级别特征聚合为 lane 级别特征

    使用最后一帧快照（seq[-1]）。

    每个 lane 的特征 (6 + num_phases 维):
    [0] waiting_vehicles / 50.0
    [1] moving_vehicles / 50.0
    [2] avg_speed / 15.0
    [3] avg_waiting_time / 300.0
    [4] avg_occupancy
    [5] is_green (0/1)
    [6:6+num_phases] phase_binding (multi-hot)

    Returns:
        feature_array: shape (num_lanes, 6 + num_phases)
    """
    lane_dynamic_features = lane_dynamic_features_seq[-1]
    feature_dim = 6 + num_phases
    num_lanes = len(lane_order)
    feature_array = np.zeros((num_lanes, feature_dim), dtype=np.float32)

    for lane_idx, lane_id in enumerate(lane_order):
        if lane_id not in lane_dynamic_features:
            continue

        cells = lane_dynamic_features[lane_id]

        total_vehicles = 0
        waiting_vehicles = 0
        moving_vehicles = 0
        total_speed = 0.0
        total_waiting_time = 0.0
        total_occupancy = 0.0
        is_green = 0
        num_cells = len(cells)

        for cell in cells:
            vehicle_count = cell['vehicle_count']
            avg_speed = cell['avg_speed']

            total_vehicles += vehicle_count
            total_speed += avg_speed * vehicle_count
            total_waiting_time += cell['avg_waiting_time'] * vehicle_count
            total_occupancy += cell['occupancy']

            if avg_speed < SPEED_THRESHOLD and vehicle_count > 0:
                waiting_vehicles += vehicle_count
            elif avg_speed >= SPEED_THRESHOLD and vehicle_count > 0:
                moving_vehicles += vehicle_count

            if cell.get('is_passable', 0) > 0:
                is_green = 1

        avg_speed_lane = total_speed / total_vehicles if total_vehicles > 0 else 0.0
        avg_waiting_time_lane = total_waiting_time / total_vehicles if total_vehicles > 0 else 0.0
        avg_occupancy = total_occupancy / num_cells if num_cells > 0 else 0.0

        feature_array[lane_idx, 0] = min(waiting_vehicles / MAX_VEHICLES, 1.0)
        feature_array[lane_idx, 1] = min(moving_vehicles / MAX_VEHICLES, 1.0)
        feature_array[lane_idx, 2] = min(avg_speed_lane / MAX_SPEED, 1.0)
        feature_array[lane_idx, 3] = min(avg_waiting_time_lane / MAX_WAITING_TIME, 1.0)
        feature_array[lane_idx, 4] = min(avg_occupancy, 1.0)
        feature_array[lane_idx, 5] = is_green

        if lane_id in static_lane_features:
            phase_binding = static_lane_features[lane_id]['phase_binding']
            feature_array[lane_idx, 6:6+len(phase_binding)] = phase_binding

    return feature_array


def lane_aggregate_obs_space(num_phases: int, num_lanes: int = 20) -> spaces.Box:
    """返回 lane_aggregate_obs 对应的 observation space"""
    return spaces.Box(
        low=0.0, high=1.0,
        shape=(num_lanes, 6 + num_phases),
        dtype=np.float32
    )
