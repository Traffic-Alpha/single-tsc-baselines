'''
@Author: WANG Maonan
@Description: 可插拔的 reward 函数集合
所有 reward 函数签名统一: (lane_dynamic_features_seq, static_lane_features) -> float
lane_dynamic_features_seq: List[Dict]，每个元素为一个子步的 lane 动态特征
默认使用最后一帧（seq[-1]）；如需利用完整序列，可自行实现
'''
from typing import Dict, Any, List


# 速度阈值，低于此速度视为排队等待
SPEED_THRESHOLD = 1.0  # m/s


def pressure_reward(lane_dynamic_features_seq: List[Dict[str, Any]], static_lane_features: Dict[str, Any]) -> float:
    """基于 pressure 的奖励

    Pressure = 进口道排队车辆数 - 出口道排队车辆数
    奖励为负的 pressure（pressure 越小，奖励越高）
    排队车辆定义：速度 < 1.0 m/s
    使用最后一帧快照计算。
    """
    lane_dynamic_features = lane_dynamic_features_seq[-1]
    incoming_waiting = 0
    outgoing_waiting = 0

    for lane_id, cells in lane_dynamic_features.items():
        if lane_id not in static_lane_features:
            continue

        io_type = static_lane_features[lane_id]['io_type']

        waiting = 0
        for cell in cells:
            if cell['avg_speed'] < SPEED_THRESHOLD and cell['vehicle_count'] > 0:
                waiting += cell['vehicle_count']

        if io_type[0] == 1:  # Incoming lane
            incoming_waiting += waiting
        elif io_type[1] == 1:  # Outgoing lane
            outgoing_waiting += waiting

    return float(-(incoming_waiting - outgoing_waiting))


def queue_length_reward(lane_dynamic_features_seq: List[Dict[str, Any]], static_lane_features: Dict[str, Any]) -> float:
    """基于排队长度的奖励

    奖励为负的进口道总排队车辆数
    使用最后一帧快照计算。
    """
    lane_dynamic_features = lane_dynamic_features_seq[-1]
    total_queue = 0

    for lane_id, cells in lane_dynamic_features.items():
        if lane_id not in static_lane_features:
            continue

        io_type = static_lane_features[lane_id]['io_type']
        if io_type[0] != 1:  # 只统计 incoming lane
            continue

        for cell in cells:
            if cell['avg_speed'] < SPEED_THRESHOLD and cell['vehicle_count'] > 0:
                total_queue += cell['vehicle_count']

    return float(-total_queue)


def waiting_time_reward(lane_dynamic_features_seq: List[Dict[str, Any]], static_lane_features: Dict[str, Any]) -> float:
    """基于等待时间的奖励

    奖励为负的进口道车辆平均等待时间
    使用最后一帧快照计算。
    """
    lane_dynamic_features = lane_dynamic_features_seq[-1]
    total_waiting_time = 0.0
    total_vehicles = 0

    for lane_id, cells in lane_dynamic_features.items():
        if lane_id not in static_lane_features:
            continue

        io_type = static_lane_features[lane_id]['io_type']
        if io_type[0] != 1:  # 只统计 incoming lane
            continue

        for cell in cells:
            count = cell['vehicle_count']
            if count > 0:
                total_waiting_time += cell['avg_waiting_time'] * count
                total_vehicles += count

    if total_vehicles == 0:
        return 0.0

    return float(-total_waiting_time / total_vehicles)
