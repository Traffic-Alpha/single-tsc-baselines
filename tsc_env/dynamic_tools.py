'''
@Author: WANG Maonan
@Date: 2026-01-11
@Description: 动态状态处理工具，将 lane 分成多个 cell，计算每个 cell 的动态信息
LastEditTime: 2026-04-14 23:00:49
'''
import numpy as np
from typing import Dict, List, Any


# Global value ranges for different metrics (for consistent visualization across time steps)
METRIC_VALUE_RANGES = {
    'occupancy': {
        'vmin': 0.0, 'vmax': 1.0,
        'description': 'Occupancy rate (0-1)', 'unit': ''
    },
    'vehicle_count': {
        'vmin': 0, 'vmax': 3,
        'description': 'Number of vehicles', 'unit': 'vehicles'
    },
    'avg_speed': {
        'vmin': 0.0, 'vmax': 15.0,
        'description': 'Average speed', 'unit': 'm/s'
    },
    'avg_waiting_time': {
        'vmin': 0.0, 'vmax': 120.0,
        'description': 'Average waiting time', 'unit': 's'
    },
    'avg_accumulated_waiting_time': {
        'vmin': 0.0, 'vmax': 300.0,
        'description': 'Average accumulated waiting time', 'unit': 's'
    },
    'distance_to_lane_start': {
        'vmin': 0.0, 'vmax': 100.0,
        'description': 'Distance to lane start', 'unit': 'm'
    },
    'is_passable': {
        'vmin': 0, 'vmax': 1,
        'description': 'Lane passability (0: red, 1: green)', 'unit': ''
    }
}


def get_metric_value_range(metric: str, custom_ranges: Dict[str, Dict] = None):
    """Get the recommended value range for a specific metric"""
    if custom_ranges and metric in custom_ranges:
        return custom_ranges[metric]['vmin'], custom_ranges[metric]['vmax']
    if metric in METRIC_VALUE_RANGES:
        return METRIC_VALUE_RANGES[metric]['vmin'], METRIC_VALUE_RANGES[metric]['vmax']
    return 0.0, 1.0


def update_metric_value_range(metric: str, vmin: float, vmax: float) -> None:
    """Update the global value range for a specific metric"""
    if metric in METRIC_VALUE_RANGES:
        METRIC_VALUE_RANGES[metric]['vmin'] = vmin
        METRIC_VALUE_RANGES[metric]['vmax'] = vmax
    else:
        METRIC_VALUE_RANGES[metric] = {
            'vmin': vmin, 'vmax': vmax,
            'description': metric, 'unit': ''
        }


def print_metric_value_ranges() -> None:
    """Print all metric value ranges for reference"""
    print("\n" + "="*70)
    print("Recommended Metric Value Ranges for Consistent Visualization")
    print("="*70)
    for metric, config in METRIC_VALUE_RANGES.items():
        print(f"\n{metric}:")
        print(f"  Range: [{config['vmin']}, {config['vmax']}] {config['unit']}")
        print(f"  Description: {config['description']}")
    print("="*70 + "\n")


class LaneCellManager:
    """管理 lane 的 cell 划分和动态信息计算

    每条 lane 根据固定的 cell 长度进行划分，不同长度的 lane 会有不同数量的 cell
    """

    def __init__(self,
        static_lane_features: Dict[str, Dict],
        cell_length: float = 10.0,
    ) -> None:
        self.static_lane_features = static_lane_features
        self.cell_length = cell_length
        self.lane_cells_info = {}
        self._initialize_lane_cells()

    def _initialize_lane_cells(self) -> None:
        """初始化每条 lane 的 cell 划分信息，结果存入 self.lane_cells_info。

        SUMO 中 lane_position 始终从 lane 起点（远离路口端）= 0 增大到 lane_length（路口端）。
        两类 lane 的 cell 索引均以 0 表示最靠近路口的 cell：

        Incoming lane（车辆驶入路口）:
            [远端]  cell[N-1] ... cell[1]  cell[0]  [路口]
            lane_pos: 0  ──────────────────────────>  L (车辆行驶方向)
            cell_boundaries 从大到小排列（降序），boundaries[i:i+2] 直接对应 cell[i]。
            cell[0] = [L-cell_length, L]（靠近路口，完整 cell），余数在 cell[N-1]（远端）。
            做法：从路口端向远端每隔 cell_length 取一个边界，保持降序。
            例：L=35, cell_length=10 → boundaries=[35, 25, 15, 5, 0]
                cell[0]=[25,35]=10m ✓   cell[3]=[0,5]=5m（余数）

        Outgoing lane（车辆驶出路口）:
            [路口]  cell[0]  cell[1] ...  cell[N-1]  [远端]
            lane_pos: 0  ──────────────────────────>  L (车辆行驶方向)
            cell_boundaries 从小到大排列（升序），boundaries[i:i+2] 直接对应 cell[i]。
            cell[0] = [0, cell_length]（靠近路口，完整 cell），余数在 cell[N-1]（远端）。
            做法：从路口端（lane_pos=0）向远端每隔 cell_length 取一个边界，直接升序。
            例：L=35, cell_length=10 → boundaries=[0, 10, 20, 30, 35]
                cell[0]=[0,10]=10m ✓    cell[3]=[30,35]=5m（余数）

        两类 lane 的 cell 0 均在路口侧，保证排队/刚离开路口的车辆聚集在低索引 cell
        """
        for lane_id, features in self.static_lane_features.items():
            lane_length = features['length'] * 100  # 反归一化得到实际长度（米）
            num_cells = int(np.ceil(lane_length / self.cell_length))

            io_type = features.get('io_type', [0, 0])
            is_incoming = (io_type[0] == 1)

            # 对 incoming lane 划分 cells：降序存储，boundaries[i:i+2] = cell[i] 的区间
            if is_incoming:
                cell_boundaries = []
                for i in range(num_cells + 1):
                    boundary = max(lane_length - i * self.cell_length, 0)
                    cell_boundaries.append(boundary)
                cell_boundaries = np.array(cell_boundaries)  # 已是降序，无需翻转
            else: # 对 outgoing lane 划分 cells
                cell_boundaries = []
                for i in range(num_cells + 1):
                    boundary = min(i * self.cell_length, lane_length)
                    cell_boundaries.append(boundary)
                cell_boundaries = np.array(cell_boundaries)

            self.lane_cells_info[lane_id] = {
                'length': lane_length,
                'num_cells': num_cells,
                'cell_boundaries': cell_boundaries,
                'cell_centers': (cell_boundaries[:-1] + cell_boundaries[1:]) / 2,
                'is_incoming': is_incoming,
            }

    def __get_vehicle_cell_index(self, lane_id: str, lane_position: float) -> int:
        """获取车辆所在的 cell 索引

        incoming lane 的 cell_boundaries 为降序，cell 0 在路口侧（高 lane_pos）：
            cell_index = num_cells - searchsorted(ascending_view, lane_pos, 'right')
        outgoing lane 的 cell_boundaries 为升序，cell 0 在路口侧（低 lane_pos）：
            cell_index = searchsorted(cell_boundaries, lane_pos, 'right') - 1
        """
        if lane_id not in self.lane_cells_info:
            return -1

        lane_info = self.lane_cells_info[lane_id]
        cell_boundaries = lane_info['cell_boundaries']
        num_cells = lane_info['num_cells']

        if lane_info['is_incoming']:
            # cell_boundaries 降序；cell[::-1] 得到升序视图（numpy view，无拷贝）
            cell_index = num_cells - np.searchsorted(cell_boundaries[::-1], lane_position, side='right')
        else:
            cell_index = np.searchsorted(cell_boundaries, lane_position, side='right') - 1

        cell_index = np.clip(cell_index, 0, num_cells - 1)
        return int(cell_index)

    def calculate_lane_dynamic_features(
        self,
        vehicles_state: Dict[str, Dict],
        current_phase_index: int = None
    ) -> Dict[str, List[Dict]]:
        """计算每条 lane 中每个 cell 的动态特征"""
        # 初始化
        lane_cell_vehicles = {}
        for lane_id, lane_info in self.lane_cells_info.items():
            num_cells = lane_info['num_cells']
            lane_cell_vehicles[lane_id] = [[] for _ in range(num_cells)]

        # 将车辆分配到对应的 cell
        for veh_id, veh_info in vehicles_state.items():
            lane_id = veh_info['lane_id']
            if lane_id not in self.lane_cells_info:
                continue
            lane_position = veh_info['lane_position']
            cell_index = self.__get_vehicle_cell_index(lane_id, lane_position)
            if cell_index >= 0:
                lane_cell_vehicles[lane_id][cell_index].append(veh_info)

        # 计算每个 cell 的统计特征
        lane_dynamic_features = {}

        for lane_id, cells_vehicles in lane_cell_vehicles.items():
            lane_features = []
            cell_boundaries = self.lane_cells_info[lane_id]['cell_boundaries']
            is_passable = self._check_lane_passable(lane_id, current_phase_index)

            for cell_idx, vehicles in enumerate(cells_vehicles):
                is_incoming = self.lane_cells_info[lane_id]['is_incoming']
                if is_incoming:
                    # boundaries 降序：cell[i] 宽度 = boundaries[i] - boundaries[i+1]
                    cell_length = cell_boundaries[cell_idx] - cell_boundaries[cell_idx + 1]
                else:
                    cell_length = cell_boundaries[cell_idx + 1] - cell_boundaries[cell_idx]
                distance_to_lane_start = self.lane_cells_info[lane_id]['cell_centers'][cell_idx]

                if len(vehicles) == 0:
                    lane_features.append({
                        'vehicle_count': 0,
                        'avg_speed': 0.0,
                        'avg_waiting_time': 0.0,
                        'avg_accumulated_waiting_time': 0.0,
                        'distance_to_lane_start': distance_to_lane_start,
                        'occupancy': 0.0,
                        'is_passable': is_passable
                    })
                else:
                    speeds = [v['speed'] for v in vehicles]
                    waiting_times = [v['waiting_time'] for v in vehicles]
                    accumulated_waiting_times = [v['accumulated_waiting_time'] for v in vehicles]
                    vehicle_lengths = [v['length'] for v in vehicles]
                    total_vehicle_length = sum(vehicle_lengths)
                    occupancy = min(total_vehicle_length / cell_length, 1.0)

                    lane_features.append({
                        'vehicle_count': len(vehicles),
                        'avg_speed': np.mean(speeds),
                        'avg_waiting_time': np.mean(waiting_times),
                        'avg_accumulated_waiting_time': np.mean(accumulated_waiting_times),
                        'distance_to_lane_start': distance_to_lane_start,
                        'occupancy': occupancy,
                        'is_passable': is_passable
                    })

            lane_dynamic_features[lane_id] = lane_features

        return lane_dynamic_features

    def _check_lane_passable(self, lane_id: str, current_phase_index: int = None) -> int:
        """检查 lane 在当前相位是否可以通行"""
        if current_phase_index is None:
            return 0
        if lane_id not in self.static_lane_features:
            return 0
        phase_binding = self.static_lane_features[lane_id].get('phase_binding', [])
        if not phase_binding or current_phase_index >= len(phase_binding):
            return 0
        return phase_binding[current_phase_index]

    def get_lane_summary(self, lane_dynamic_features: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """获取每条 lane 的汇总统计信息"""
        lane_summary = {}
        for lane_id, cells in lane_dynamic_features.items():
            total_vehicles = sum(cell['vehicle_count'] for cell in cells)
            if total_vehicles == 0:
                lane_summary[lane_id] = {
                    'total_vehicles': 0, 'avg_speed': 0.0,
                    'avg_waiting_time': 0.0, 'avg_accumulated_waiting_time': 0.0,
                    'avg_occupancy': 0.0
                }
            else:
                weights = [cell['vehicle_count'] for cell in cells]
                total_weight = sum(weights)
                lane_summary[lane_id] = {
                    'total_vehicles': total_vehicles,
                    'avg_speed': sum(c['avg_speed'] * w for c, w in zip(cells, weights)) / total_weight,
                    'avg_waiting_time': sum(c['avg_waiting_time'] * w for c, w in zip(cells, weights)) / total_weight,
                    'avg_accumulated_waiting_time': sum(c['avg_accumulated_waiting_time'] * w for c, w in zip(cells, weights)) / total_weight,
                    'avg_occupancy': np.mean([c['occupancy'] for c in cells])
                }
        return lane_summary


def aggregate_features_seq(
    seq: List[Dict[str, List[Dict]]],
    method: str = 'last'
) -> Dict[str, List[Dict]]:
    """将特征序列聚合为单个快照

    Args:
        seq: TSCInfoWrapper.step() 返回的特征序列，长度 = 子步数
        method: 聚合方式
            'last'  - 取最后一帧（默认，等价于旧行为）
            'mean'  - 各指标在时间维度上取均值
            'max'   - vehicle_count/occupancy/waiting_time 取最大值，速度取最小值

    Returns:
        聚合后的 lane_dynamic_features（与单帧格式相同）
    """
    if method == 'last':
        return seq[-1]

    lane_ids = list(seq[0].keys())

    if method == 'mean':
        result = {}
        for lane_id in lane_ids:
            num_cells = len(seq[0][lane_id])
            cells_out = []
            for cell_idx in range(num_cells):
                keys = seq[0][lane_id][cell_idx].keys()
                agg = {}
                for k in keys:
                    vals = [step[lane_id][cell_idx][k] for step in seq]
                    agg[k] = float(np.mean(vals))
                cells_out.append(agg)
            result[lane_id] = cells_out
        return result

    if method == 'max':
        # 对排队压力相关指标取最大，速度取最小，is_passable 取最后帧
        MAX_KEYS = {'vehicle_count', 'avg_waiting_time', 'avg_accumulated_waiting_time', 'occupancy'}
        MIN_KEYS = {'avg_speed'}
        result = {}
        for lane_id in lane_ids:
            num_cells = len(seq[0][lane_id])
            cells_out = []
            for cell_idx in range(num_cells):
                keys = seq[0][lane_id][cell_idx].keys()
                agg = {}
                for k in keys:
                    vals = [step[lane_id][cell_idx][k] for step in seq]
                    if k in MAX_KEYS:
                        agg[k] = float(np.max(vals))
                    elif k in MIN_KEYS:
                        agg[k] = float(np.min(vals))
                    else:  # distance_to_lane_start, is_passable 等取最后帧
                        agg[k] = vals[-1]
                cells_out.append(agg)
            result[lane_id] = cells_out
        return result

    raise ValueError(f"Unknown aggregation method: {method!r}. Choose 'last', 'mean', or 'max'.")


def format_lane_features_to_array(
    lane_dynamic_features: Dict[str, List[Dict]],
    lane_order: List[str] = None,
    max_cells: int = None
) -> np.ndarray:
    """将 lane 动态特征格式化为数组

    Returns:
        feature_array: shape (num_lanes, max_cells, 6)
    """
    if lane_order is None:
        lane_order = sorted(lane_dynamic_features.keys())
    if max_cells is None:
        max_cells = max(len(cells) for cells in lane_dynamic_features.values())

    num_features = 6
    num_lanes = len([lid for lid in lane_order if lid in lane_dynamic_features])
    feature_array = np.zeros((num_lanes, max_cells, num_features), dtype=np.float32)

    lane_idx = 0
    for lane_id in lane_order:
        if lane_id not in lane_dynamic_features:
            continue
        cells = lane_dynamic_features[lane_id]
        for cell_idx, cell in enumerate(cells):
            if cell_idx >= max_cells:
                break
            feature_array[lane_idx, cell_idx, :] = [
                cell['vehicle_count'], cell['avg_speed'],
                cell['avg_waiting_time'], cell['avg_accumulated_waiting_time'],
                cell['occupancy'], cell.get('is_passable', 0)
            ]
        lane_idx += 1
    return feature_array


def create_lane_cell_mask(
    lane_cells_info: Dict[str, Dict],
    lane_order: List[str] = None,
    max_cells: int = None
) -> np.ndarray:
    """创建 lane cell 的 mask

    Returns:
        mask: shape (num_lanes, max_cells), True 表示有效 cell
    """
    if lane_order is None:
        lane_order = sorted(lane_cells_info.keys())
    if max_cells is None:
        max_cells = max(info['num_cells'] for info in lane_cells_info.values())

    num_lanes = len([lid for lid in lane_order if lid in lane_cells_info])
    mask = np.zeros((num_lanes, max_cells), dtype=bool)

    lane_idx = 0
    for lane_id in lane_order:
        if lane_id not in lane_cells_info:
            continue
        num_cells = lane_cells_info[lane_id]['num_cells']
        mask[lane_idx, :num_cells] = True
        lane_idx += 1
    return mask
