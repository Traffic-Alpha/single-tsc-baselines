'''
@Author: WANG Maonan
@Date: 2026-01-11
@Description: 动态状态处理工具，将 lane 分成多个 cell，计算每个 cell 的动态信息
LastEditTime: 2026-02-12 19:39:57
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Any, Optional


# Global value ranges for different metrics (for consistent visualization across time steps)
METRIC_VALUE_RANGES = {
    'occupancy': {
        'vmin': 0.0,
        'vmax': 1.0,
        'description': 'Occupancy rate (0-1)',
        'unit': ''
    },
    'vehicle_count': {
        'vmin': 0,
        'vmax': 3,  # Typical maximum vehicles per cell
        'description': 'Number of vehicles',
        'unit': 'vehicles'
    },
    'avg_speed': {
        'vmin': 0.0,
        'vmax': 15.0,  # Typical urban speed limit (m/s, ~54 km/h)
        'description': 'Average speed',
        'unit': 'm/s'
    },
    'avg_waiting_time': {
        'vmin': 0.0,
        'vmax': 120.0,  # Maximum 2 minutes waiting time
        'description': 'Average waiting time',
        'unit': 's'
    },
    'avg_accumulated_waiting_time': {
        'vmin': 0.0,
        'vmax': 300.0,  # Maximum 5 minutes accumulated waiting
        'description': 'Average accumulated waiting time',
        'unit': 's'
    },
    'distance_to_lane_start': {
        'vmin': 0.0,
        'vmax': 100.0,  # Typical lane length
        'description': 'Distance to lane start',
        'unit': 'm'
    },
    'is_passable': {
        'vmin': 0,
        'vmax': 1,
        'description': 'Lane passability (0: red, 1: green)',
        'unit': ''
    }
}


def get_metric_value_range(metric: str, custom_ranges: Dict[str, Dict] = None) -> Tuple[float, float]:
    """Get the recommended value range for a specific metric
    
    Args:
        metric: Metric name
        custom_ranges: Custom value ranges to override defaults
        
    Returns:
        (vmin, vmax): Minimum and maximum values for the metric
    """
    if custom_ranges and metric in custom_ranges:
        return custom_ranges[metric]['vmin'], custom_ranges[metric]['vmax']
    
    if metric in METRIC_VALUE_RANGES:
        return METRIC_VALUE_RANGES[metric]['vmin'], METRIC_VALUE_RANGES[metric]['vmax']
    
    # Default range if metric not found
    return 0.0, 1.0


def update_metric_value_range(metric: str, vmin: float, vmax: float) -> None:
    """Update the global value range for a specific metric
    
    Args:
        metric: Metric name
        vmin: New minimum value
        vmax: New maximum value
    """
    if metric in METRIC_VALUE_RANGES:
        METRIC_VALUE_RANGES[metric]['vmin'] = vmin
        METRIC_VALUE_RANGES[metric]['vmax'] = vmax
    else:
        METRIC_VALUE_RANGES[metric] = {
            'vmin': vmin,
            'vmax': vmax,
            'description': metric,
            'unit': ''
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
    
    print("\n" + "="*70)
    print("Usage:")
    print("  wrapped_env.visualize_congestion(")
    print("      metric='occupancy',")
    print("      vmin=0.0, vmax=1.0,  # Use consistent range")
    print("      save_path=f'./step_{t}.png'")
    print("  )")
    print("="*70 + "\n")


class LaneCellManager:
    """管理 lane 的 cell 划分和动态信息计算
    
    每条 lane 根据固定的 cell 长度进行划分，不同长度的 lane 会有不同数量的 cell
    """
    
    def __init__(self, 
        static_lane_features: Dict[str, Dict], 
        cell_length: float = 10.0, 
    ) -> None:
        """初始化 Lane Cell Manager
        
        Args:
            static_lane_features: 车道静态特征字典
            cell_length: 每个 cell 的固定长度（米），默认 10 米
        """
        self.static_lane_features = static_lane_features
        self.cell_length = cell_length
        
        # 存储每条 lane 的 cell 划分信息
        self.lane_cells_info = {}
        self._initialize_lane_cells()
    
    def _initialize_lane_cells(self) -> None:
        """初始化每条 lane 的 cell 划分信息
        
        根据固定的 cell_length 和 lane 的实际长度，计算每条 lane 的 cell 数量。
        例如：lane_length=55m, cell_length=10m -> 6 个 cell (最后一个 5m)
        
        区分 incoming lane 和 outgoing lane：
        - incoming lane: cell 从尾部（靠近路口）开始切分，索引0在路口端
        - outgoing lane: cell 从头部开始切分，索引0在起点端
        """
        for lane_id, features in self.static_lane_features.items():
            lane_length = features['length'] * 100  # 反归一化得到实际长度（米）
            
            # 计算该 lane 的 cell 数量（向上取整）
            num_cells = int(np.ceil(lane_length / self.cell_length))
            
            # 判断是 incoming 还是 outgoing lane
            io_type = features.get('io_type', [0, 0])
            is_incoming = (io_type[0] == 1)  # [1, 0] 表示 incoming lane
            
            if is_incoming:
                # incoming lane: 从尾部（lane_length）开始，向起点（0）方向切分
                # cell 0 距离路口最近，cell n-1 距离路口最远
                cell_boundaries = []  # N 个 cell 会有 N+1 个边界
                for i in range(num_cells + 1):
                    boundary = max(lane_length - i * self.cell_length, 0)
                    cell_boundaries.append(boundary)
                cell_boundaries = np.array(cell_boundaries[::-1])  # 反转，使其仍然是递增的
            else:
                # outgoing lane: 从头部（0）开始，向尾部（lane_length）方向切分
                # cell 0 在起点，cell n-1 在终点
                cell_boundaries = []  # N 个 cell 会有 N+1 个边界
                for i in range(num_cells + 1):
                    boundary = min(i * self.cell_length, lane_length)
                    cell_boundaries.append(boundary)
                cell_boundaries = np.array(cell_boundaries)
            
            self.lane_cells_info[lane_id] = {
                'length': lane_length,
                'num_cells': num_cells,
                'cell_boundaries': cell_boundaries,
                'cell_centers': (cell_boundaries[:-1] + cell_boundaries[1:]) / 2,  # 每个 cell 的中心距离 lane 起点的距离
                'is_incoming': is_incoming,  # 记录 lane 类型，方便后续使用
            }
    
    def __get_vehicle_cell_index(self, lane_id: str, lane_position: float) -> int:
        """获取车辆所在的 cell 索引
        
        Args:
            lane_id: 车道 ID
            lane_position: 车辆在车道上的位置（米）
            
        Returns:
            cell_index: cell 索引 (0 到 num_cells-1)，如果车辆不在该 lane 返回 -1
        """
        if lane_id not in self.lane_cells_info:
            return -1
        
        lane_info = self.lane_cells_info[lane_id]
        cell_boundaries = lane_info['cell_boundaries']
        num_cells = lane_info['num_cells']
        
        # 使用 numpy 的 searchsorted 找到车辆所在的 cell
        # searchsorted 返回应该插入的位置，减1得到所在的 cell 索引
        cell_index = np.searchsorted(cell_boundaries, lane_position, side='right') - 1
        
        # 确保索引在有效范围内
        cell_index = np.clip(cell_index, 0, num_cells - 1)
        
        return int(cell_index)
    
    def calculate_lane_dynamic_features(
        self, 
        vehicles_state: Dict[str, Dict],
        current_phase_index: int = None
    ) -> Dict[str, List[Dict]]:
        """计算每条 lane 中每个 cell 的动态特征
        
        Args:
            vehicles_state: 车辆状态字典，来自 state['vehicle']
            current_phase_index: 当前信号灯相位索引，用于判断 lane 是否可以通行
            
        Returns:
            lane_dynamic_features: 每条 lane 的 cell 动态特征
                {
                    'lane_id': [
                        {  # cell 0
                            'vehicle_count': int,
                            'avg_speed': float,
                            'avg_waiting_time': float,
                            'avg_accumulated_waiting_time': float,
                            'distance_to_lane_start': float,  # cell 中心距离 lane 入口的距离
                            'occupancy': float,  # 占用率 (0-1)
                            'is_passable': int  # 是否可以通行 (1: 可以, 0: 不可以)
                        },
                        ...  # 更多 cell（数量取决于 lane 长度）
                    ]
                }
        """
        # 初始化每条 lane 的每个 cell 的统计信息（每条 lane 的 cell 数量可能不同）
        lane_cell_vehicles = {}
        for lane_id, lane_info in self.lane_cells_info.items():
            num_cells = lane_info['num_cells']
            lane_cell_vehicles[lane_id] = [[] for _ in range(num_cells)]
        
        # 将车辆分配到对应的 cell
        for veh_id, veh_info in vehicles_state.items():
            lane_id = veh_info['lane_id'] # 车辆所在的车道 id
            
            if lane_id not in self.lane_cells_info:
                continue # 车辆不在当前路口，跳过
            
            lane_position = veh_info['lane_position'] # 车辆在车道上的位置（米）
            cell_index = self.__get_vehicle_cell_index(lane_id, lane_position)
            
            if cell_index >= 0:
                lane_cell_vehicles[lane_id][cell_index].append(veh_info)
        
        # 计算每个 cell 的统计特征
        lane_dynamic_features = {}
        
        for lane_id, cells_vehicles in lane_cell_vehicles.items():
            lane_features = []
            cell_boundaries = self.lane_cells_info[lane_id]['cell_boundaries']
            
            # 判断该 lane 是否可以通行（基于 phase_binding）
            is_passable = self._check_lane_passable(lane_id, current_phase_index)
            
            for cell_idx, vehicles in enumerate(cells_vehicles):
                # 计算 cell 长度
                cell_length = cell_boundaries[cell_idx + 1] - cell_boundaries[cell_idx]
                
                # 计算 cell 中心距离 lane 入口的距离
                distance_to_lane_start = self.lane_cells_info[lane_id]['cell_centers'][cell_idx]
                
                # 如果 cell 中没有车辆
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
                    # 计算平均值
                    speeds = [v['speed'] for v in vehicles]
                    waiting_times = [v['waiting_time'] for v in vehicles]
                    accumulated_waiting_times = [v['accumulated_waiting_time'] for v in vehicles]
                    vehicle_lengths = [v['length'] for v in vehicles]
                    
                    # 计算占用率：车辆总长度 / cell 长度
                    total_vehicle_length = sum(vehicle_lengths)
                    occupancy = min(total_vehicle_length / cell_length, 1.0)  # 最大为1
                    
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
        """检查 lane 在当前相位是否可以通行
        
        Args:
            lane_id: 车道 ID
            current_phase_index: 当前信号灯相位索引
            
        Returns:
            1 表示可以通行（绿灯），0 表示不可以通行（红灯）
            如果 current_phase_index 为 None 或者 lane 没有 phase_binding，返回 0
        """
        if current_phase_index is None:
            return 0
        
        if lane_id not in self.static_lane_features:
            return 0
        
        phase_binding = self.static_lane_features[lane_id].get('phase_binding', [])
        
        # 如果 phase_binding 为空或者 current_phase_index 超出范围
        if not phase_binding or current_phase_index >= len(phase_binding):
            return 0
        
        # 返回该 lane 在当前相位是否可以通行
        return phase_binding[current_phase_index]
    
    def get_lane_summary(self, lane_dynamic_features: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """获取每条 lane 的汇总统计信息
        
        Args:
            lane_dynamic_features: 每条 lane 的 cell 动态特征
            
        Returns:
            lane_summary: 每条 lane 的汇总信息
        """
        lane_summary = {}
        
        for lane_id, cells in lane_dynamic_features.items():
            total_vehicles = sum(cell['vehicle_count'] for cell in cells)
            
            if total_vehicles == 0:
                lane_summary[lane_id] = {
                    'total_vehicles': 0,
                    'avg_speed': 0.0,
                    'avg_waiting_time': 0.0,
                    'avg_accumulated_waiting_time': 0.0,
                    'avg_occupancy': 0.0
                }
            else:
                # 使用车辆数量作为权重计算加权平均
                weights = [cell['vehicle_count'] for cell in cells]
                total_weight = sum(weights)
                
                weighted_speed = sum(
                    cell['avg_speed'] * weight 
                    for cell, weight in zip(cells, weights)
                ) / total_weight
                
                weighted_waiting_time = sum(
                    cell['avg_waiting_time'] * weight 
                    for cell, weight in zip(cells, weights)
                ) / total_weight
                
                weighted_accumulated_waiting_time = sum(
                    cell['avg_accumulated_waiting_time'] * weight 
                    for cell, weight in zip(cells, weights)
                ) / total_weight
                
                avg_occupancy = np.mean([cell['occupancy'] for cell in cells])
                
                lane_summary[lane_id] = {
                    'total_vehicles': total_vehicles,
                    'avg_speed': weighted_speed,
                    'avg_waiting_time': weighted_waiting_time,
                    'avg_accumulated_waiting_time': weighted_accumulated_waiting_time,
                    'avg_occupancy': avg_occupancy
                }
        
        return lane_summary


def format_lane_features_to_array(
    lane_dynamic_features: Dict[str, List[Dict]], 
    lane_order: List[str] = None,
    max_cells: int = None
) -> np.ndarray:
    """将 lane 动态特征格式化为数组，方便神经网络处理
    
    由于不同 lane 的长度不同，cell 数量也不同。此函数会将所有 lane 填充到相同的 cell 数量。
    
    Args:
        lane_dynamic_features: 每条 lane 的 cell 动态特征
        lane_order: lane 的排序列表，如果为 None 则使用字典顺序
        max_cells: 最大 cell 数量，如果为 None 则自动计算（使用最大值）
        
    Returns:
        feature_array: shape (num_lanes, max_cells, num_features)
            num_features = 6 (vehicle_count, avg_speed, avg_waiting_time, 
                             avg_accumulated_waiting_time, occupancy, is_passable)
            对于 cell 数量少于 max_cells 的 lane，多余的 cell 用 0 填充
    """
    if lane_order is None:
        lane_order = sorted(lane_dynamic_features.keys())
    
    # 找到最大的 cell 数量
    if max_cells is None:
        max_cells = max(len(cells) for cells in lane_dynamic_features.values())
    
    num_features = 6  # 更新为 6，包含 is_passable
    num_lanes = len([lid for lid in lane_order if lid in lane_dynamic_features])
    
    # 初始化数组（用 0 填充）
    feature_array = np.zeros((num_lanes, max_cells, num_features), dtype=np.float32)
    
    lane_idx = 0
    for lane_id in lane_order:
        if lane_id not in lane_dynamic_features:
            continue
        
        cells = lane_dynamic_features[lane_id]
        
        for cell_idx, cell in enumerate(cells):
            if cell_idx >= max_cells:
                break  # 如果超过 max_cells，跳过
                
            feature_array[lane_idx, cell_idx, :] = [
                cell['vehicle_count'],
                cell['avg_speed'],
                cell['avg_waiting_time'],
                cell['avg_accumulated_waiting_time'],
                cell['occupancy'],
                cell.get('is_passable', 0)  # 添加 is_passable，如果不存在则默认为 0
            ]
        
        lane_idx += 1
    
    return feature_array


def create_lane_cell_mask(
    lane_cells_info: Dict[str, Dict],
    lane_order: List[str] = None,
    max_cells: int = None
) -> np.ndarray:
    """创建 lane cell 的 mask，标记哪些 cell 是有效的（不是 padding）
    
    Args:
        lane_cells_info: 每条 lane 的 cell 信息（来自 LaneCellManager.lane_cells_info）
        lane_order: lane 的排序列表，如果为 None 则使用字典顺序
        max_cells: 最大 cell 数量，如果为 None 则自动计算
        
    Returns:
        mask: shape (num_lanes, max_cells)，True 表示有效的 cell，False 表示 padding
    """
    if lane_order is None:
        lane_order = sorted(lane_cells_info.keys())
    
    # 找到最大的 cell 数量
    if max_cells is None:
        max_cells = max(info['num_cells'] for info in lane_cells_info.values())
    
    num_lanes = len([lid for lid in lane_order if lid in lane_cells_info])
    
    # 创建 mask（初始化为 False）
    mask = np.zeros((num_lanes, max_cells), dtype=bool)
    
    lane_idx = 0
    for lane_id in lane_order:
        if lane_id not in lane_cells_info:
            continue
        
        num_cells = lane_cells_info[lane_id]['num_cells']
        mask[lane_idx, :num_cells] = True  # 标记有效的 cell
        
        lane_idx += 1
    
    return mask


def visualize_lane_congestion(
    lane_dynamic_features: Dict[str, List[Dict]],
    lane_order: List[str] = None,
    metric: str = 'occupancy',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    title: Optional[str] = None,
    cmap: str = None,
    show_values: bool = True,
    vmin: float = None,
    vmax: float = None
) -> None:
    """Visualize lane cell congestion heatmap
    
    Args:
        lane_dynamic_features: Dynamic features for each lane's cells
        lane_order: Lane ordering list, uses dictionary order if None
        metric: Metric to visualize, options:
            - 'occupancy': Occupancy rate (0-1)
            - 'vehicle_count': Number of vehicles
            - 'avg_speed': Average speed (m/s)
            - 'avg_waiting_time': Average waiting time (s)
            - 'avg_accumulated_waiting_time': Average accumulated waiting time (s)
        save_path: Path to save the figure, displays if None
        figsize: Figure size
        title: Figure title, uses default if None
        cmap: Color map, uses auto-selected professional palette if None
        show_values: Whether to show values on cells
        vmin: Minimum value for color mapping, auto-calculated if None
        vmax: Maximum value for color mapping, auto-calculated if None
    """
    if lane_order is None:
        lane_order = sorted(lane_dynamic_features.keys())
    
    # 找到最大的 cell 数量
    max_cells = max(len(cells) for cells in lane_dynamic_features.values())
    num_lanes = len([lid for lid in lane_order if lid in lane_dynamic_features])
    
    # 创建数据矩阵（用 NaN 填充）
    data_matrix = np.full((num_lanes, max_cells), np.nan)
    
    # 填充数据
    lane_idx = 0
    for lane_id in lane_order:
        if lane_id not in lane_dynamic_features:
            continue
        
        cells = lane_dynamic_features[lane_id]
        for cell_idx, cell in enumerate(cells):
            if cell_idx >= max_cells:
                break
            data_matrix[lane_idx, cell_idx] = cell[metric]
        
        lane_idx += 1
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set color mapping range
    # Use recommended global ranges for consistency across time steps
    if vmin is None or vmax is None:
        default_vmin, default_vmax = get_metric_value_range(metric)
        if vmin is None:
            vmin = default_vmin
        if vmax is None:
            vmax = default_vmax
    
    # Auto-select professional color maps for each metric
    if cmap is None:
        default_cmaps = {
            'occupancy': 'YlOrRd',              # Yellow to Orange to Red
            'vehicle_count': 'Blues',            # White to Blue
            'avg_speed': 'RdYlBu_r',            # Red (low) to Yellow to Blue (high)
            'avg_waiting_time': 'plasma',        # Viridis-like, perceptually uniform
            'avg_accumulated_waiting_time': 'viridis'  # Popular perceptually uniform
        }
        cmap_use = default_cmaps.get(metric, 'viridis')
    else:
        cmap_use = cmap
    
    # Create masked array to handle NaN values
    masked_data = np.ma.masked_invalid(data_matrix)
    
    # Draw heatmap
    im = ax.imshow(
        masked_data, 
        cmap=cmap_use, 
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    
    # Add colorbar
    metric_labels = {
        'occupancy': 'Occupancy',
        'vehicle_count': 'Vehicle Count',
        'avg_speed': 'Average Speed (m/s)',
        'avg_waiting_time': 'Average Waiting Time (s)',
        'avg_accumulated_waiting_time': 'Avg. Accumulated Waiting Time (s)'
    }
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_labels.get(metric, metric), fontsize=12)
    
    # Show values on each cell
    if show_values:
        for i in range(num_lanes):
            for j in range(max_cells):
                if not np.isnan(data_matrix[i, j]):
                    # Choose text color based on background color
                    normalized_val = (data_matrix[i, j] - vmin) / (vmax - vmin + 1e-10)
                    text_color = 'white' if normalized_val > 0.5 else 'black'
                    
                    # Format values
                    if metric == 'vehicle_count':
                        text = f'{int(data_matrix[i, j])}'
                    else:
                        text = f'{data_matrix[i, j]:.2f}'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                           color=text_color, fontsize=8)
    
    # Set axes
    valid_lane_ids = [lid for lid in lane_order if lid in lane_dynamic_features]
    ax.set_yticks(range(num_lanes))
    ax.set_yticklabels(valid_lane_ids, fontsize=9)
    ax.set_ylabel('Lane ID', fontsize=12, fontweight='bold')
    
    ax.set_xticks(range(max_cells))
    ax.set_xticklabels([f'Cell {i}' for i in range(max_cells)], rotation=45, ha='right')
    ax.set_xlabel('Cell Index', fontsize=12, fontweight='bold')
    
    # Set title
    if title is None:
        title = f'Lane Cell Congestion Heatmap - {metric_labels.get(metric, metric)}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid lines
    ax.set_xticks(np.arange(max_cells) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_lanes) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multiple_metrics(
    lane_dynamic_features: Dict[str, List[Dict]],
    lane_order: List[str] = None,
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12),
    title: Optional[str] = None
) -> None:
    """Visualize multiple metrics of lane cell congestion simultaneously
    
    Args:
        lane_dynamic_features: Dynamic features for each lane's cells
        lane_order: Lane ordering list
        metrics: List of metrics to visualize, uses default 4 metrics if None
        save_path: Path to save the figure
        figsize: Figure size
        title: Figure title
    """
    if metrics is None:
        metrics = ['occupancy', 'vehicle_count', 'avg_speed', 'avg_waiting_time']
    
    if lane_order is None:
        lane_order = sorted(lane_dynamic_features.keys())
    
    # 找到最大的 cell 数量
    max_cells = max(len(cells) for cells in lane_dynamic_features.values())
    num_lanes = len([lid for lid in lane_order if lid in lane_dynamic_features])
    
    # 创建子图
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    metric_labels = {
        'occupancy': 'Occupancy',
        'vehicle_count': 'Vehicle Count',
        'avg_speed': 'Average Speed (m/s)',
        'avg_waiting_time': 'Average Waiting Time (s)',
        'avg_accumulated_waiting_time': 'Avg. Accumulated Waiting Time (s)'
    }
    
    metric_cmaps = {
        'occupancy': 'YlOrRd',            # Yellow to Orange to Red
        'vehicle_count': 'Blues',          # White to Blue
        'avg_speed': 'RdYlBu_r',          # Red (low) to Yellow to Blue (high)
        'avg_waiting_time': 'plasma',      # Perceptually uniform
        'avg_accumulated_waiting_time': 'viridis'  # Perceptually uniform
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create data matrix
        data_matrix = np.full((num_lanes, max_cells), np.nan)
        
        lane_idx = 0
        for lane_id in lane_order:
            if lane_id not in lane_dynamic_features:
                continue
            
            cells = lane_dynamic_features[lane_id]
            for cell_idx, cell in enumerate(cells):
                if cell_idx >= max_cells:
                    break
                data_matrix[lane_idx, cell_idx] = cell[metric]
            
            lane_idx += 1
        
        # Create masked array
        masked_data = np.ma.masked_invalid(data_matrix)
        
        # Draw heatmap
        im = ax.imshow(
            masked_data,
            cmap=metric_cmaps.get(metric, 'viridis'),
            aspect='auto',
            interpolation='nearest'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_labels.get(metric, metric), fontsize=10)
        
        # Set axes
        valid_lane_ids = [lid for lid in lane_order if lid in lane_dynamic_features]
        ax.set_yticks(range(num_lanes))
        ax.set_yticklabels(valid_lane_ids, fontsize=8)
        
        if idx == n_metrics - 1:
            ax.set_xticks(range(max_cells))
            ax.set_xticklabels([f'Cell {i}' for i in range(max_cells)], rotation=45, ha='right')
            ax.set_xlabel('Cell Index', fontsize=10, fontweight='bold')
        else:
            ax.set_xticks([])
        
        ax.set_ylabel('Lane ID', fontsize=10, fontweight='bold')
        ax.set_title(metric_labels.get(metric, metric), fontsize=11, fontweight='bold')
        
        # Add grid lines
        ax.set_xticks(np.arange(max_cells) - 0.5, minor=True)
        ax.set_yticks(np.arange(num_lanes) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set overall title
    if title is None:
        title = 'Lane Cell Multi-Metric Congestion Visualization'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

