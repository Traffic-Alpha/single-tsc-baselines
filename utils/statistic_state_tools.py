'''
@Author: WANG Maonan
@Date: 2026-01-10
@Description: 工具函数，用于计算交通信号控制环境中的特征
LastEditTime: 2026-02-12 16:35:42
'''
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any


def angle_to_vector(angle: float, is_degrees: bool = False) -> List[float]:
    """将角度转换为单位方向向量
    
    Args:
        angle: 角度值（弧度或度数）
        is_degrees: 如果为 True，输入角度为度数；如果为 False，输入角度为弧度（默认）
        
    Returns:
        [vx, vy] 单位方向向量
        
    Notes:
        - 角度约定：0 度/弧度指向正东（正 X 轴方向）
        - 逆时针为正方向
    """
    if is_degrees:
        angle_rad = np.radians(angle)
    else:
        angle_rad = angle
    
    vx = np.cos(angle_rad)
    vy = np.sin(angle_rad)
    
    return [float(vx), float(vy)]


def calculate_normalized_length(length: float) -> float:
    """计算归一化的车道长度
    
    Args:
        length: 车道长度
        
    Returns:
        归一化后的长度 (除以 100)
    """
    return length / 100.0


def calculate_normalized_position(
    shape: List[Tuple[float, float]], 
    junction_center: Tuple[float, float]
) -> List[float]:
    """计算车道相对于路口中心的归一化位置
    
    使用 shape 的最后一个点（车道出口位置），计算相对于路口中心的位置
    
    Args:
        shape: 车道的形状点列表 [(x1, y1), (x2, y2), ...]
        junction_center: 路口中心坐标 (x, y)
        
    Returns:
        [x1, y1, x2, y2] 归一化后的位置 (除以 100)
    """
    point1 = shape[0] # 起点
    point2 = shape[1] # 终点
    
    # 相对于路口中心的坐标
    rel_x1 = (point1[0] - junction_center[0]) / 100.0
    rel_y1 = (point1[1] - junction_center[1]) / 100.0

    rel_x2 = (point2[0] - junction_center[0]) / 100.0
    rel_y2 = (point2[1] - junction_center[1]) / 100.0
    
    return [rel_x1, rel_y1, rel_x2, rel_y2]


def build_lane_turn_function_mapping(
    movement_lanes: Dict[str, List[str]]
) -> Dict[str, List[int]]:
    """构建 lane 到转向功能的映射
    
    根据 movement_lanes 中的信息，判断每条车道的转向功能：
    - --s: Straight (直行) -> [1, 0, 0]
    - --l: Left (左转) -> [0, 1, 0]
    - --r: Right (右转) -> [0, 0, 1]
    - Other (Outgoing/Unknown) -> [0, 0, 0]
    
    Args:
        movement_lanes: movement 到 lane 列表的映射
            例如 {'-E1--r': ['-E1_0'], '-E1--s': ['-E1_1'], '-E1--l': ['-E1_2'], ...}
    
    Returns:
        lane_to_turn: lane 到转向功能的映射
            例如 {'-E1_0': [0, 0, 1], '-E1_1': [1, 0, 0], '-E1_2': [0, 1, 0], ...}
    """
    lane_to_turn = {}
    
    for movement_id, lanes in movement_lanes.items():
        # 根据 movement_id 确定转向类型
        if '--s' in movement_id:
            turn_function = [1, 0, 0]  # Straight
        elif '--l' in movement_id:
            turn_function = [0, 1, 0]  # Left
        elif '--r' in movement_id:
            turn_function = [0, 0, 1]  # Right
        else:
            turn_function = [0, 0, 0]  # Other/Unknown
        
        # 为该 movement 的所有 lane 设置相同的转向功能
        for lane_id in lanes:
            lane_to_turn[lane_id] = turn_function
    
    return lane_to_turn


def build_lane_phase_binding_mapping(
    movement_lanes: Dict[str, List[str]], 
    phase2movements: Dict[int, List[str]]
) -> Dict[str, List[int]]:
    """构建 lane 到相位绑定的映射
    
    根据 phase2movements 中的信息，判断每条车道属于哪些相位：
    - Incoming Lane: Multi-hot 编码，如果属于某个相位则对应位为 1
    - Outgoing Lane: 全 0 (不受灯控)
    
    Args:
        movement_lanes: movement 到 lane 列表的映射
            例如 {'-E1--r': ['-E1_0'], '-E1--s': ['-E1_1'], '-E1--l': ['-E1_2'], ...}
        phase2movements: 相位到 movement 列表的映射
            例如 {0: ['-E3--s', '-E2--s'], 1: ['-E3--l', '-E2--l'], ...}
    
    Returns:
        lane_to_phases: lane 到相位绑定的映射
            例如 {'-E1_0': [1, 0, 1, 0], '-E1_1': [0, 1, 0, 0], ...}
    """
    num_phases = len(phase2movements)
    lane_to_phases = {}
    
    # 首先构建 movement 到 phases 的映射
    movement_to_phases = {}
    for phase_id, movements in phase2movements.items():
        for movement_id in movements:
            if movement_id not in movement_to_phases:
                movement_to_phases[movement_id] = []
            movement_to_phases[movement_id].append(phase_id)
    
    # 然后为每个 lane 构建相位绑定
    for movement_id, lanes in movement_lanes.items():
        # 初始化相位绑定（全 0）
        phase_binding = [0] * num_phases
        
        # 如果该 movement 属于某些相位，设置对应位为 1
        if movement_id in movement_to_phases:
            for phase_id in movement_to_phases[movement_id]:
                phase_binding[phase_id] = 1
        
        # 为该 movement 的所有 lane 设置相同的相位绑定
        for lane_id in lanes:
            lane_to_phases[lane_id] = phase_binding
    
    return lane_to_phases


def visualize_lane_features(
    static_lane_features: Dict[str, Dict[str, Any]], 
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 12),
    arrow_scale: float = 0.3,
) -> None:
    """Visualize static features of lanes
    
    Plot each lane as an arrow starting from its position.
    Supports both basic and enhanced heading vector visualization.
    
    Args:
        static_lane_features: Dictionary of lane static features, format:
            {
                'lane_id': {
                    'length': float,        # Normalized length
                    'position': [x, y],     # Lane position (relative to junction center)
                    'heading': [vx, vy]     # Heading vector (direction depends on method used)
                }
            }
        save_path: Path to save the figure, if None, display the figure
        figsize: Figure size
        arrow_scale: Scale factor for arrows (default 0.3)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot junction center
    ax.scatter(0, 0, c='red', s=300, marker='X', zorder=5, 
               edgecolors='darkred', linewidth=2, label='Junction Center')
    
    # Iterate through all lanes
    for lane_id, features in static_lane_features.items():
        position = features['position']  # [x, y] - lane position
        heading = features['heading']    # [vx, vy] - direction vector
        
        # Extract position coordinates
        x1, y1, x2, y2 = position
        vx, vy = heading
        
        direction_color = 'green'
        position_color = 'blue'
        alpha = 0.8
    
        # Draw arrow starting from the position
        ax.arrow(
            x1, y1,  # Start from lane position
            vx * arrow_scale, vy * arrow_scale,  # Direction and length
            head_width=0.04, head_length=0.03,
            fc=direction_color, ec=direction_color, linewidth=1.5,
            zorder=3, alpha=alpha, length_includes_head=True
        )
        
        # Mark the lane position
        ax.scatter(x1, y1, c=position_color, s=50, zorder=4, alpha=alpha, 
                  edgecolors='black', linewidth=0.8)
        ax.scatter(x2, y2, c=position_color, s=50, zorder=4, alpha=alpha, 
                  edgecolors='black', linewidth=0.8)
        ax.plot([x1, x2], [y1, y2], c='grey', linewidth=1.5, zorder=2, alpha=0.3, linestyle='--')
    
    # Set figure properties
    ax.set_xlabel('Relative X Coordinate (Normalized)', fontsize=12)
    ax.set_ylabel('Relative Y Coordinate (Normalized)', fontsize=12)
    ax.set_title(f'Lane Position and Direction Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrow
    legend_elements = [
        Line2D([0], [0], color='red', marker='X', linestyle='', 
               markersize=12, markeredgewidth=2, markeredgecolor='darkred', label='Junction Center'),
        FancyArrow(0, 0, 0.1, 0, width=0.02, facecolor='blue', edgecolor='blue', label='Lane Position'),
        FancyArrow(0, 0, 0.1, 0, width=0.02, facecolor='green', edgecolor='green', label='Lane Direction'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Visualization saved to: {save_path}')
    else:
        plt.show()
    
    plt.close()

