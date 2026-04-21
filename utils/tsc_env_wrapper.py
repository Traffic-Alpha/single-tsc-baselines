'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: 处理 TSCHub ENV 中的 state, reward
+ state: 
+ reward: max pressure of the junction
LastEditTime: 2026-02-12 16:50:40
'''
import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict

from .statistic_state_tools import (
    calculate_normalized_length,
    calculate_normalized_position,
    angle_to_vector,
    visualize_lane_features,
    build_lane_turn_function_mapping,
    build_lane_phase_binding_mapping
)
from .dynamic_state_tools import (
    LaneCellManager,
    format_lane_features_to_array,
    create_lane_cell_mask,
    visualize_lane_congestion,
    visualize_multiple_metrics
)


class TSCEnvWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction with tls_id
    """
    def __init__(self, 
        env: Env, tls_id:str, 
        cell_length:float=15.0
    ) -> None:
        super().__init__(env)
        self.tls_id = tls_id # 单路口的 id
        self.cell_length = cell_length  # 每个 cell 的固定长度（米）
        self.steps = 0 # 记录仿真步数
        
        # 静态特征
        self.static_lane_features = None  # 存储车道静态特征
        
        # 动态特征管理
        self.lane_cell_manager = None  # Lane Cell 管理器
        self.lane_dynamic_features = None  # 存储当前 lane 的动态特征
        self.lane_order = None  # lane 的排序，确保每次输出顺序一致
    
    # #######################################
    # Space for RL Training
    # #######################################
    # @property
    # def action_space(self):
    #     return gym.spaces.Discrete(4)
    
    # @property
    # def observation_space(self):
    #     obs_space = gym.spaces.Box(
    #         low=np.zeros((5,12)),
    #         high=np.ones((5,12)),
    #         shape=(5,12)
    #     ) # self.states 是一个时间序列
    #     return obs_space
    
    # #######################################
    # Wrapper
    # #######################################
    def state_wrapper(self, state):
        """返回当前每个 movement 的 occupancy
        """
        pass
    
    def reward_wrapper(self, states) -> float:
        """计算 Max-Pressure
        """
        pass
    
    def info_wrapper(self, infos, occupancy):
        pass

    def __statistic_state(self, state):
        """道路的静态信息, 每一行为 lane 记录 lane 的信息, 包括:
        1. I/O Type, 进出属性 (One-hot), 2 维
            1.1 [1, 0] : Incoming (进口道，车辆进入路口) 
            1.2 [0, 1] : Outgoing (出口道，车辆离开路口)
        2. Turn Function (One-hot), 3 维
            2.1 [1, 0, 0] : Straight (直行)
            2.2 [0, 1, 0] : Left (左转)
            2.3 [0, 0, 1] : Right (右转)
            2.4 [0, 0, 0] : Other (Outgoing)
        3. Phase Binding, Multi-hot 编码 (假设共有 4 个相位)
            Incoming Lane: 如果属于相位 1 和 3, 则对应位为 1, 否则为 0。例如 [0, 1, 0, 1]
            Outgoing Lane: 全 0 (不受灯控)。
        4. Lane Length, 车道长度 (除以 100, 归一化), 1 维
        5. Lane Position, 以路口中心为原点, 记录 lane 的最后一个坐标 (除以 100, 归一化), 2 维
        6. Heading Vector, 车道在出口处的切线方向, 2 维
        """
        # 获得路口中心坐标
        junction_center = state['node'][self.tls_id]['node_coord']
        # 获得 in roads
        in_roads = state['tls'][self.tls_id]['in_roads']
        in_roads_heading = state['tls'][self.tls_id]['in_roads_heading']
        # 获得 out roads
        out_roads = state['tls'][self.tls_id]['out_roads']
        out_roads_heading = state['tls'][self.tls_id]['out_roads_heading']
        # 获得每个 movement 包含的 lanes
        movement_lanes = state['tls'][self.tls_id]['movement_lane_ids']
        phase2movement = state['tls'][self.tls_id]['phase2movements']

        # 提前构建 lane 到转向功能和相位绑定的映射（提高效率）
        lane_to_turn = build_lane_turn_function_mapping(movement_lanes)
        lane_to_phases = build_lane_phase_binding_mapping(movement_lanes, phase2movement)

        # 存储每个 lane 的静态特征
        lane_features = {}
        
        for lane_id, lane_info in state['lane'].items():
            _road_id = lane_id.split('_')[0] # 获得 lane 所属的 road

            # 去掉不属于控制信号灯周围的 lane
            if _road_id not in in_roads and _road_id not in out_roads:
                continue
            
            features = {}
            
            # 1. I/O Type
            if _road_id in in_roads:
                features['io_type'] = [1, 0]
            elif _road_id in out_roads:
                features['io_type'] = [0, 1]
            else:
                features['io_type'] = [0, 0]  # 其它未识别

            # 2. Turn Function (使用预计算的映射)
            features['turn_function'] = lane_to_turn.get(lane_id, [0, 0, 0])

            # 3. Phase Binding (使用预计算的映射)
            features['phase_binding'] = lane_to_phases.get(lane_id, [0] * len(phase2movement))

            # 4. Lane Length (归一化)
            features['length'] = calculate_normalized_length(lane_info['length'])
            
            # 5. Lane Position (相对于路口中心的归一化位置)
            features['position'] = calculate_normalized_position(
                lane_info['shape'], 
                junction_center
            )

            # 6. Heading Vector (车道方向向量), 这里 90-angle 是因为角度系统不一样
            # | 角度系统 | 0度在哪里 | 旋转方向 |
            # | 数学/三角函数 | 3点钟 (右) | 逆时针 |
            # | 时钟/导航 | 12点钟 (上) | 顺时针 |
            if _road_id in in_roads:
                angle = 90 - in_roads_heading[_road_id]
                features['heading'] = angle_to_vector(angle, is_degrees=True)
            elif _road_id in out_roads:
                angle = 90 - out_roads_heading[_road_id]
                features['heading'] = angle_to_vector(angle, is_degrees=True)
            else:
                features['heading'] = [0, 0]
            
            lane_features[lane_id] = features
        
        return lane_features

    # #######################################
    # Reset and Step
    # #######################################
    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state =  self.env.reset()
        self.steps = 0
        
        # 1. 初始化路口静态信息
        self.static_lane_features = self.__statistic_state(state) # 计算车道静态特征
        self.visualize_static_features(save_path='./static_lane_features.png')

        # 2. 初始化动态信息管理器
        self.lane_cell_manager = LaneCellManager(
            static_lane_features=self.static_lane_features,
            cell_length=self.cell_length,
        ) # self.lane_cell_manager.lane_cells_info 查看信息
        
        # 确定 lane 的排序（incoming lanes 在前，outgoing lanes 在后）, 确保最后的 state 顺序一致
        incoming_lanes = [
            lane_id for lane_id, features in self.static_lane_features.items()
            if features['io_type'][0] == 1
        ]
        outgoing_lanes = [
            lane_id for lane_id, features in self.static_lane_features.items()
            if features['io_type'][1] == 1
        ]
        self.lane_order = sorted(incoming_lanes) + sorted(outgoing_lanes)
        
        # 3. 计算初始的 lane 动态特征
        phase_index_now = state['tls'][self.tls_id]['this_phase_index']
        self.lane_dynamic_features = self.lane_cell_manager.calculate_lane_dynamic_features(
            vehicles_state=state['vehicle'],
            current_phase_index=phase_index_now
        )
        
        return state, {'step_time':0}
    

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        can_perform_action = False
        while not can_perform_action:
            action = {self.tls_id: action} # 构建单路口 action 的动作
            states, rewards, truncated, dones, infos = super().step(action) # 与环境交互
            self.steps += 1
            can_perform_action = states['tls'][self.tls_id]['can_perform_action'] # 是否可以执行动作

            # 更新 lane 动态特征 (根据车辆信息和当前相位)
            phase_index_now = states['tls'][self.tls_id]['this_phase_index']
            self.lane_dynamic_features = self.lane_cell_manager.calculate_lane_dynamic_features(
                vehicles_state=states['vehicle'],
                current_phase_index=phase_index_now
            )

            # self.visualize_congestion(save_path=f'./congestion_{self.steps}.png', metric='vehicle_count', show_values=False)
        
        # 将 lane 动态特征添加到 infos 中
        infos['lane_summary'] = self.lane_cell_manager.get_lane_summary(self.lane_dynamic_features)
        
        return self.lane_dynamic_features, rewards, truncated, dones, infos
    
    def close(self) -> None:
        return super().close()
    

    # #######################################
    # Tools for DEBUG --> Reset() 之后才能使用
    # #######################################
    def visualize_static_features(
        self, 
        save_path: str = None, 
        figsize: Tuple[int, int] = (12, 12),
        arrow_scale: float = 0.3
    ) -> None:
        """Visualize lane static features
        
        Call reset() before this method to initialize static features.
        Plot lanes as arrows starting from their positions:
        - Incoming lanes (far from center): arrows point towards junction
        - Outgoing lanes (close to center): arrows point away from junction
        
        Args:
            save_path: Path to save the figure, if None, display the figure
            figsize: Figure size
            arrow_scale: Scale factor for arrows (default 0.3)
        """
        if self.static_lane_features is None:
            raise ValueError("Static features not initialized, please call reset() first")
        
        visualize_lane_features(
            static_lane_features=self.static_lane_features,
            save_path=save_path,
            figsize=figsize,
            arrow_scale=arrow_scale,
        )
    
    def get_lane_dynamic_features_array(self) -> np.ndarray:
        """获取 lane 动态特征的数组表示
        
        Returns:
            feature_array: shape (num_lanes, num_cells, 6)
                特征维度: [vehicle_count, avg_speed, avg_waiting_time, 
                          avg_accumulated_waiting_time, occupancy, is_passable]
        """
        if self.lane_dynamic_features is None:
            raise ValueError("Dynamic features not initialized, please call reset() first")
        
        return format_lane_features_to_array(
            self.lane_dynamic_features, 
            lane_order=self.lane_order
        )
    
    def get_lane_summary(self) -> Dict[str, Dict]:
        """获取每条 lane 的汇总统计信息
        
        Returns:
            lane_summary: 每条 lane 的汇总信息
        """
        if self.lane_cell_manager is None or self.lane_dynamic_features is None:
            raise ValueError("Dynamic features not initialized, please call reset() first")
        
        return self.lane_cell_manager.get_lane_summary(self.lane_dynamic_features)
    
    def get_lane_cell_info(self) -> Dict[str, Dict]:
        """获取每条 lane 的 cell 划分信息
        
        Returns:
            lane_cell_info: 每条 lane 的 cell 划分信息
                {
                    'lane_id': {
                        'length': lane 长度（米）,
                        'num_cells': cell 数量,
                        'cell_length': 固定的 cell 长度（米）,
                        'last_cell_length': 最后一个 cell 的实际长度（米）
                    }
                }
        """
        if self.lane_cell_manager is None:
            raise ValueError("Cell manager not initialized, please call reset() first")
        
        lane_cell_info = {}
        for lane_id, info in self.lane_cell_manager.lane_cells_info.items():
            cell_boundaries = info['cell_boundaries']
            last_cell_length = cell_boundaries[-1] - cell_boundaries[-2]
            
            lane_cell_info[lane_id] = {
                'length': info['length'],
                'num_cells': info['num_cells'],
                'cell_length': self.cell_length,
                'last_cell_length': last_cell_length
            }
        
        return lane_cell_info
    
    def get_lane_cell_mask(self) -> np.ndarray:
        """获取 lane cell 的 mask，标记哪些 cell 是有效的（不是 padding）
        
        Returns:
            mask: shape (num_lanes, max_cells)，True 表示有效的 cell，False 表示 padding
        """
        if self.lane_cell_manager is None:
            raise ValueError("Cell manager not initialized, please call reset() first")
        
        return create_lane_cell_mask(
            lane_cells_info=self.lane_cell_manager.lane_cells_info,
            lane_order=self.lane_order
        )
    
    def visualize_congestion(
        self,
        metric: str = 'occupancy',
        save_path: str = None,
        figsize: Tuple[int, int] = (16, 10),
        title: str = None,
        show_values: bool = True
    ) -> None:
        """可视化 lane cell 的拥堵情况
        
        Args:
            metric: 要可视化的指标 ('occupancy', 'vehicle_count', 'avg_speed', 'avg_waiting_time')
            save_path: 保存图片的路径，如果为 None 则显示图片
            figsize: 图片大小
            title: 图片标题
            show_values: 是否在 cell 上显示数值
        """
        if self.lane_dynamic_features is None:
            raise ValueError("Dynamic features not initialized, please call reset() first")
        
        visualize_lane_congestion(
            lane_dynamic_features=self.lane_dynamic_features,
            lane_order=self.lane_order,
            metric=metric,
            save_path=save_path,
            figsize=figsize,
            title=title,
            show_values=show_values
        )
    
    def visualize_all_metrics(
        self,
        save_path: str = None,
        figsize: Tuple[int, int] = (20, 12),
        title: str = None
    ) -> None:
        """同时可视化多个指标的 lane cell 拥堵情况
        
        Args:
            save_path: 保存图片的路径，如果为 None 则显示图片
            figsize: 图片大小
            title: 图片标题
        """
        if self.lane_dynamic_features is None:
            raise ValueError("Dynamic features not initialized, please call reset() first")
        
        visualize_multiple_metrics(
            lane_dynamic_features=self.lane_dynamic_features,
            lane_order=self.lane_order,
            save_path=save_path,
            figsize=figsize,
            title=title
        )