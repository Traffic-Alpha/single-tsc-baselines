'''
@Author: WANG Maonan
@Date: 2026-02-12
@Description: TSC RL Wrapper - 将 TSCInfoWrapper 的输出转换为适合 RL 训练的格式
+ 将字典格式的 observation 转换为数组格式
+ 定义符合 rl 的 observation_space 和 action_space
+ 提取单路口的 reward
LastEditTime: 2026-02-12 18:07:32
'''
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Env
from typing import Any, Tuple, Dict


class TSCRLWrapper(gym.Wrapper):
    """TSC RL Wrapper
    
    该 Wrapper 负责：
    1. 将 TSCInfoWrapper 的字典输出转换为数组格式
    2. 定义 observation_space 和 action_space
    3. 提取单路口的 reward 值
    4. 确保返回格式符合 Gym 标准
    """
    def __init__(self, env: Env, tls_id: str, num_phases: int) -> None:
        """初始化 TSC RL Wrapper
        
        Args:
            env: TSCInfoWrapper 环境
            tls_id: 交通信号灯 ID
            num_phases: 相位数量
        """
        super().__init__(env)
                
        # 存储路口 ID
        self.tls_id = tls_id
        
        # 相位数量
        self.num_phases = num_phases
        self._action_space = spaces.Discrete(num_phases)
        self._observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(20,10),
            dtype=np.float32
        )
        
    @property
    def observation_space(self) -> spaces.Space:
        """获取观测空间
        
        Returns:
            observation_space: Box 空间，shape 为 (num_lanes, max_cells, 6)
        """
        if self._observation_space is None:
            raise ValueError(
                "Observation space not initialized. Call reset() first."
            )
        return self._observation_space
    
    @property
    def action_space(self) -> spaces.Space:
        """获取动作空间
        
        Returns:
            action_space: Discrete 空间，大小为相位数量
        """
        if self._action_space is None:
            raise ValueError(
                "Action space not initialized. Call reset() first."
            )
        return self._action_space
    
    def reset(
        self, 
        seed: int = None, 
        options: Dict = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境
        
        Args:
            seed: 随机种子
            options: 其他选项
            
        Returns:
            obs_array: 观测数组，shape 为 (num_lanes, feature_dim)
            info: 附加信息
        """
        # 调用父类 reset，返回 lane_dynamic_features 字典和 info
        lane_dynamic_features, info = self.env.reset(seed=seed)
        
        # 转换为 lane 级别的特征向量
        obs_array = self._aggregate_lane_features(lane_dynamic_features)
        
        return obs_array.astype(np.float32), info
    
    def step(
        self, 
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步环境交互
        
        Args:
            action: 动作（相位索引）
            
        Returns:
            obs_array: 观测数组，shape 为 (num_lanes, feature_dim)
            reward: 奖励值（标量）
            truncated: 是否截断
            done: 是否结束
            info: 附加信息
        """
        # 调用父类 step，返回 lane_dynamic_features 字典
        lane_dynamic_features, rewards, truncated, done, info = self.env.step(action)
        
        # 转换为 lane 级别的特征向量
        obs_array = self._aggregate_lane_features(lane_dynamic_features)
        
        # 计算 pressure 作为奖励
        reward = self._calculate_pressure_reward(lane_dynamic_features)
        
        return obs_array.astype(np.float32), reward, truncated, done, info
    
    # #######################################
    # State & Reward Functions
    # #######################################
    def _aggregate_lane_features(self, lane_dynamic_features: Dict[str, Any]) -> np.ndarray:
        """将 lane 的 cell 级别特征聚合为 lane 级别特征
        
        每个 lane 的特征包括：
        1. 等待车辆数（归一化）
        2. 行驶车辆数（归一化）
        3. 平均速度（归一化）
        4. 平均等待时间（归一化）
        5. 占用率（归一化）
        6. 是否绿灯（0/1）
        7. Phase binding（multi-hot，num_phases 维）
        
        Args:
            lane_dynamic_features: 每条 lane 的 cell 动态特征
            
        Returns:
            feature_array: shape (num_lanes, feature_dim)
                feature_dim = 6 + num_phases
        """
        # 获取静态特征
        static_features = self.env.get_static_features()
        
        # 获取 lane 顺序（确保顺序一致）
        lane_order = self.env.lane_order
        
        # 归一化参数
        MAX_VEHICLES = 50.0  # 假设单条 lane 最多 50 辆车
        MAX_SPEED = 15.0     # m/s (约 54 km/h)
        MAX_WAITING_TIME = 300.0  # 秒
        SPEED_THRESHOLD = 1.0  # m/s，低于此速度视为等待
        
        # 特征维度：6 基础特征 + num_phases（phase binding）
        feature_dim = 6 + self.num_phases
        num_lanes = len(lane_order)
        
        # 初始化特征数组
        feature_array = np.zeros((num_lanes, feature_dim), dtype=np.float32)
        
        for lane_idx, lane_id in enumerate(lane_order):
            if lane_id not in lane_dynamic_features:
                continue
            
            cells = lane_dynamic_features[lane_id]
            
            # 聚合 cell 级别的特征
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
                avg_waiting_time = cell['avg_waiting_time']
                occupancy = cell['occupancy']
                is_passable = cell.get('is_passable', 0)
                
                total_vehicles += vehicle_count
                total_speed += avg_speed * vehicle_count  # 加权平均
                total_waiting_time += avg_waiting_time * vehicle_count  # 加权平均
                total_occupancy += occupancy
                
                # 判断等待车辆和行驶车辆
                if avg_speed < SPEED_THRESHOLD and vehicle_count > 0:
                    waiting_vehicles += vehicle_count
                elif avg_speed >= SPEED_THRESHOLD and vehicle_count > 0:
                    moving_vehicles += vehicle_count
                
                # 如果任意 cell 可通行，则认为是绿灯
                if is_passable > 0:
                    is_green = 1
            
            # 计算平均值
            avg_speed_lane = total_speed / total_vehicles if total_vehicles > 0 else 0.0
            avg_waiting_time_lane = total_waiting_time / total_vehicles if total_vehicles > 0 else 0.0
            avg_occupancy = total_occupancy / num_cells if num_cells > 0 else 0.0
            
            # 归一化特征
            feature_array[lane_idx, 0] = min(waiting_vehicles / MAX_VEHICLES, 1.0)
            feature_array[lane_idx, 1] = min(moving_vehicles / MAX_VEHICLES, 1.0)
            feature_array[lane_idx, 2] = min(avg_speed_lane / MAX_SPEED, 1.0)
            feature_array[lane_idx, 3] = min(avg_waiting_time_lane / MAX_WAITING_TIME, 1.0)
            feature_array[lane_idx, 4] = min(avg_occupancy, 1.0)
            feature_array[lane_idx, 5] = is_green
            
            # 添加 phase binding 信息
            if lane_id in static_features:
                phase_binding = static_features[lane_id]['phase_binding']
                feature_array[lane_idx, 6:6+len(phase_binding)] = phase_binding
        
        return feature_array
    
    def _calculate_pressure_reward(self, lane_dynamic_features: Dict[str, Any]) -> float:
        """计算基于 pressure 的奖励
        
        Pressure 定义为：进口道排队车辆数 - 出口道排队车辆数
        排队车辆：速度 < 1.0 m/s 的车辆
        奖励为负的 pressure（pressure 越小，奖励越高）
        
        Args:
            lane_dynamic_features: 每条 lane 的动态特征
            
        Returns:
            reward: 负的 pressure 值
        """
        # 获取静态特征来区分进口道和出口道
        static_features = self.env.get_static_features()
        
        # 速度阈值，低于此速度视为排队等待
        SPEED_THRESHOLD = 1.0  # m/s
        
        incoming_waiting_vehicles = 0
        outgoing_waiting_vehicles = 0
        
        # 遍历所有 lane
        for lane_id, cells in lane_dynamic_features.items():
            # 获取该 lane 的静态特征
            if lane_id not in static_features:
                continue
            
            io_type = static_features[lane_id]['io_type']
            
            # 计算该 lane 的排队等待车辆数
            waiting_vehicles = 0
            for cell in cells:
                vehicle_count = cell['vehicle_count']
                avg_speed = cell['avg_speed']
                
                # 只统计低速（排队）的车辆
                if avg_speed < SPEED_THRESHOLD and vehicle_count > 0:
                    waiting_vehicles += vehicle_count
            
            # 根据 I/O 类型累加
            if io_type[0] == 1:  # Incoming lane
                incoming_waiting_vehicles += waiting_vehicles
            elif io_type[1] == 1:  # Outgoing lane
                outgoing_waiting_vehicles += waiting_vehicles
        
        # 计算 pressure（基于排队车辆）
        pressure = incoming_waiting_vehicles - outgoing_waiting_vehicles
        
        # 返回负的 pressure 作为奖励（pressure 越小越好）
        reward = -pressure
        
        return float(reward)
    
    def close(self) -> None:
        """关闭环境"""
        return super().close()

