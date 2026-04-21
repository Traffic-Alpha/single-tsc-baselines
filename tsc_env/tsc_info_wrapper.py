'''
@Author: WANG Maonan
@Description: TSC Info Wrapper - 精简版，只保留核心 wrapper 逻辑
+ reset 时提取静态特征、初始化动态管理器
+ step 时收集完整子步序列，返回 List[Dict]
LastEditTime: 2026-04-14 21:24:29
'''
import gymnasium as gym
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict, List

from .dynamic_tools import LaneCellManager
from .static_tools import extract_static_features



class TSCInfoWrapper(gym.Wrapper):
    """TSC Info Wrapper - 从环境中提取静态和动态信息

    公开属性（外部可直接访问）:
    - static_lane_features: 车道静态特征字典
    - lane_dynamic_features: 最后一个子步的动态特征（便捷属性）
    - lane_dynamic_features_seq: 当前决策间隔内所有子步的特征序列
    - lane_cell_manager: LaneCellManager 实例
    - lane_order: lane 排序列表

    step() / reset() 均返回完整序列 List[Dict]，算法自行决定如何使用。
    """
    def __init__(self,
        env: Env,
        tls_id: str,
        cell_length: float = 15.0
    ) -> None:
        super().__init__(env)
        self.tls_id = tls_id
        self.cell_length = cell_length
        self.steps = 0

        self.static_lane_features = None
        self.lane_cell_manager = None
        self.lane_dynamic_features = None
        self.lane_dynamic_features_seq = None
        self.lane_order = None

    def _build_lane_order(self):
        """确定 lane 排序: incoming lanes (sorted) + outgoing lanes (sorted)"""
        incoming = sorted(
            lid for lid, f in self.static_lane_features.items() if f['io_type'][0] == 1
        )
        outgoing = sorted(
            lid for lid, f in self.static_lane_features.items() if f['io_type'][1] == 1
        )
        return incoming + outgoing

    def _update_dynamic_features(self, state):
        """根据当前 state 计算一个子步的动态特征"""
        phase_index_now = state['tls'][self.tls_id]['this_phase_index']
        return self.lane_cell_manager.calculate_lane_dynamic_features(
            vehicles_state=state['vehicle'],
            current_phase_index=phase_index_now
        )

    def reset(self, seed=1) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """reset 时初始化静态信息和动态信息，返回长度为 1 的序列"""
        state = self.env.reset()
        self.steps = 0

        # 1. 提取静态特征（独立函数）
        self.static_lane_features = extract_static_features(state, self.tls_id)

        # 2. 初始化动态特征管理器
        self.lane_cell_manager = LaneCellManager(
            static_lane_features=self.static_lane_features,
            cell_length=self.cell_length,
        )

        # 3. 确定 lane 排序
        self.lane_order = self._build_lane_order()

        # 4. 计算初始动态特征，封装为序列
        initial_features = self._update_dynamic_features(state)
        self.lane_dynamic_features = initial_features
        self.lane_dynamic_features_seq = [initial_features]

        return self.lane_dynamic_features_seq, {'step_time': 0}

    def step(self, action: int) -> Tuple[List[Dict[str, Any]], SupportsFloat, bool, bool, Dict[str, Any]]:
        """执行一步环境交互，返回决策间隔内所有子步的特征序列"""
        can_perform_action = False
        features_seq = []
        while not can_perform_action:
            action_dict = {self.tls_id: action}
            states, rewards, truncated, dones, infos = super().step(action_dict)

            self.steps += 1
            can_perform_action = states['tls'][self.tls_id]['can_perform_action']
            features_seq.append(self._update_dynamic_features(states))

        self.lane_dynamic_features_seq = features_seq
        self.lane_dynamic_features = features_seq[-1]

        infos['step_time'] = self.steps
        return self.lane_dynamic_features_seq, rewards, truncated, dones, infos

    def close(self) -> None:
        return super().close()
