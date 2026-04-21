'''
Author: WANG Maonan
Date: 2026-04-14 11:23:45
LastEditTime: 2026-04-14 11:28:28
LastEditors: WANG Maonan
Description: FixTime 固定配时算法 - 按顺序轮流切换相位
'''
from typing import Dict, Any, List
from tsc_algos.traditional.base_traditional import BaseTraditionalAgent

class FixTimeAgent(BaseTraditionalAgent):
    """固定配时算法

    按顺序循环切换相位: 0 -> 1 -> 2 -> 3 -> 0 -> ...
    支持自定义每个相位的持续步数，通过 phase_durations 控制。
    例如 phase_durations=[2, 3, 1, 4] 表示
    - 相位0持续2步，相位1持续3步，
    - 相位2持续1步，
    - 相位3持续4步，然后循环。
    如果不指定 phase_durations，则每个相位持续1步。
    """
    def __init__(self, num_phases: int = 4, phase_durations: list = None):
        self.num_phases = num_phases
        # phase_durations[i] 表示相位 i 持续多少步；None 时默认为 1
        self.phase_durations = phase_durations or [1] * num_phases
        self.current_phase = 0
        self.current_phase_step = 0  # 当前相位已持续步数

    def choose_action(
        self,
        lane_dynamic_features_seq: List[Dict[str, Any]],
        static_lane_features: Dict[str, Any],
    ) -> int:
        action = self.current_phase
        self.current_phase_step += 1
        # 达到当前相位的持续时间后，切换到下一相位
        if self.current_phase_step >= self.phase_durations[self.current_phase]:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.current_phase_step = 0
        return action
