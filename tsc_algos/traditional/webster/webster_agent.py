'''
@Author: WANG Maonan
@Description: Webster 算法实现（比例计划式自适应定时控制）

# ============================================================
# 算法原理
# ============================================================
#
# 【原始 Webster 方法（交通工程理论）】
#
#   Webster 方法的核心目标是最小化交叉口的总延误。
#   对于 N 个相位的信号灯，定义：
#
#     q_i  = 相位 i 临界车道的到达流量（veh/s）
#     s_i  = 相位 i 临界车道的饱和流量（veh/s，由道路几何决定）
#     y_i  = q_i / s_i        （流量比，无量纲，反映该相位的需求强度）
#     Y    = Σ y_i             （总流量比，Y < 1 表示交叉口未饱和）
#     L    = N × l             （总损失时间，l 为每相位黄灯+全红时间）
#
#   最优周期长度（单位：秒）：
#     C* = (1.5 × L + 5) / (1 - Y)
#
#   各相位绿灯时间分配：
#     g_i = (C* - L) × (y_i / Y)
#
#   直觉：需求越大（y_i 越高）的相位，分配到的绿灯时间越长。
#
# ------------------------------------------------------------
#
# 【本实现的适配（仿真环境）】
#
#   在离散决策步的仿真环境中，无法直接使用以秒为单位的公式，
#   因此做如下简化，保留 Webster 的核心精神：
#
#   1. 流量观测：用仿真中临界进口车道的车辆数代替 q_i
#        q_i ≈ 一个周期内，相位 i 所有进口车道中，
#               车辆数最多的那条车道的均值（临界车道原则）
#
#   2. 跳过周期公式：直接指定目标周期步数 G（target_cycle_steps），
#      不依赖 s_i 和饱和流量参数，避免仿真中难以标定的参数。
#
#   3. 比例分配绿灯步数：
#        g_i = round( G × q_i / Σq_j )
#        g_i = clamp( g_i, min_green_steps, max_green_steps )
#
#   4. 生成执行计划（schedule）：
#        schedule = [0]*g_0 + [1]*g_1 + ... + [N-1]*g_{N-1}
#        例：g = [3, 5, 2, 4] -> [0,0,0, 1,1,1,1,1, 2,2, 3,3,3,3]
#
#   5. 自适应更新：每个周期结束时，用上一周期的观测重新计算 g_i，
#      生成新 schedule 供下一周期使用。
#
# ------------------------------------------------------------
#
# 【与 MaxPressure / SOTL 的本质区别】
#
#   MaxPressure / SOTL：响应式（reactive）
#     每一步根据当前交通状态实时竞争选择相位，类似"贪心算法"。
#
#   Webster（本实现）：计划式（pre-planned）
#     预先将绿灯时间按流量比分配给各相位，按顺序执行计划；
#     不做实时竞争，通过周期性更新计划来适应流量变化。
#
# ============================================================
'''
from typing import Dict, Any, List, Optional

from tsc_algos.traditional.base_traditional import BaseTraditionalAgent


class WebsterAgent(BaseTraditionalAgent):
    """Webster 比例计划式信号控制算法

    每个信号周期结束时，根据上一周期各相位临界车道的平均车辆数，
    按比例分配下一周期的绿灯决策步数，生成相位执行计划（schedule）。
    执行时按计划顺序服务各相位，周期结束后自适应更新计划。

    Args:
        num_phases: 相位数（None 则从 static_lane_features 自动推断）
        target_cycle_steps: 目标周期总决策步数
        min_green_steps: 每相位最小绿灯决策步数
        max_green_steps: 每相位最大绿灯决策步数
    """

    def __init__(
        self,
        num_phases: Optional[int] = None,
        target_cycle_steps: int = 20,
        min_green_steps: int = 2,
        max_green_steps: int = 12,
    ):
        self.num_phases = num_phases
        self.target_cycle_steps = target_cycle_steps
        self.min_green_steps = min_green_steps
        self.max_green_steps = max_green_steps

        self._schedule: List[int] = []
        self._step_idx: int = 0
        self._phase_flow_obs: List[List[float]] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def choose_action(
        self,
        lane_dynamic_features_seq: List[Dict[str, Any]],
        static_lane_features: Dict[str, Any],
    ) -> int:
        """按预计算的周期计划返回当前应执行的相位

        每步记录各相位临界流量；周期结束时按比例重新分配绿灯步数。

        Args:
            lane_dynamic_features_seq: 决策间隔内所有子步的特征序列
            static_lane_features: lane 静态特征

        Returns:
            action: 当前应执行的相位索引
        """
        lane_dynamic_features = lane_dynamic_features_seq[-1]

        # 初次调用：推断相位数并初始化等长计划
        if not self._schedule:
            self._init(static_lane_features)

        # 1. 记录各相位临界流量（供周期结束时计算比例用）
        phase_flows = self._compute_critical_flows(lane_dynamic_features, static_lane_features)
        for i, flow in enumerate(phase_flows):
            self._phase_flow_obs[i].append(flow)

        # 2. 从计划中读取当前应执行的相位
        action = self._schedule[self._step_idx]

        # 3. 推进索引；周期结束时重新分配绿灯步数并更新计划
        self._step_idx += 1 # 3.1 推进索引

        # 3.2 周期结束后重新计算相位计划
        if self._step_idx >= len(self._schedule):
            self._step_idx = 0
            new_steps = self._compute_proportional_allocation()
            self._schedule = self._build_schedule(new_steps)
            self._phase_flow_obs = [[] for _ in range(self.num_phases)]

        return action

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init(self, static_lane_features: Dict[str, Any]) -> None:
        """首次调用时推断相位数，初始化等长计划"""
        if self.num_phases is None:
            max_phases = max(
                (len(feat.get('phase_binding', [])) for feat in static_lane_features.values()),
                default=1,
            )
            self.num_phases = max_phases or 1

        self._phase_flow_obs = [[] for _ in range(self.num_phases)]
        equal_steps = max(self.min_green_steps, self.target_cycle_steps // self.num_phases)
        self._schedule = self._build_schedule([equal_steps] * self.num_phases)

    def _build_schedule(self, green_steps_per_phase: List[int]) -> List[int]:
        """按各相位绿灯步数构建扁平执行计划

        例: [3, 5, 2, 4] -> [0,0,0, 1,1,1,1,1, 2,2, 3,3,3,3]
        """
        schedule = []
        for phase_idx, steps in enumerate(green_steps_per_phase):
            schedule.extend([phase_idx] * steps)
        return schedule

    def _compute_critical_flows(
        self,
        lane_dynamic_features: Dict[str, Any],
        static_lane_features: Dict[str, Any],
    ) -> List[float]:
        """计算各相位临界车道的车辆数（取该相位所有进口车道中最大的一条）

        使用临界车道（max 而非 sum）更符合 Webster 理论：
        绿灯分配取决于最难服务的那条车道，而非所有车道之和。
        """
        critical = [0.0] * self.num_phases

        for lane_id, static_feat in static_lane_features.items():
            if static_feat['io_type'][0] != 1:  # 只看进口车道
                continue
            phase_binding = static_feat['phase_binding']
            lane_count = sum(
                cell['vehicle_count']
                for cell in lane_dynamic_features.get(lane_id, [])
            )
            for phase_idx in range(self.num_phases):
                if phase_idx < len(phase_binding) and phase_binding[phase_idx] == 1:
                    critical[phase_idx] = max(critical[phase_idx], lane_count)

        return critical

    def _compute_proportional_allocation(self) -> List[int]:
        """按上一周期的平均临界流量比例分配绿灯步数

        g_i = round(target_cycle_steps × q_i / Σq_j)
        约束到 [min_green_steps, max_green_steps]。
        若所有相位流量均为 0，退化为等长分配。
        """
        avg_flows = [
            sum(obs) / len(obs) if obs else 0.0
            for obs in self._phase_flow_obs
        ]

        total_flow = sum(avg_flows)

        if total_flow <= 0.0:
            equal = max(self.min_green_steps, self.target_cycle_steps // self.num_phases)
            return [equal] * self.num_phases

        green_steps = []
        for q in avg_flows:
            g_i = round(self.target_cycle_steps * q / total_flow)
            g_i = max(self.min_green_steps, min(self.max_green_steps, g_i))
            green_steps.append(g_i)

        return green_steps
