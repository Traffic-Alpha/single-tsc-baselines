'''
@Author: WANG Maonan
@Description: TSC RL Wrapper - 可配置的 RL 环境封装
+ 接受可插拔的 reward_fn 和 obs_fn
+ 根据 action_type 定义 action space
+ obs_fn / reward_fn 均接收完整特征序列 List[Dict]
'''
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Env
from typing import Any, Tuple, Dict, Callable


class TSCRLWrapper(gym.Wrapper):
    """TSC RL Wrapper - 将 TSCInfoWrapper 的输出转换为 RL 训练格式

    通过 reward_fn, obs_fn, action_type 三个参数实现可配置:
    - reward_fn: (lane_dynamic_features_seq, static_lane_features) -> float
    - obs_fn: (lane_dynamic_features_seq, static_lane_features, lane_order, num_phases) -> np.ndarray
    - action_type: "choose_next_phase" -> Discrete(num_phases), "next_or_not" -> Discrete(2)
    """
    def __init__(self,
        env: Env,
        reward_fn: Callable,
        obs_fn: Callable,
        obs_space: spaces.Space,
        action_type: str = "choose_next_phase",
        num_phases: int = 4,
    ) -> None:
        """初始化 TSC RL Wrapper

        Args:
            env: TSCInfoWrapper 环境
            reward_fn: 奖励函数
            obs_fn: 观测构造函数
            obs_space: 观测空间 (gymnasium.spaces.Space)
            action_type: 动作类型
            num_phases: 相位数量
        """
        super().__init__(env)

        self.reward_fn = reward_fn
        self.obs_fn = obs_fn
        self.num_phases = num_phases

        # action_type 决定 action space
        if action_type == "choose_next_phase":
            self._action_space = spaces.Discrete(num_phases)
        elif action_type == "next_or_not":
            self._action_space = spaces.Discrete(2)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        # obs_space 由调用方传入
        self._observation_space = obs_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        lane_dynamic_features_seq, info = self.env.reset(seed=seed)

        obs = self.obs_fn(
            lane_dynamic_features_seq,
            self.env.static_lane_features,
            self.env.lane_order,
            self.num_phases,
        )
        return obs.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        lane_dynamic_features_seq, rewards, truncated, done, info = self.env.step(action)

        obs = self.obs_fn(
            lane_dynamic_features_seq,
            self.env.static_lane_features,
            self.env.lane_order,
            self.num_phases,
        )
        reward = self.reward_fn(
            lane_dynamic_features_seq,
            self.env.static_lane_features,
        )
        return obs.astype(np.float32), reward, truncated, done, info

    def close(self) -> None:
        return super().close()
