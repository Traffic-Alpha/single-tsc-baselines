'''
@Author: WANG Maonan
@Description: PPO + choose_next_phase 环境组装
'''
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from tsc_env import TSCEnvironment, TSCInfoWrapper, TSCRLWrapper
from tsc_env.reward_funcs import pressure_reward
from tsc_env.obs_funcs import lane_aggregate_obs, lane_aggregate_obs_space


def make_env(
    tls_id: str,
    num_seconds: int,
    num_phases: int,
    sumo_cfg: str,
    net_file: str,
    use_gui: bool = False,
    log_file: str = None,
    env_index: int = 0,
    cell_length: float = 15.0,
    trip_info: str = "",
    fcd_output: str = "",
):
    """创建 PPO + choose_next_phase 环境

    Pipeline: TSCEnvironment -> TSCInfoWrapper -> TSCRLWrapper -> Monitor
    """
    def _init() -> gym.Env:
        env = TSCEnvironment(
            sumo_cfg=sumo_cfg,
            net_file=net_file,
            num_seconds=num_seconds,
            tls_ids=[tls_id],
            tls_action_type="choose_next_phase",
            use_gui=use_gui,
            trip_info=trip_info,
            fcd_output=fcd_output,
        )
        env = TSCInfoWrapper(env, tls_id=tls_id, cell_length=cell_length)
        env = TSCRLWrapper(
            env,
            reward_fn=pressure_reward,
            obs_fn=lane_aggregate_obs,
            obs_space=lane_aggregate_obs_space(num_phases),
            action_type="choose_next_phase",
            num_phases=num_phases,
        )
        if log_file:
            env = Monitor(env, filename=f'{log_file}/{env_index}')
        return env

    return _init
