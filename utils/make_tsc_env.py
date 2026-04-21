'''
@Author: WANG Maonan
@Date: 2023-09-08 17:45:54
@Description: 创建 TSC Env + Wrapper
+ 使用 TSCInfoWrapper 提取静态和动态信息
+ 使用 TSCRLWrapper 转换为 RL 训练格式
LastEditTime: 2026-04-13 17:27:10
'''
import gymnasium as gym
from utils.base_tsc_env import TSCEnvironment
from utils.tsc_info_wrapper import TSCInfoWrapper
from utils.tsc_rl_wrapper import TSCRLWrapper
from stable_baselines3.common.monitor import Monitor

def make_tsc_env(
        tls_id: str,
        num_seconds: int,
        num_phases: int,
        sumo_cfg: str,
        net_file: str,
        use_gui: bool,
        log_file: str, 
        env_index: int,
        cell_length: float = 15.0,
        ):
    """创建 TSC 环境
    
    Args:
        tls_id: 交通信号灯 ID
        num_seconds: 仿真时长（秒）
        num_phases: 相位数量
        sumo_cfg: SUMO 配置文件路径
        net_file: SUMO 网络文件路径
        use_gui: 是否使用 GUI
        log_file: 日志文件路径
        env_index: 环境索引
        cell_length: Cell 长度（米）
        reward_type: 奖励类型 ('queue_length', 'waiting_time', 'pressure', 'throughput')
        
    Returns:
        _init: 环境初始化函数
    """
    def _init() -> gym.Env: 
        # 创建基础 TSC 环境
        tsc_scenario = TSCEnvironment(
            sumo_cfg=sumo_cfg, 
            net_file=net_file,
            num_seconds=num_seconds,
            tls_ids=[tls_id], # 单路口
            use_gui=use_gui,
        )

        tsc_wrapper = TSCInfoWrapper(
            env=tsc_scenario, 
            tls_id=tls_id,
            cell_length=cell_length,
        )

        # 使用 RL Wrapper
        tsc_wrapper = TSCRLWrapper(
            env=tsc_wrapper, 
            tls_id=tls_id,
            num_phases=num_phases,
        )
            
        # 使用 Monitor 记录训练过程
        return Monitor(tsc_wrapper, filename=f'{log_file}/{env_index}')
    
    return _init