'''
@Author: WANG Maonan
@Date: 2023-09-08 15:57:34
@Description: 测试 TSC Env 环境
LastEditTime: 2026-02-12 17:50:47
'''
import numpy as np
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from stable_baselines3.common.env_checker import check_env
from utils.make_tsc_env import make_tsc_env

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    sumo_cfg = path_convert("./exp_junction/synthetic_junction/single_junction.sumocfg")
    net_file = path_convert("./exp_junction/synthetic_junction/env/single_junction.net.xml")
    log_path = path_convert('./log/')
    tsc_env_generate = make_tsc_env(
        tls_id='J1',
        num_phases=4,
        sumo_cfg=sumo_cfg, 
        net_file=net_file,
        num_seconds=600,
        use_gui=True,
        log_file=log_path,
        env_index=0,
    )
    tsc_env = tsc_env_generate()

    # Check Env
    # print(tsc_env.observation_space.sample())
    # print(tsc_env.action_space.n)
    # check_env(tsc_env)

    # Simulation with environment
    dones = False
    step_time = 0
    tsc_env.reset()
    while not dones:
        action = np.random.randint(4)
        states, rewards, truncated, dones, infos = tsc_env.step(action=action)
        logger.info(f"SIM: {infos['step_time']} \n+State:{states}; \n+Reward:{rewards}.")
        # tsc_env.env.visualize_congestion(save_path=f'./congestion_{step_time}.png', metric='vehicle_count', show_values=False)
        step_time += 1
    tsc_env.close()