'''
@Author: WANG Maonan
@Date: 2023-09-08 15:48:26
@Description: 基于 Stabe Baseline3 来控制单路口
+ State Design: Last step occupancy for each movement
+ Action Design: Choose Next Phase 
+ Reward Design: Total Waiting Time
LastEditTime: 2026-02-12 18:08:13
'''
import os
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from utils.make_tsc_env import make_tsc_env
from utils.sb3_utils.linear_schedule import linear_schedule
from utils.sb3_utils.vec_normalize import VecNormalizeCallback
from utils.sb3_utils.junction_models import TransformerJunctionModel

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level="INFO")

if __name__ == '__main__':
    log_path = path_convert('./log/')
    model_path = path_convert('./models/')
    tensorboard_path = path_convert('./tensorboard/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    
    # #########
    # Init Env
    # #########
    sumo_cfg = path_convert("./exp_junction/synthetic_junction/single_junction.sumocfg")
    net_file = path_convert("./exp_junction/synthetic_junction/env/single_junction.net.xml")
    params = {
        'tls_id':'J1',
        'num_seconds':600,
        'num_phases':4,
        'sumo_cfg':sumo_cfg,
        'net_file':net_file,
        'use_gui':False,
        'log_file':log_path,
    }
    env = SubprocVecEnv([make_tsc_env(env_index=f'{i}', **params) for i in range(8)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # #########
    # Callback
    # #########
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # 多少个 step, 需要根据与环境的交互来决定
        save_path=model_path,
    )
    vec_normalize_callback = VecNormalizeCallback(
        save_freq=10000,
        save_path=model_path,
    ) # 保存环境参数
    callback_list = CallbackList([checkpoint_callback, vec_normalize_callback])

    # #########
    # Training
    # #########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Transformer 模型配置
    policy_kwargs = dict(
        features_extractor_class=TransformerJunctionModel,
        features_extractor_kwargs=dict(
            features_dim=128,        # 输出特征维度（传给 actor-critic heads）
            num_heads=4,             # 注意力头数（建议 2/4/8）
            num_layers=2,            # Transformer 层数（1-3 层）
            dim_feedforward=256,     # FFN 隐藏层维度（通常是 d_model 的 2-4 倍）
            dropout=0.1,             # Dropout 概率
        ),
    )
    
    logger.info("Model Configuration:")
    logger.info(f"  - Features dim: {policy_kwargs['features_extractor_kwargs']['features_dim']}")
    logger.info(f"  - Num heads: {policy_kwargs['features_extractor_kwargs']['num_heads']}")
    logger.info(f"  - Num layers: {policy_kwargs['features_extractor_kwargs']['num_layers']}")
    logger.info(f"  - FFN dim: {policy_kwargs['features_extractor_kwargs']['dim_feedforward']}")
    
    model = PPO(
                "MlpPolicy", 
                env, 
                batch_size=128,      # 增大 batch size 有利于 Transformer 训练
                n_steps=512,         # 增大 n_steps 收集更多经验
                n_epochs=10,         # 增加训练轮数
                learning_rate=linear_schedule(3e-4),  # 降低学习率，Transformer 需要较小学习率
                gamma=0.99,          # 折扣因子
                gae_lambda=0.95,     # GAE lambda
                clip_range=0.2,      # PPO clip range
                ent_coef=0.01,       # 熵系数，鼓励探索
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_path, 
                device=device
            )
    model.learn(total_timesteps=3e5, tb_log_name='J1', callback=callback_list)
    
    # #################
    # 保存 model 和 env
    # #################
    env.save(f'{model_path}/last_vec_normalize.pkl')
    model.save(f'{model_path}/last_rl_model.zip')
    print('训练结束, 达到最大步数.')

    env.close()