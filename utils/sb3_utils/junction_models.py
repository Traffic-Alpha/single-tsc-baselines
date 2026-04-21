'''
Author: WANG Maonan
Date: 2026-02-12 17:58:35
Description: 
LastEditTime: 2026-02-12 18:03:04
'''
'''
@Author: WANG Maonan
@Date: 2026-02-12
@Description: 基于 Transformer 的交通信号控制模型
+ 使用 Transformer Encoder 处理每个 lane 的特征
+ 添加可学习的 CLS token 提取全局信息
+ 适配 Stable-Baselines3 的特征提取器接口
LastEditTime: 2026-02-12 18:00:00
'''
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerJunctionModel(BaseFeaturesExtractor):
    """基于 Transformer 的路口模型
    
    架构：
    1. 输入 lane 级别的特征 (num_lanes, feature_dim)
    2. 线性投影到 embedding 空间
    3. 添加位置编码
    4. 添加可学习的 CLS token
    5. Transformer Encoder 处理
    6. 从 CLS token 提取全局特征
    7. MLP 映射到输出特征
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """初始化 Transformer 路口模型
        
        Args:
            observation_space: 观测空间 (num_lanes, lane_feature_dim)
            features_dim: 输出特征维度（传给后续策略网络）
            num_heads: Transformer 注意力头数
            num_layers: Transformer 层数
            dim_feedforward: Transformer FFN 维度
            dropout: Dropout 概率
        """
        # features_dim 是输出维度
        super().__init__(observation_space, features_dim)
        
        # 获取输入维度
        # observation_space.shape = (num_lanes, lane_feature_dim)
        self.num_lanes = observation_space.shape[0]
        self.lane_feature_dim = observation_space.shape[1]
        
        # Transformer 的隐藏维度（embedding 维度）
        self.d_model = features_dim
        
        # 输入投影：将 lane_feature_dim 映射到 d_model
        self.input_projection = nn.Linear(self.lane_feature_dim, self.d_model)
        
        # 可学习的 CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # 位置编码（可学习）
        # num_lanes + 1 是因为有 CLS token
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_lanes + 1, self.d_model)
        )
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # 输入格式 (batch, seq, feature)
            norm_first=True    # Pre-LN（更稳定）
        )
        
        # Transformer Encoder（堆叠多层）
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
        # 输出 MLP（从 CLS token 特征映射到最终特征）
        self.output_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, features_dim),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 使用 Xavier 初始化线性层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # CLS token 使用正态分布初始化
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 位置编码使用正态分布初始化
        nn.init.normal_(self.positional_encoding, std=0.02)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            observations: shape (batch_size, num_lanes, lane_feature_dim)
            
        Returns:
            features: shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        
        # 1. 输入投影：(batch, num_lanes, lane_feature_dim) -> (batch, num_lanes, d_model)
        lane_embeddings = self.input_projection(observations)
        
        # 2. 添加 CLS token：(batch, num_lanes, d_model) -> (batch, num_lanes+1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        embeddings = torch.cat([cls_tokens, lane_embeddings], dim=1)  # (batch, num_lanes+1, d_model)
        
        # 3. 添加位置编码
        embeddings = embeddings + self.positional_encoding
        
        # 4. Transformer Encoder
        transformer_output = self.transformer_encoder(embeddings)  # (batch, num_lanes+1, d_model)
        
        # 5. 提取 CLS token 的输出（全局特征）
        cls_output = transformer_output[:, 0, :]  # (batch, d_model)
        
        # 6. 通过输出 MLP 得到最终特征
        features = self.output_mlp(cls_output)  # (batch, features_dim)
        
        return features


class MLPJunctionModel(BaseFeaturesExtractor):
    """简单的 MLP 基线模型（用于对比）
    
    将所有 lane 特征展平后通过 MLP 处理
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        hidden_dims: list = [256, 256],
    ):
        """初始化 MLP 模型
        
        Args:
            observation_space: 观测空间 (num_lanes, lane_feature_dim)
            features_dim: 输出特征维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__(observation_space, features_dim)
        
        # 计算展平后的输入维度
        input_dim = observation_space.shape[0] * observation_space.shape[1]
        
        # 构建 MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, features_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            observations: shape (batch_size, num_lanes, lane_feature_dim)
            
        Returns:
            features: shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        # 展平 lane 特征
        flattened = observations.reshape(batch_size, -1)
        # 通过 MLP
        features = self.mlp(flattened)
        return features

