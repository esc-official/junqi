"""
神经网络架构。
ResNet backbone + Policy/Value/Belief Heads。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..utils.encoding import NUM_CHANNELS, ROWS, COLS
from ..env.board import NUM_POSITIONS
from ..env.pieces import NUM_PIECE_TYPES


# ──────────────────────────────────────
#  基础模块
# ──────────────────────────────────────

class ResidualBlock(nn.Module):
    """残差块: Conv → BN → ReLU → Conv → BN → skip"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


# ──────────────────────────────────────
#  主干网络
# ──────────────────────────────────────

class SharedBackbone(nn.Module):
    """
    共享特征提取器。
    输入: (batch, 62, 12, 5)
    输出: (batch, hidden_dim, 12, 5)
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, hidden_dim: int = 128,
                 num_blocks: int = 12):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3,
                                  padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_dim)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        return x


# ──────────────────────────────────────
#  策略头
# ──────────────────────────────────────

class PolicyHead(nn.Module):
    """
    策略头：输出动作 logits。
    输入: (batch, hidden_dim, 12, 5)
    输出: (batch, 3600) — 60×60 动作空间的 logits
    """

    def __init__(self, hidden_dim: int = 128, action_size: int = NUM_POSITIONS * NUM_POSITIONS):
        super().__init__()
        self.action_size = action_size
        self.conv = nn.Conv2d(hidden_dim, 32, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * ROWS * COLS, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(features)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


# ──────────────────────────────────────
#  价值头
# ──────────────────────────────────────

class ValueHead(nn.Module):
    """
    价值头：评估局面 V(s) ∈ [-1, 1]。
    输入: (batch, hidden_dim, 12, 5)
    输出: (batch, 1)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, 4, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(4 * ROWS * COLS, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(features)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


# ──────────────────────────────────────
#  信念预测头（辅助任务）
# ──────────────────────────────────────

class BeliefHead(nn.Module):
    """
    信念预测辅助头：预测对方每个位置棋子的类型。
    输入: (batch, hidden_dim, 12, 5)
    输出: (batch, 12, 12, 5) — 每个位置 12 种类型的 logits
    
    训练时使用完整状态作为标签（利用训练时的上帝视角信息）。
    推理时不直接使用此头的输出，但它帮助 backbone 学习信息推理能力。
    """

    def __init__(self, hidden_dim: int = 128, num_types: int = NUM_PIECE_TYPES):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, num_types, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(features)))
        logits = self.conv2(x)  # (batch, 12, 12, 5)
        return logits


# ──────────────────────────────────────
#  完整网络
# ──────────────────────────────────────

class JunqiNetwork(nn.Module):
    """
    军棋 AI 神经网络。
    
    架构:
        Input (62, 12, 5) → SharedBackbone → {PolicyHead, ValueHead, BeliefHead}
    
    输出:
        - policy_logits: (batch, 3600) 动作概率分布的 logits
        - value: (batch, 1) 局面价值估计
        - belief_logits: (batch, 12, 12, 5) 对方棋子类型预测（辅助）
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden_dim: int = 128,
        num_res_blocks: int = 12,
    ):
        super().__init__()
        self.backbone = SharedBackbone(in_channels, hidden_dim, num_res_blocks)
        self.policy_head = PolicyHead(hidden_dim)
        self.value_head = ValueHead(hidden_dim)
        self.belief_head = BeliefHead(hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            state: (batch, 62, 12, 5) 状态张量
            action_mask: (batch, 3600) 合法动作掩码 (1=合法, 0=非法)
        
        Returns:
            policy_logits: (batch, 3600)  — 应用掩码后的 logits
            value: (batch, 1)
            belief_logits: (batch, 12, 12, 5)
        """
        features = self.backbone(state)

        # 策略
        policy_logits = self.policy_head(features)
        if action_mask is not None:
            # 将非法动作的 logits 设为极小值
            policy_logits = policy_logits + (1 - action_mask) * (-1e8)

        # 价值
        value = self.value_head(features)

        # 信念
        belief_logits = self.belief_head(features)

        return policy_logits, value, belief_logits

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作及其对应的 log_prob 和 value（PPO 训练用）。
        
        Args:
            state: (batch, C, H, W)
            action_mask: (batch, 3600)
            action: (batch,) 如果提供则计算给定动作的概率，否则采样新动作
            deterministic: 是否贪心选择
        
        Returns:
            action: (batch,)
            log_prob: (batch,)
            entropy: (batch,)
            value: (batch,)
        """
        policy_logits, value, _ = self.forward(state, action_mask)
        value = value.squeeze(-1)

        # 构建分类分布
        dist = torch.distributions.Categorical(logits=policy_logits)

        if action is None:
            if deterministic:
                action = policy_logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """仅获取价值估计"""
        features = self.backbone(state)
        return self.value_head(features).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
