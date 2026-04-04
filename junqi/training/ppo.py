"""
PPO (Proximal Policy Optimization) 算法实现。
适配军棋环境的离散动作空间与动作掩码。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class RolloutBuffer:
    """
    经验回放缓冲区（单轨迹存储）。
    存储一局游戏中一个玩家的所有 (s, a, r, ...) 元组。
    """
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    # 信念标签（辅助任务）
    belief_labels: List[np.ndarray] = field(default_factory=list)

    def add(self, state, action, action_mask, log_prob, reward, value, done,
            belief_label=None):
        self.states.append(state)
        self.actions.append(action)
        self.action_masks.append(action_mask)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        if belief_label is not None:
            self.belief_labels.append(belief_label)

    def __len__(self):
        return len(self.states)

    def clear(self):
        for lst in [self.states, self.actions, self.action_masks,
                    self.log_probs, self.rewards, self.values, self.dones,
                    self.belief_labels]:
            lst.clear()


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 GAE (Generalized Advantage Estimation)。
    
    Returns:
        advantages: (T,)
        returns: (T,) = advantages + values
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_non_terminal = 1.0 - float(dones[t])
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


class PPOAgent:
    """
    PPO 训练器。
    
    管理模型优化、经验收集和策略更新。
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        belief_coeff: float = 0.1,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        mini_batch_size: int = 64,
        device: str = 'cpu',
    ):
        self.network = network.to(device)
        self.device = device

        self.optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)

        # PPO 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.belief_coeff = belief_coeff
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        选择动作。
        
        Args:
            state: (C, H, W) 状态张量
            action_mask: (3600,) 合法动作掩码
            deterministic: 是否贪心
        
        Returns:
            (action, log_prob, value)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        action, log_prob, _, value = self.network.get_action_and_value(
            state_t, mask_t, deterministic=deterministic
        )

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    def update(self, buffers: List[RolloutBuffer]) -> Dict[str, float]:
        """
        使用收集的经验更新网络。
        
        Args:
            buffers: 多局游戏的经验缓冲区列表
        
        Returns:
            训练统计信息
        """
        # 合并所有经验
        all_states = []
        all_actions = []
        all_masks = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        all_belief_labels = []

        for buf in buffers:
            if len(buf) == 0:
                continue

            # 计算 GAE
            # 对于已完成的游戏，last_value = 0
            last_value = 0.0
            advantages, returns = compute_gae(
                buf.rewards, buf.values, buf.dones,
                last_value, self.gamma, self.gae_lambda
            )

            all_states.extend(buf.states)
            all_actions.extend(buf.actions)
            all_masks.extend(buf.action_masks)
            all_old_log_probs.extend(buf.log_probs)
            all_advantages.append(advantages)
            all_returns.append(returns)
            if buf.belief_labels:
                all_belief_labels.extend(buf.belief_labels)

        if not all_states:
            return {}

        # 转为张量
        states = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)
        masks = torch.FloatTensor(np.array(all_masks)).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        advantages = torch.FloatTensor(np.concatenate(all_advantages)).to(self.device)
        returns = torch.FloatTensor(np.concatenate(all_returns)).to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 信念标签
        has_belief = len(all_belief_labels) == len(all_states)
        if has_belief:
            belief_labels = torch.LongTensor(np.array(all_belief_labels)).to(self.device)

        # PPO 多轮更新
        N = len(states)
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'belief_loss': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
        }
        num_updates = 0

        for epoch in range(self.num_epochs):
            indices = torch.randperm(N)

            for start in range(0, N, self.mini_batch_size):
                end = min(start + self.mini_batch_size, N)
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_masks = masks[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # 前向传播
                _, new_log_probs, entropy, new_values = \
                    self.network.get_action_and_value(
                        b_states, b_masks, action=b_actions
                    )

                # PPO Clip Loss
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(new_values, b_returns)

                # Entropy Bonus
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = (policy_loss
                        + self.value_coeff * value_loss
                        + self.entropy_coeff * entropy_loss)

                # Belief Loss (辅助任务)
                if has_belief:
                    b_belief_labels = belief_labels[batch_idx]
                    _, _, belief_logits = self.network(b_states, b_masks)
                    # belief_logits: (batch, 12, 12, 5)
                    # belief_labels: (batch, 12, 5) — 每个位置的真实类型 index
                    # 只在有对方棋子的位置计算 loss
                    belief_loss = F.cross_entropy(
                        belief_logits.reshape(-1, belief_logits.size(1)),
                        b_belief_labels.reshape(-1),
                        ignore_index=-1,
                    )
                    loss += self.belief_coeff * belief_loss
                    stats['belief_loss'] += belief_loss.item()

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # 统计
                with torch.no_grad():
                    approx_kl = (b_old_log_probs - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.mean().item()
                stats['total_loss'] += loss.item()
                stats['approx_kl'] += approx_kl
                stats['clip_fraction'] += clip_frac
                num_updates += 1

        # 平均化统计
        if num_updates > 0:
            for key in stats:
                stats[key] /= num_updates

        return stats

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
