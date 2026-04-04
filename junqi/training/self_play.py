"""
自博弈管理器。
管理对手池、对手选择策略和 ELO 评分。
"""
import copy
import random
import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class OpponentSnapshot:
    """对手快照"""
    state_dict: dict
    elo: float = 1000.0
    generation: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played


class RuleBasedAI:
    """
    简单规则 AI（用于 Self-Play 的多样性对手）。
    
    策略：
    1. 如果能吃掉已暴露的高价值棋子 → 吃
    2. 否则推进前排棋子
    3. 随机合法动作作为兜底
    """

    def select_action(
        self,
        game_state,
        action_mask: 'numpy.ndarray',
    ) -> int:
        """选择动作"""
        import numpy as np
        legal_indices = np.where(action_mask > 0)[0]
        if len(legal_indices) == 0:
            return 0  # 不应该发生
        # 随机选择合法动作
        return int(np.random.choice(legal_indices))


class SelfPlayManager:
    """
    自博弈管理器。
    
    管理历史对手池，提供多样化的训练对手。
    """

    def __init__(
        self,
        network_class,
        network_kwargs: dict,
        pool_size: int = 20,
        device: str = 'cpu',
        save_dir: str = './checkpoints/opponents',
    ):
        self.network_class = network_class
        self.network_kwargs = network_kwargs
        self.pool_size = pool_size
        self.device = device
        self.save_dir = save_dir

        self.opponent_pool: List[OpponentSnapshot] = []
        self.current_generation = 0

        self.rule_ai = RuleBasedAI()

        os.makedirs(save_dir, exist_ok=True)

    def add_snapshot(self, network: nn.Module, elo: float = 1000.0):
        """将当前网络添加到对手池"""
        snapshot = OpponentSnapshot(
            state_dict=copy.deepcopy(network.state_dict()),
            elo=elo,
            generation=self.current_generation,
        )
        self.opponent_pool.append(snapshot)

        # 保持池大小
        if len(self.opponent_pool) > self.pool_size:
            # 保留第一个和最后几个，移除中间的
            self.opponent_pool = (
                [self.opponent_pool[0]]  # 最早的
                + self.opponent_pool[-(self.pool_size - 1):]  # 最新的
            )

        self.current_generation += 1

    def select_opponent(
        self,
        strategy: str = 'mixed',
    ) -> Tuple[str, object]:
        """
        选择一个对手。
        
        Args:
            strategy: 'latest' | 'random' | 'rule' | 'mixed'
        
        Returns:
            (opponent_type, opponent)
            opponent_type: 'network' | 'rule'
            opponent: 网络实例或 RuleBasedAI 实例
        """
        if strategy == 'rule' or not self.opponent_pool:
            return 'rule', self.rule_ai

        if strategy == 'latest':
            return 'network', self._load_opponent(self.opponent_pool[-1])

        if strategy == 'random':
            snapshot = random.choice(self.opponent_pool)
            return 'network', self._load_opponent(snapshot)

        # mixed 策略
        r = random.random()
        if r < 0.5:
            # 50%: 最新版本
            return 'network', self._load_opponent(self.opponent_pool[-1])
        elif r < 0.8 and len(self.opponent_pool) > 1:
            # 30%: 随机历史版本
            snapshot = random.choice(self.opponent_pool[:-1])
            return 'network', self._load_opponent(snapshot)
        else:
            # 20%: 规则 AI
            return 'rule', self.rule_ai

    def _load_opponent(self, snapshot: OpponentSnapshot) -> nn.Module:
        """从快照加载对手网络"""
        network = self.network_class(**self.network_kwargs)
        network.load_state_dict(snapshot.state_dict)
        network = network.to(self.device)
        network.eval()
        return network

    def update_elo(
        self,
        player_elo: float,
        opponent_idx: int,
        result: float,  # 1.0=win, 0.5=draw, 0.0=loss
        k: float = 32.0,
    ) -> Tuple[float, float]:
        """
        更新 ELO 评分。
        
        Returns:
            (new_player_elo, new_opponent_elo)
        """
        if opponent_idx >= len(self.opponent_pool):
            return player_elo, 0.0

        opponent = self.opponent_pool[opponent_idx]
        opp_elo = opponent.elo

        # 预期胜率
        expected_player = 1.0 / (1.0 + 10 ** ((opp_elo - player_elo) / 400))
        expected_opp = 1.0 - expected_player

        # 更新
        new_player_elo = player_elo + k * (result - expected_player)
        new_opp_elo = opp_elo + k * ((1 - result) - expected_opp)

        opponent.elo = new_opp_elo

        # 更新胜负记录
        if result > 0.5:
            opponent.losses += 1
        elif result < 0.5:
            opponent.wins += 1
        else:
            opponent.draws += 1

        return new_player_elo, new_opp_elo

    def save_pool(self, path: Optional[str] = None):
        """保存整个对手池"""
        if path is None:
            path = os.path.join(self.save_dir, 'opponent_pool.pt')
        torch.save({
            'pool': [(s.state_dict, s.elo, s.generation) for s in self.opponent_pool],
            'generation': self.current_generation,
        }, path)

    def load_pool(self, path: Optional[str] = None):
        """加载对手池"""
        if path is None:
            path = os.path.join(self.save_dir, 'opponent_pool.pt')
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location=self.device)
        self.opponent_pool = []
        for sd, elo, gen in data['pool']:
            self.opponent_pool.append(OpponentSnapshot(
                state_dict=sd, elo=elo, generation=gen
            ))
        self.current_generation = data['generation']
