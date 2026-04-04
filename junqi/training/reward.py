"""
奖励计算模块。
实现基于信念的动态奖励和 PBRS（基于势函数的奖励塑形）。
"""
import numpy as np
from typing import Dict, Optional

from ..env.pieces import (
    PieceType, BattleResult, PIECE_BASE_VALUE, PIECE_COUNTS,
    NUM_PIECE_TYPES,
)
from ..env.game import GameState, GameResult, MoveEvent
from ..belief.belief_state import BeliefState, ALL_TYPES


class RewardCalculator:
    """
    奖励计算器。
    
    R_total = R_terminal
              + α · R_piece_value      (基于信念的吃子奖励)
              + β · R_info_gain        (信息增益奖励)
              + PBRS(Φ)                (基于势函数的奖励塑形)
    """

    def __init__(
        self,
        alpha_piece: float = 0.3,       # 吃子奖励系数
        beta_info: float = 0.01,        # 信息增益系数
        gamma_discount: float = 0.99,   # 折扣因子
        pbrs_weight: float = 0.1,       # PBRS 权重
        terminal_reward: float = 1.0,   # 终局奖励
    ):
        self.alpha_piece = alpha_piece
        self.beta_info = beta_info
        self.gamma = gamma_discount
        self.pbrs_weight = pbrs_weight
        self.terminal_reward = terminal_reward

    def compute_reward(
        self,
        event: MoveEvent,
        game: GameState,
        belief: BeliefState,
        old_belief: BeliefState,
        player: int,
        old_potential: float,
    ) -> float:
        """
        计算单步奖励。
        
        Args:
            event: 本步的移动事件
            game: 执行后的游戏状态
            belief: 更新后的信念
            old_belief: 更新前的信念
            player: 行动玩家
            old_potential: 行动前的势函数值
        
        Returns:
            总奖励
        """
        reward = 0.0

        # 1. 终局奖励
        reward += self._terminal_reward(game, player)

        # 2. 基于信念的吃子奖励
        reward += self.alpha_piece * self._piece_value_reward(event, belief, player)

        # 3. 信息增益奖励
        reward += self.beta_info * self._info_gain_reward(old_belief, belief)

        # 4. PBRS
        new_potential = self.potential(game, belief, player)
        pbrs = self.gamma * new_potential - old_potential
        reward += self.pbrs_weight * pbrs

        return reward

    def _terminal_reward(self, game: GameState, player: int) -> float:
        """终局奖励"""
        if game.result == GameResult.ONGOING:
            return 0.0
        elif game.result == GameResult.DRAW:
            return 0.0
        elif (game.result == GameResult.PLAYER0_WIN and player == 0) or \
             (game.result == GameResult.PLAYER1_WIN and player == 1):
            return self.terminal_reward
        else:
            return -self.terminal_reward

    def _piece_value_reward(
        self,
        event: MoveEvent,
        belief: BeliefState,
        player: int,
    ) -> float:
        """
        基于信念的吃子奖励。
        吃掉对方棋子 → 正奖励 = 该棋子的期望价值
        己方被吃 → 负奖励 = 己方棋子的确定价值
        """
        if event.battle_result is None:
            return 0.0

        reward = 0.0

        if event.battle_result == BattleResult.ATTACKER_WIN:
            if event.player == player:
                # 己方进攻赢了 → 吃掉对方棋子
                # 被吃的对方棋子现在类型已确认
                defender_value = PIECE_BASE_VALUE.get(
                    event.defender.piece_type, 0.0
                )
                reward += defender_value
            else:
                # 对方进攻赢了 → 己方棋子被吃
                defender_value = PIECE_BASE_VALUE.get(
                    event.defender.piece_type, 0.0
                )
                reward -= defender_value

        elif event.battle_result == BattleResult.DEFENDER_WIN:
            if event.player == player:
                # 己方进攻输了 → 己方棋子阵亡
                attacker_value = PIECE_BASE_VALUE.get(
                    event.attacker.piece_type, 0.0
                )
                reward -= attacker_value
            else:
                # 对方进攻输了 → 对方棋子阵亡
                attacker_value = PIECE_BASE_VALUE.get(
                    event.attacker.piece_type, 0.0
                )
                reward += attacker_value

        elif event.battle_result == BattleResult.BOTH_DIE:
            if event.player == player:
                # 己方发起同归于尽
                my_val = PIECE_BASE_VALUE.get(event.attacker.piece_type, 0.0)
                opp_val = PIECE_BASE_VALUE.get(event.defender.piece_type, 0.0)
                reward += (opp_val - my_val)
            else:
                my_val = PIECE_BASE_VALUE.get(event.defender.piece_type, 0.0)
                opp_val = PIECE_BASE_VALUE.get(event.attacker.piece_type, 0.0)
                reward += (opp_val - my_val)

        # 归一化
        return reward / 100.0

    def _info_gain_reward(
        self,
        old_belief: BeliefState,
        new_belief: BeliefState,
    ) -> float:
        """
        信息增益奖励 = 总信息熵的减少量。
        鼓励 AI 主动试探来降低不确定性。
        """
        old_entropy = old_belief.get_total_entropy()
        new_entropy = new_belief.get_total_entropy()
        return max(0.0, old_entropy - new_entropy)

    def potential(
        self,
        game: GameState,
        belief: BeliefState,
        player: int,
    ) -> float:
        """
        势函数 Φ(s): 综合评估当前局面优势。
        
        Φ = w1 · (己方棋力 - 对方预估棋力)
            + w2 · 位置优势
            + w3 · 信息优势
        """
        opponent = 1 - player

        # 己方棋力
        my_power = 0.0
        for pt, count in game.alive_counts[player].items():
            my_power += PIECE_BASE_VALUE.get(pt, 0.0) * count

        # 对方预估棋力
        opp_power = 0.0
        for pt, count in game.alive_counts[opponent].items():
            opp_power += PIECE_BASE_VALUE.get(pt, 0.0) * count

        # 棋力差
        power_diff = (my_power - opp_power) / 200.0

        # 位置优势（简化：平均推进程度）
        pos_adv = self._position_advantage(game, player)

        # 信息优势（己方不确定性低于对方）
        my_entropy = belief.get_total_entropy()
        info_adv = -my_entropy / 50.0  # 归一化

        return 0.5 * power_diff + 0.3 * pos_adv + 0.2 * info_adv

    def _position_advantage(self, game: GameState, player: int) -> float:
        """计算位置优势（棋子推进程度）"""
        from ..env.board import row_col

        adv = 0.0
        count = 0
        for pid, piece in game.pieces[player].items():
            if not piece.is_movable():
                continue
            if pid not in game.position_of[player]:
                continue
            p = game.position_of[player][pid]
            r, _ = row_col(p)
            # player 0 从底部(row 11)向顶部(row 0)进攻
            # player 1 从顶部(row 0)向底部(row 11)进攻
            if player == 0:
                advancement = (11 - r) / 11.0
            else:
                advancement = r / 11.0
            adv += advancement
            count += 1

        return adv / max(count, 1)
