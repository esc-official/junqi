"""
Gymnasium 风格的军棋环境。
包装 GameState + BeliefState，提供标准的 RL 环境接口。
"""
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .game import GameState, GameResult
from .pieces import PieceType, BattleResult, PIECE_BASE_VALUE
from .board import NUM_POSITIONS
from ..belief.belief_state import BeliefState
from ..utils.encoding import StateEncoder, STATE_SHAPE
from ..training.reward import RewardCalculator


class JunqiEnv:
    """
    军棋 RL 环境（两人对战）。
    
    观察空间: (62, 12, 5) float32 张量
    动作空间: 3600 (= 60 × 60) 离散动作
    
    每个 step 从当前玩家的视角返回 (obs, reward, done, truncated, info)。
    """

    # 类级别共享编码器
    _encoder = StateEncoder()

    def __init__(
        self,
        max_turns: int = 500,
        reward_config: Optional[dict] = None,
    ):
        self.max_turns = max_turns
        self.reward_calc = RewardCalculator(**(reward_config or {}))

        # 状态空间和动作空间大小
        self.observation_shape = STATE_SHAPE
        self.action_space_n = NUM_POSITIONS * NUM_POSITIONS  # 3600

        # 内部状态
        self.game: Optional[GameState] = None
        self.beliefs: Dict[int, BeliefState] = {}
        self._prev_potential: Dict[int, float] = {}

    def reset(
        self,
        template_idx_p0: Optional[int] = None,
        template_idx_p1: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        重置环境。
        
        Returns:
            (observation, info) for current player
        """
        self.game = GameState(
            template_idx_p0=template_idx_p0,
            template_idx_p1=template_idx_p1,
            seed=seed,
        )
        self.game.max_turns = self.max_turns

        # 初始化双方信念
        self.beliefs = {
            0: BeliefState(observer=0),
            1: BeliefState(observer=1),
        }

        # 初始化势函数
        for player in [0, 1]:
            self._prev_potential[player] = self.reward_calc.potential(
                self.game, self.beliefs[player], player
            )

        player = self.game.current_player
        obs = self._get_obs(player)
        info = self._get_info(player)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行动作。
        
        Args:
            action: 动作索引 [0, 3600)
        
        Returns:
            (obs, reward, terminated, truncated, info)
            obs 是下一个玩家的观察
        """
        player = self.game.current_player
        from_pos, to_pos = self.game.index_to_action(action)

        # 计算更新前的势函数
        old_potential = self._prev_potential[player]

        # 保存旧信念（用于信息增益计算）
        old_belief = self.beliefs[player].clone()

        # 执行动作
        event = self.game.step(from_pos, to_pos)

        # 更新双方信念
        self._update_beliefs(event)

        # 计算奖励
        reward = self.reward_calc.compute_reward(
            event=event,
            game=self.game,
            belief=self.beliefs[player],
            old_belief=old_belief,
            player=player,
            old_potential=old_potential,
        )

        # 更新势函数
        new_potential = self.reward_calc.potential(
            self.game, self.beliefs[player], player
        )
        self._prev_potential[player] = new_potential

        # 检查游戏是否结束
        terminated = self.game.result != GameResult.ONGOING
        truncated = (self.game.result == GameResult.DRAW and
                     self.game.turn_count >= self.max_turns)

        # 下一个玩家的观察
        next_player = self.game.current_player
        obs = self._get_obs(next_player) if not terminated else self._get_obs(player)
        info = self._get_info(player)
        info['event'] = event
        info['current_player'] = player
        info['next_player'] = next_player

        return obs, reward, terminated, truncated, info

    def get_action_mask(self, player: Optional[int] = None) -> np.ndarray:
        """获取当前玩家的合法动作掩码"""
        if player is None:
            player = self.game.current_player
        return self.game.get_action_mask(player)

    def get_current_player(self) -> int:
        return self.game.current_player

    def render(self, perspective: Optional[int] = None):
        """渲染棋盘"""
        self.game.render(perspective=perspective)

    # ──────────────────────────────────────
    #  内部方法
    # ──────────────────────────────────────

    def _get_obs(self, player: int) -> np.ndarray:
        """获取指定玩家的观察张量"""
        return self._encoder.encode(self.game, self.beliefs[player], player)

    def _get_info(self, player: int) -> dict:
        """获取附加信息"""
        return {
            'player': player,
            'turn': self.game.turn_count,
            'result': self.game.result,
            'action_mask': self.get_action_mask(player),
            'belief_entropy': self.beliefs[player].get_total_entropy(),
        }

    def _update_beliefs(self, event):
        """根据移动事件更新双方信念"""
        acting_player = event.player
        opponent = 1 - acting_player

        # 对方视角：观察到 acting_player 的棋子移动了
        # → 更新对方对 acting_player 棋子的信念
        # 注意：这里 beliefs[opponent] 追踪的是 acting_player 的棋子
        opp_belief = self.beliefs[opponent]

        # acting_player 的棋子从 from_pos 移动到 to_pos
        # 需要找到对应的 piece_id
        attacker = event.attacker
        if attacker is not None:
            pid = attacker.piece_id
            opp_belief.update_on_move(
                pid, event.from_pos, event.to_pos,
                is_rail_turn=event.is_rail_turn
            )

        # 如果发生交战
        if event.battle_result is not None and event.defender is not None:
            defender = event.defender
            attacker = event.attacker

            # acting_player 的信念中：更新对方棋子的信念
            my_belief = self.beliefs[acting_player]
            my_belief.update_on_battle(
                opponent_piece_id=defender.piece_id,
                my_piece_type=attacker.piece_type,
                result=event.battle_result,
            )

            # opponent 的信念中：更新 acting_player 棋子的信念
            # 需要翻转 battle result
            flipped_result = self._flip_battle_result(event.battle_result)
            opp_belief.update_on_battle(
                opponent_piece_id=attacker.piece_id,
                my_piece_type=defender.piece_type,
                result=flipped_result,
            )

            # 处理阵亡
            if event.battle_result == BattleResult.ATTACKER_WIN:
                my_belief.update_on_death(
                    defender.piece_id,
                    confirmed_type=defender.piece_type
                )
                opp_belief.update_on_piece_revealed(
                    attacker.piece_id, attacker.piece_type
                )
            elif event.battle_result == BattleResult.DEFENDER_WIN:
                opp_belief.update_on_death(
                    attacker.piece_id,
                    confirmed_type=attacker.piece_type
                )
                my_belief.update_on_piece_revealed(
                    defender.piece_id, defender.piece_type
                )
            elif event.battle_result == BattleResult.BOTH_DIE:
                my_belief.update_on_death(
                    defender.piece_id,
                    confirmed_type=defender.piece_type
                )
                opp_belief.update_on_death(
                    attacker.piece_id,
                    confirmed_type=attacker.piece_type
                )

    @staticmethod
    def _flip_battle_result(result: BattleResult) -> BattleResult:
        """翻转交战结果（从对方视角）"""
        if result == BattleResult.ATTACKER_WIN:
            return BattleResult.DEFENDER_WIN
        elif result == BattleResult.DEFENDER_WIN:
            return BattleResult.ATTACKER_WIN
        return result  # BOTH_DIE 不变
