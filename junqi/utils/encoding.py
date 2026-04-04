"""
状态编码器。
将 GameState + BeliefState 编码为神经网络输入张量。
"""
import numpy as np
from typing import Dict, Optional

from ..env.pieces import PieceType, NUM_PIECE_TYPES, PIECE_COUNTS
from ..env.board import (
    NUM_POSITIONS, ROWS, COLS, ALL_CAMPS, HEADQUARTERS,
    RAILWAY_POSITIONS, pos, row_col,
)
from ..env.game import GameState, MoveEvent
from ..belief.belief_state import BeliefState, ALL_TYPES, TYPE_TO_IDX


# ──────────────────────────────────────
#  通道定义
# ──────────────────────────────────────
# 己方棋子类型:     12 通道  (one-hot per type)
# 己方可移动性:      1 通道
# 对方存在:          1 通道  (位置已知，类型未知)
# 对方信念分布:     12 通道  (每种类型的概率)
# 对方已暴露:       12 通道  (one-hot per revealed type)
# 棋盘拓扑:          3 通道  (行营, 大本营, 铁路)
# 历史移动:          8 通道  (最近4步的from/to)
# 对方剩余计数:     12 通道  (广播)
# 回合数:            1 通道  (广播)
# ──────────────────────────────────────
# 总通道数:         62

NUM_CHANNELS = 62
BOARD_SHAPE = (ROWS, COLS)  # (12, 5)
STATE_SHAPE = (NUM_CHANNELS, ROWS, COLS)


class StateEncoder:
    """将游戏状态编码为神经网络输入张量"""

    def __init__(self):
        # 预计算棋盘拓扑通道（固定不变）
        self._topo_camp = np.zeros(BOARD_SHAPE, dtype=np.float32)
        self._topo_hq = np.zeros(BOARD_SHAPE, dtype=np.float32)
        self._topo_rail = np.zeros(BOARD_SHAPE, dtype=np.float32)

        for p in ALL_CAMPS:
            r, c = row_col(p)
            self._topo_camp[r, c] = 1.0

        for player in [0, 1]:
            for hq in HEADQUARTERS[player]:
                r, c = row_col(hq)
                self._topo_hq[r, c] = 1.0

        for p in RAILWAY_POSITIONS:
            r, c = row_col(p)
            self._topo_rail[r, c] = 1.0

    def encode(
        self,
        game: GameState,
        belief: BeliefState,
        player: int,
    ) -> np.ndarray:
        """
        编码当前状态为张量。
        
        Args:
            game: 游戏状态
            belief: 信念状态（关于对方棋子）
            player: 当前观察者玩家
        
        Returns:
            np.ndarray of shape (62, 12, 5)
        """
        state = np.zeros(STATE_SHAPE, dtype=np.float32)
        ch = 0  # 当前通道偏移

        opponent = 1 - player
        obs = game.get_observation(player)

        # ── 己方棋子类型 (12通道) ──
        for position, piece_type in obs['my_pieces'].items():
            r, c = row_col(position)
            type_idx = TYPE_TO_IDX[piece_type]
            state[ch + type_idx, r, c] = 1.0
        ch += NUM_PIECE_TYPES  # +12 = 12

        # ── 己方可移动性 (1通道) ──
        for position, piece_type in obs['my_pieces'].items():
            if piece_type in (PieceType.FLAG, PieceType.LANDMINE):
                continue
            r, c = row_col(position)
            state[ch, r, c] = 1.0
        ch += 1  # +1 = 13

        # ── 对方棋子存在 (1通道) ──
        for position in obs['opponent_positions']:
            r, c = row_col(position)
            state[ch, r, c] = 1.0
        ch += 1  # +1 = 14

        # ── 对方信念分布 (12通道) ──
        # 需要将 belief 矩阵 (25 pieces × 12 types) 映射到棋盘位置
        opp_piece_positions = {}  # piece_id → position
        for pid, piece in game.pieces[opponent].items():
            if piece.alive and pid in game.position_of[opponent]:
                opp_piece_positions[pid] = game.position_of[opponent][pid]

        for pid, position in opp_piece_positions.items():
            r, c = row_col(position)
            belief_vec = belief.get_belief(pid)
            for j in range(NUM_PIECE_TYPES):
                state[ch + j, r, c] = belief_vec[j]
        ch += NUM_PIECE_TYPES  # +12 = 26

        # ── 对方已暴露棋子 (12通道) ──
        for position, piece_type in obs['opponent_revealed'].items():
            r, c = row_col(position)
            type_idx = TYPE_TO_IDX[piece_type]
            state[ch + type_idx, r, c] = 1.0
        ch += NUM_PIECE_TYPES  # +12 = 38

        # ── 棋盘拓扑 (3通道) ──
        state[ch] = self._topo_camp
        state[ch + 1] = self._topo_hq
        state[ch + 2] = self._topo_rail
        ch += 3  # +3 = 41

        # ── 历史移动 (8通道: 最近4步的from和to) ──
        recent_moves = obs['move_history'][-4:]
        for i, event in enumerate(recent_moves):
            # from 通道
            r_from, c_from = row_col(event.from_pos)
            state[ch + i * 2, r_from, c_from] = 1.0
            # to 通道
            r_to, c_to = row_col(event.to_pos)
            state[ch + i * 2 + 1, r_to, c_to] = 1.0
        ch += 8  # +8 = 49

        # ── 对方剩余棋子计数 (12通道, 广播到全图) ──
        opp_counts = obs['alive_counts'].get(opponent, {})
        for j, pt in enumerate(ALL_TYPES):
            count = opp_counts.get(pt, 0)
            max_count = PIECE_COUNTS[pt]
            # 归一化到 [0, 1]
            state[ch + j, :, :] = count / max(max_count, 1)
        ch += NUM_PIECE_TYPES  # +12 = 61

        # ── 回合数 (1通道, 广播) ──
        normalized_turn = min(obs['turn_count'] / 500.0, 1.0)
        state[ch, :, :] = normalized_turn
        ch += 1  # +1 = 62

        assert ch == NUM_CHANNELS, f"Channel mismatch: {ch} != {NUM_CHANNELS}"

        # 如果是 player 1 的视角，翻转棋盘使得己方始终在底部
        if player == 1:
            state = state[:, ::-1, :].copy()  # 上下翻转行

        return state

    def encode_batch(
        self,
        games: list,
        beliefs: list,
        player: int,
    ) -> np.ndarray:
        """批量编码"""
        batch = np.stack([
            self.encode(g, b, player)
            for g, b in zip(games, beliefs)
        ])
        return batch
