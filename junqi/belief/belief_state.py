"""
信念状态管理。
为对方每个存活棋子维护一个类型概率分布，
并根据游戏中的观察事件进行贝叶斯更新。
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ..env.pieces import (
    PieceType, PIECE_COUNTS, NUM_PIECE_TYPES, BattleResult,
    PIECE_NAMES, MOVABLE_TYPES,
)
from ..env.board import (
    Board, NUM_POSITIONS, HEADQUARTERS, ALL_CAMPS,
    pos, row_col, ROWS, COLS, RAILWAY_POSITIONS,
)


# 所有棋子类型列表（按 enum 值排序，不含 NONE）
ALL_TYPES = [pt for pt in PieceType if pt != PieceType.NONE]
TYPE_TO_IDX = {pt: i for i, pt in enumerate(ALL_TYPES)}
IDX_TO_TYPE = {i: pt for pt, i in TYPE_TO_IDX.items()}


class BeliefState:
    """
    信念状态：追踪对方每个棋子的类型概率分布。
    
    对方有25个棋子，每个棋子有12种可能的类型。
    信念矩阵: (25, 12) — beliefs[piece_id, type_idx] = P(type | observations)
    """

    def __init__(self, observer: int = 0):
        """
        Args:
            observer: 观察者玩家编号 (信念是关于对方(1-observer)的棋子)
        """
        self.observer = observer
        self.opponent = 1 - observer

        # 信念矩阵: (25, 12), 初始为均匀先验
        # beliefs[i, j] = P(piece i is type ALL_TYPES[j])
        self.beliefs = np.zeros((25, NUM_PIECE_TYPES), dtype=np.float64)

        # 初始化均匀先验（基于每种类型的数量）
        total = sum(PIECE_COUNTS.values())  # 25
        for j, pt in enumerate(ALL_TYPES):
            self.beliefs[:, j] = PIECE_COUNTS[pt] / total

        # 已确认类型的棋子
        self.confirmed: Dict[int, PieceType] = {}  # piece_id → confirmed type

        # 对方每种类型的剩余数量上限
        self.remaining_counts: Dict[PieceType, int] = dict(PIECE_COUNTS)

        # 已阵亡棋子 ID
        self.dead_pieces: Set[int] = set()

        # 追踪每个棋子的移动历史长度
        self.move_counts: Dict[int, int] = defaultdict(int)

    def update_on_move(self, piece_id: int, from_pos: int, to_pos: int,
                       is_rail_turn: bool = False):
        """
        根据移动事件更新信念。
        
        推断规则:
        - 移动了 → 不是地雷
        - 从大本营移动 → 不是军旗
        - 铁路上拐弯移动 → 一定是工兵
        """
        if piece_id in self.confirmed:
            return

        idx = piece_id
        
        # 移动了 → 排除不可移动类型
        self.beliefs[idx, TYPE_TO_IDX[PieceType.LANDMINE]] = 0

        # 从大本营位置移动 → 不是军旗
        if from_pos in HEADQUARTERS[self.opponent]:
            self.beliefs[idx, TYPE_TO_IDX[PieceType.FLAG]] = 0

        # 铁路拐弯 → 一定是工兵
        if is_rail_turn:
            for j in range(NUM_PIECE_TYPES):
                if ALL_TYPES[j] != PieceType.ENGINEER:
                    self.beliefs[idx, j] = 0
            self._confirm_piece(piece_id, PieceType.ENGINEER)

        self.move_counts[piece_id] += 1
        self._normalize(idx)
        self._apply_global_constraints()

    def update_on_battle(self, opponent_piece_id: int,
                         my_piece_type: PieceType,
                         result: BattleResult):
        """
        根据交战结果更新信念。
        
        Args:
            opponent_piece_id: 对方参战棋子的 ID
            my_piece_type: 己方参战棋子的类型（已知）
            result: 交战结果
        """
        if opponent_piece_id in self.confirmed:
            return

        idx = opponent_piece_id

        if result == BattleResult.ATTACKER_WIN:
            # 己方进攻赢了 → 对方棋子等级 < 己方（或对方是军旗）
            # 或者: 对方防守输了 → 对方等级 < 进攻方
            # 需要区分谁是进攻方
            # 这里统一处理: 己方赢了说明对方更弱
            my_rank = my_piece_type.value
            for j, pt in enumerate(ALL_TYPES):
                if pt == PieceType.BOMB:
                    # 炸弹会同归于尽，不会让对方赢
                    self.beliefs[idx, j] = 0
                elif pt == PieceType.FLAG:
                    pass  # 军旗可以被吃
                elif pt == PieceType.LANDMINE:
                    # 碰地雷只有工兵能赢
                    if my_piece_type != PieceType.ENGINEER:
                        self.beliefs[idx, j] = 0
                elif pt.value >= my_rank:
                    # 等级 >= 己方的不可能输
                    self.beliefs[idx, j] = 0

        elif result == BattleResult.DEFENDER_WIN:
            # 己方输了 → 对方等级 > 己方（或对方是地雷且己方非工兵）
            my_rank = my_piece_type.value
            for j, pt in enumerate(ALL_TYPES):
                if pt == PieceType.BOMB:
                    self.beliefs[idx, j] = 0  # 炸弹同归于尽
                elif pt == PieceType.LANDMINE:
                    if my_piece_type == PieceType.ENGINEER:
                        self.beliefs[idx, j] = 0  # 工兵排雷应该赢
                elif pt == PieceType.FLAG:
                    self.beliefs[idx, j] = 0  # 军旗不能赢
                elif pt.value <= my_rank:
                    self.beliefs[idx, j] = 0  # 更小的不可能赢

        elif result == BattleResult.BOTH_DIE:
            # 同归于尽 → 对方是炸弹，或等级相同，或碰地雷（非工兵）
            my_rank = my_piece_type.value
            for j, pt in enumerate(ALL_TYPES):
                if pt == PieceType.BOMB:
                    pass  # 炸弹可以同归于尽
                elif pt == PieceType.LANDMINE:
                    if my_piece_type == PieceType.ENGINEER:
                        self.beliefs[idx, j] = 0  # 工兵排雷是赢不是同归于尽
                    # 非工兵碰地雷确实同归于尽
                elif pt == PieceType.FLAG:
                    self.beliefs[idx, j] = 0
                elif pt.value != my_rank:
                    self.beliefs[idx, j] = 0

        self._normalize(idx)
        self._apply_global_constraints()

    def update_on_death(self, opponent_piece_id: int,
                        confirmed_type: Optional[PieceType] = None):
        """
        对方棋子阵亡。
        
        Args:
            opponent_piece_id: 阵亡的对方棋子 ID
            confirmed_type: 如果能确定类型（如交战后可确定）
        """
        self.dead_pieces.add(opponent_piece_id)

        if confirmed_type is not None:
            self._confirm_piece(opponent_piece_id, confirmed_type)
            self.remaining_counts[confirmed_type] = max(
                0, self.remaining_counts[confirmed_type] - 1)

        # 将该棋子的信念清零
        self.beliefs[opponent_piece_id, :] = 0

        self._apply_global_constraints()

    def update_on_piece_revealed(self, piece_id: int, piece_type: PieceType):
        """直接确认某个棋子的类型"""
        self._confirm_piece(piece_id, piece_type)
        self._apply_global_constraints()

    def get_belief(self, piece_id: int) -> np.ndarray:
        """获取指定棋子的类型概率分布 shape=(12,)"""
        return self.beliefs[piece_id].copy()

    def get_expected_value(self, piece_id: int,
                           value_table: Dict[PieceType, float]) -> float:
        """
        基于信念分布计算对方某棋子的期望价值。
        V(piece) = Σ P(type) × Value(type)
        """
        value = 0.0
        for j, pt in enumerate(ALL_TYPES):
            value += self.beliefs[piece_id, j] * value_table.get(pt, 0.0)
        return value

    def get_entropy(self, piece_id: int) -> float:
        """计算指定棋子信念分布的信息熵"""
        probs = self.beliefs[piece_id]
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def get_total_entropy(self) -> float:
        """所有存活未确认棋子的总信息熵"""
        total = 0.0
        for pid in range(25):
            if pid not in self.dead_pieces and pid not in self.confirmed:
                total += self.get_entropy(pid)
        return total

    def get_belief_matrix(self) -> np.ndarray:
        """返回完整信念矩阵 (25, 12)"""
        return self.beliefs.copy()

    # ──────────────────────────────────────
    #  内部方法
    # ──────────────────────────────────────

    def _confirm_piece(self, piece_id: int, piece_type: PieceType):
        """确认棋子类型"""
        self.confirmed[piece_id] = piece_type
        idx = piece_id
        self.beliefs[idx, :] = 0
        self.beliefs[idx, TYPE_TO_IDX[piece_type]] = 1.0

    def _normalize(self, piece_idx: int):
        """归一化指定棋子的信念分布"""
        total = self.beliefs[piece_idx].sum()
        if total > 0:
            self.beliefs[piece_idx] /= total
        else:
            # 所有概率为0，这不应该发生（回退到均匀分布）
            self.beliefs[piece_idx] = 1.0 / NUM_PIECE_TYPES

    def _apply_global_constraints(self):
        """
        应用全局约束：
        - 如果某种类型已全部确认/阵亡，其他棋子该类型概率归零
        - 如果某种类型只剩一个名额且只有一个棋子有该类型可能，则确认
        """
        # 计算每种类型的已确认/已消耗数量
        confirmed_counts: Dict[PieceType, int] = defaultdict(int)
        for pid, pt in self.confirmed.items():
            confirmed_counts[pt] += 1

        dead_confirmed_counts: Dict[PieceType, int] = defaultdict(int)
        for pid in self.dead_pieces:
            if pid in self.confirmed:
                dead_confirmed_counts[self.confirmed[pid]] += 1

        for j, pt in enumerate(ALL_TYPES):
            total_count = PIECE_COUNTS[pt]
            used_count = confirmed_counts.get(pt, 0)
            remaining = total_count - used_count

            if remaining <= 0:
                # 该类型已全部确认，其他未确认棋子不可能是该类型
                for pid in range(25):
                    if pid not in self.confirmed and pid not in self.dead_pieces:
                        self.beliefs[pid, j] = 0

        # 重新归一化所有未确认棋子
        for pid in range(25):
            if pid not in self.confirmed and pid not in self.dead_pieces:
                total = self.beliefs[pid].sum()
                if total > 0:
                    self.beliefs[pid] /= total

        # 检查是否有棋子可以唯一确定
        changed = True
        while changed:
            changed = False
            for pid in range(25):
                if pid in self.confirmed or pid in self.dead_pieces:
                    continue
                nonzero = np.nonzero(self.beliefs[pid] > 1e-10)[0]
                if len(nonzero) == 1:
                    pt = ALL_TYPES[nonzero[0]]
                    self._confirm_piece(pid, pt)
                    changed = True

    def clone(self) -> 'BeliefState':
        """深拷贝"""
        import copy
        return copy.deepcopy(self)

    def __repr__(self):
        confirmed = len(self.confirmed)
        dead = len(self.dead_pieces)
        entropy = self.get_total_entropy()
        return (f"BeliefState(observer=P{self.observer}, "
                f"confirmed={confirmed}, dead={dead}, "
                f"total_entropy={entropy:.2f})")
