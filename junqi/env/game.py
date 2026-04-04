"""
游戏状态管理。
管理棋盘上所有棋子的位置、执行移动、交战判定、胜负判定。
"""
import copy
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from .pieces import (
    PieceType, Piece, BattleResult, battle,
    PIECE_COUNTS, MOVABLE_TYPES, NUM_PIECE_TYPES, PIECE_NAMES,
)
from .board import (
    Board, NUM_POSITIONS, HEADQUARTERS, CAMPS, ALL_CAMPS,
    pos, row_col, ROWS, COLS,
)
from .templates import get_random_template, get_template


class GameResult(IntEnum):
    ONGOING = 0
    PLAYER0_WIN = 1
    PLAYER1_WIN = 2
    DRAW = 3


@dataclass
class MoveEvent:
    """一次移动/交战的完整信息"""
    player: int
    from_pos: int
    to_pos: int
    attacker: Optional[Piece] = None
    defender: Optional[Piece] = None
    battle_result: Optional[BattleResult] = None
    flag_captured: bool = False
    is_rail_move: bool = False
    is_rail_turn: bool = False  # 铁路拐弯移动（仅工兵）


class GameState:
    """
    军棋游戏状态。
    
    管理:
    - 双方棋子的位置
    - 合法动作生成
    - 移动执行与交战
    - 胜负判定
    """

    def __init__(
        self,
        template_idx_p0: Optional[int] = None,
        template_idx_p1: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        初始化游戏。
        
        Args:
            template_idx_p0: player 0 的布阵模板索引 (0-49), None 则随机
            template_idx_p1: player 1 的布阵模板索引 (0-49), None 则随机
            seed: 随机种子
        """
        import random
        rng = random.Random(seed)

        # 获取布阵
        if template_idx_p0 is not None:
            deploy_0 = get_template(template_idx_p0, player=0)
        else:
            deploy_0 = get_random_template(player=0, rng=rng)

        if template_idx_p1 is not None:
            deploy_1 = get_template(template_idx_p1, player=1)
        else:
            deploy_1 = get_random_template(player=1, rng=rng)

        # 创建棋子
        self.pieces: Dict[int, Dict[int, Piece]] = {0: {}, 1: {}}  # player → {piece_id: Piece}
        self.board: Dict[int, Optional[Piece]] = {i: None for i in range(NUM_POSITIONS)}  # position → Piece
        self.position_of: Dict[int, Dict[int, int]] = {0: {}, 1: {}}  # player → {piece_id: position}

        pid = 0
        for position, piece_type in deploy_0.items():
            piece = Piece(piece_type, player=0, piece_id=pid)
            self.pieces[0][pid] = piece
            self.board[position] = piece
            self.position_of[0][pid] = position
            pid += 1

        pid = 0
        for position, piece_type in deploy_1.items():
            piece = Piece(piece_type, player=1, piece_id=pid)
            self.pieces[1][pid] = piece
            self.board[position] = piece
            self.position_of[1][pid] = position
            pid += 1

        # 游戏状态
        self.current_player = 0  # 当前行动方
        self.turn_count = 0
        self.result = GameResult.ONGOING
        self.move_history: List[MoveEvent] = []
        self.max_turns = 500  # 最大回合数

        # 每方的存活棋子计数 {PieceType: remaining_count}
        self.alive_counts: Dict[int, Dict[PieceType, int]] = {
            0: dict(PIECE_COUNTS),
            1: dict(PIECE_COUNTS),
        }

    # ──────────────────────────────────────
    #  位置查询
    # ──────────────────────────────────────

    def get_piece_at(self, position: int) -> Optional[Piece]:
        return self.board[position]

    def get_self_positions(self, player: int) -> Set[int]:
        """获取指定玩家所有存活棋子的位置"""
        return {self.position_of[player][pid]
                for pid, piece in self.pieces[player].items()
                if piece.alive}

    def get_opponent_positions(self, player: int) -> Set[int]:
        return self.get_self_positions(1 - player)

    def get_occupied_positions(self) -> Set[int]:
        return self.get_self_positions(0) | self.get_self_positions(1)

    # ──────────────────────────────────────
    #  合法动作
    # ──────────────────────────────────────

    def get_legal_actions(self, player: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        获取当前玩家的所有合法动作。
        
        Returns:
            List of (from_pos, to_pos) tuples
        """
        if player is None:
            player = self.current_player

        actions = []
        self_positions = self.get_self_positions(player)
        opponent_positions = self.get_opponent_positions(player)

        for pid, piece in self.pieces[player].items():
            if not piece.is_movable():
                continue
            from_pos = self.position_of[player][pid]
            is_engineer = (piece.piece_type == PieceType.ENGINEER)

            targets = Board.get_all_reachable(
                from_pos, self_positions, opponent_positions, is_engineer
            )

            for to_pos in targets:
                # 不能攻击行营中的棋子
                if to_pos in ALL_CAMPS and to_pos in opponent_positions:
                    continue
                # 不能进入己方大本营（已被占据或不该进）
                # 实际上大本营在游戏中不能再进入（布阵后锁定）
                # 简化：只有对方可以进入你的大本营来夺旗
                if Board.is_headquarters(to_pos, player):
                    continue
                actions.append((from_pos, to_pos))

        return actions

    def get_action_mask(self, player: Optional[int] = None) -> 'numpy.ndarray':
        """
        获取 60×60 = 3600 的动作掩码。
        
        Returns:
            numpy array of shape (3600,), 1 for legal, 0 for illegal
        """
        import numpy as np
        mask = np.zeros(NUM_POSITIONS * NUM_POSITIONS, dtype=np.float32)
        for from_pos, to_pos in self.get_legal_actions(player):
            mask[from_pos * NUM_POSITIONS + to_pos] = 1.0
        return mask

    def action_to_index(self, from_pos: int, to_pos: int) -> int:
        return from_pos * NUM_POSITIONS + to_pos

    def index_to_action(self, index: int) -> Tuple[int, int]:
        return index // NUM_POSITIONS, index % NUM_POSITIONS

    # ──────────────────────────────────────
    #  执行动作
    # ──────────────────────────────────────

    def step(self, from_pos: int, to_pos: int) -> MoveEvent:
        """
        执行一步移动。
        
        Args:
            from_pos: 起始位置
            to_pos: 目标位置
        
        Returns:
            MoveEvent 记录本次移动的完整信息
        """
        player = self.current_player
        attacker_piece = self.board[from_pos]

        assert attacker_piece is not None, f"No piece at position {from_pos}"
        assert attacker_piece.player == player, f"Not your piece at {from_pos}"
        assert attacker_piece.is_movable(), f"Piece at {from_pos} cannot move"

        event = MoveEvent(
            player=player,
            from_pos=from_pos,
            to_pos=to_pos,
            attacker=attacker_piece,
        )

        # 判断是否铁路移动
        if Board.is_railway(from_pos) and Board.is_railway(to_pos):
            r1, c1 = row_col(from_pos)
            r2, c2 = row_col(to_pos)
            if abs(r1 - r2) + abs(c1 - c2) > 1:
                event.is_rail_move = True
                if r1 != r2 and c1 != c2:
                    event.is_rail_turn = True

        defender_piece = self.board[to_pos]

        if defender_piece is None:
            # 移动到空位
            self._move_piece(attacker_piece, from_pos, to_pos)
        elif defender_piece.player != player:
            # 攻击对方棋子
            event.defender = defender_piece
            result = battle(attacker_piece.piece_type, defender_piece.piece_type)
            event.battle_result = result

            if result == BattleResult.ATTACKER_WIN:
                # 检查是否夺旗
                if defender_piece.piece_type == PieceType.FLAG:
                    event.flag_captured = True
                self._kill_piece(defender_piece)
                self._move_piece(attacker_piece, from_pos, to_pos)
            elif result == BattleResult.DEFENDER_WIN:
                self._kill_piece(attacker_piece)
            elif result == BattleResult.BOTH_DIE:
                self._kill_piece(attacker_piece)
                self._kill_piece(defender_piece)

            # 标记双方棋子已暴露
            attacker_piece.revealed = True
            if defender_piece.alive:
                defender_piece.revealed = True
        else:
            raise ValueError(f"Cannot attack own piece at {to_pos}")

        # 更新游戏状态
        self.move_history.append(event)
        self.turn_count += 1

        # 判定胜负
        self._check_game_result(event)

        # 切换玩家
        if self.result == GameResult.ONGOING:
            self.current_player = 1 - self.current_player
            # 检查下一个玩家是否有合法动作
            if not self.get_legal_actions(self.current_player):
                # 无子可动 → 该玩家输
                if self.current_player == 0:
                    self.result = GameResult.PLAYER1_WIN
                else:
                    self.result = GameResult.PLAYER0_WIN

        return event

    def step_by_index(self, action_index: int) -> MoveEvent:
        """通过动作索引执行移动"""
        from_pos, to_pos = self.index_to_action(action_index)
        return self.step(from_pos, to_pos)

    # ──────────────────────────────────────
    #  内部方法
    # ──────────────────────────────────────

    def _move_piece(self, piece: Piece, from_pos: int, to_pos: int):
        """移动棋子"""
        self.board[from_pos] = None
        self.board[to_pos] = piece
        self.position_of[piece.player][piece.piece_id] = to_pos

    def _kill_piece(self, piece: Piece):
        """消灭棋子"""
        piece.alive = False
        pos = self.position_of[piece.player].pop(piece.piece_id, None)
        if pos is not None:
            if self.board[pos] == piece:
                self.board[pos] = None
        self.alive_counts[piece.player][piece.piece_type] -= 1

    def _check_game_result(self, event: MoveEvent):
        """检查胜负"""
        # 超过最大回合数 → 平局
        if self.turn_count >= self.max_turns:
            self.result = GameResult.DRAW
            return

        # 军旗被夺
        if event.flag_captured:
            if event.defender.player == 0:
                self.result = GameResult.PLAYER1_WIN
            else:
                self.result = GameResult.PLAYER0_WIN
            return

    # ──────────────────────────────────────
    #  观察接口
    # ──────────────────────────────────────

    def get_observation(self, player: int) -> dict:
        """
        获取指定玩家视角的观察信息（不完全信息）。
        
        Returns:
            dict with:
            - 'my_pieces': {position: PieceType} 己方所有棋子
            - 'opponent_positions': set of positions with opponent pieces (类型未知)
            - 'opponent_revealed': {position: PieceType} 已暴露的对方棋子
            - 'alive_counts': {player: {PieceType: count}} 双方存活计数
            - 'move_history': recent moves
            - 'current_player': int
            - 'turn_count': int
        """
        opponent = 1 - player
        
        my_pieces = {}
        for pid, piece in self.pieces[player].items():
            if piece.alive:
                p = self.position_of[player][pid]
                my_pieces[p] = piece.piece_type

        opp_positions = set()
        opp_revealed = {}
        for pid, piece in self.pieces[opponent].items():
            if piece.alive:
                p = self.position_of[opponent][pid]
                opp_positions.add(p)
                if piece.revealed:
                    opp_revealed[p] = piece.piece_type

        return {
            'my_pieces': my_pieces,
            'opponent_positions': opp_positions,
            'opponent_revealed': opp_revealed,
            'alive_counts': {
                player: dict(self.alive_counts[player]),
                opponent: dict(self.alive_counts[opponent]),
            },
            'move_history': list(self.move_history[-8:]),  # 最近8步
            'current_player': self.current_player,
            'turn_count': self.turn_count,
        }

    def get_full_state(self) -> dict:
        """获取完整游戏状态（含所有信息，用于训练时计算 belief 标签）"""
        all_pieces = {}
        for player in [0, 1]:
            for pid, piece in self.pieces[player].items():
                if piece.alive:
                    p = self.position_of[player][pid]
                    all_pieces[p] = (piece.piece_type, piece.player, pid)
        return {
            'all_pieces': all_pieces,
            'alive_counts': copy.deepcopy(self.alive_counts),
            'current_player': self.current_player,
            'turn_count': self.turn_count,
            'result': self.result,
        }

    def clone(self) -> 'GameState':
        """深拷贝当前游戏状态"""
        return copy.deepcopy(self)

    def __repr__(self):
        return (f"GameState(turn={self.turn_count}, player={self.current_player}, "
                f"result={self.result.name})")

    def render(self, perspective: Optional[int] = None):
        """
        文本渲染棋盘（调试用）。
        
        Args:
            perspective: 以哪个玩家的视角渲染。None=全知视角。
        """
        print(f"\n{'='*40}")
        print(f"Turn {self.turn_count} | Player {self.current_player}'s move | {self.result.name}")
        print(f"{'='*40}")

        for r in range(ROWS):
            row_str = []
            for c in range(COLS):
                p = pos(r, c)
                piece = self.board[p]
                if piece is None:
                    if p in ALL_CAMPS:
                        row_str.append(" [营] ")
                    elif any(p in hqs for hqs in HEADQUARTERS.values()):
                        row_str.append(" [本] ")
                    else:
                        row_str.append("  ·   ")
                else:
                    if perspective is not None and piece.player != perspective and not piece.revealed:
                        # 未暴露的对方棋子显示为 "?"
                        marker = "?" if piece.player == 1 else "¿"
                        row_str.append(f" {piece.player}{marker}   ")
                    else:
                        name = PIECE_NAMES[piece.piece_type][:2]
                        row_str.append(f" {piece.player}{name} ")
                
            print(f"  R{r:2d}: {'|'.join(row_str)}")
            if r == 5:
                print(f"  {'─' * 38}")

        print(f"{'='*40}\n")
