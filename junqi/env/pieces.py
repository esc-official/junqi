"""
棋子定义、等级体系和交战规则。
军棋共12种棋子，每方25枚。
"""
from enum import IntEnum
from typing import Tuple, Optional


class PieceType(IntEnum):
    """棋子类型枚举，数值同时代表等级（越大越强）"""
    NONE = -1       # 空位
    FLAG = 0        # 军旗 (不可移动)
    LANDMINE = 1    # 地雷 (不可移动)
    BOMB = 2        # 炸弹 (与任何棋子同归于尽)
    ENGINEER = 3    # 工兵 (可排雷, 铁路上可拐弯)
    PRIVATE = 4     # 排长
    LIEUTENANT = 5  # 连长
    CAPTAIN = 6     # 营长
    COLONEL = 7     # 团长
    BRIGADIER = 8   # 旅长
    DIVISION = 9    # 师长
    CORPS = 10      # 军长
    COMMANDER = 11  # 司令


# 棋子中文名称
PIECE_NAMES = {
    PieceType.NONE: "空",
    PieceType.FLAG: "军旗",
    PieceType.LANDMINE: "地雷",
    PieceType.BOMB: "炸弹",
    PieceType.ENGINEER: "工兵",
    PieceType.PRIVATE: "排长",
    PieceType.LIEUTENANT: "连长",
    PieceType.CAPTAIN: "营长",
    PieceType.COLONEL: "团长",
    PieceType.BRIGADIER: "旅长",
    PieceType.DIVISION: "师长",
    PieceType.CORPS: "军长",
    PieceType.COMMANDER: "司令",
}

# 每方棋子数量
PIECE_COUNTS = {
    PieceType.FLAG: 1,
    PieceType.LANDMINE: 3,
    PieceType.BOMB: 2,
    PieceType.ENGINEER: 3,
    PieceType.PRIVATE: 3,
    PieceType.LIEUTENANT: 3,
    PieceType.CAPTAIN: 2,
    PieceType.COLONEL: 2,
    PieceType.BRIGADIER: 2,
    PieceType.DIVISION: 2,
    PieceType.CORPS: 1,
    PieceType.COMMANDER: 1,
}

# 棋子数量总计 = 25
TOTAL_PIECES = sum(PIECE_COUNTS.values())

# 可移动棋子种类（排除军旗和地雷）
MOVABLE_TYPES = {pt for pt in PieceType if pt not in
                 (PieceType.NONE, PieceType.FLAG, PieceType.LANDMINE)}

# 战斗棋子种类（排除不动子和特殊子）
RANK_TYPES = [pt for pt in PieceType if pt.value >= PieceType.ENGINEER.value]

# 棋子等级（用于比较大小），值越大等级越高
PIECE_RANKS = {pt: pt.value for pt in PieceType if pt.value >= 0}

# 棋子基础战略价值（用于奖励计算）
PIECE_BASE_VALUE = {
    PieceType.FLAG: 100.0,
    PieceType.LANDMINE: 4.0,
    PieceType.BOMB: 7.0,
    PieceType.ENGINEER: 3.0,
    PieceType.PRIVATE: 1.5,
    PieceType.LIEUTENANT: 2.0,
    PieceType.CAPTAIN: 3.0,
    PieceType.COLONEL: 4.0,
    PieceType.BRIGADIER: 5.0,
    PieceType.DIVISION: 6.0,
    PieceType.CORPS: 8.0,
    PieceType.COMMANDER: 10.0,
}

NUM_PIECE_TYPES = 12  # 不含 NONE


class BattleResult(IntEnum):
    """交战结果"""
    ATTACKER_WIN = 1    # 进攻方获胜
    DEFENDER_WIN = 2    # 防守方获胜
    BOTH_DIE = 3        # 同归于尽
    INVALID = 0         # 非法交战


def battle(attacker: PieceType, defender: PieceType) -> BattleResult:
    """
    计算两个棋子交战的结果。
    
    规则:
    1. 炸弹与任何棋子交遇 → 同归于尽
    2. 工兵碰地雷 → 工兵赢（排雷）
    3. 其他子碰地雷 → 同归于尽
    4. 同级棋子 → 同归于尽
    5. 否则大子吃小子
    
    Args:
        attacker: 进攻方棋子类型
        defender: 防守方棋子类型
    
    Returns:
        BattleResult
    """
    # 不能攻击军旗前方的地雷之前的棋子... 简化为直接交战
    # 军旗不能主动攻击
    if attacker == PieceType.FLAG or attacker == PieceType.LANDMINE:
        return BattleResult.INVALID

    # 进攻方是炸弹 → 同归于尽（炸弹可以主动攻击任何棋子）
    if attacker == PieceType.BOMB:
        return BattleResult.BOTH_DIE

    # 防守方是炸弹 → 同归于尽
    if defender == PieceType.BOMB:
        return BattleResult.BOTH_DIE

    # 防守方是地雷
    if defender == PieceType.LANDMINE:
        if attacker == PieceType.ENGINEER:
            return BattleResult.ATTACKER_WIN  # 工兵排雷
        else:
            return BattleResult.BOTH_DIE  # 其他子碰地雷同归于尽

    # 防守方是军旗 → 进攻方获胜（夺旗）
    if defender == PieceType.FLAG:
        return BattleResult.ATTACKER_WIN

    # 常规比较：比等级
    if attacker.value > defender.value:
        return BattleResult.ATTACKER_WIN
    elif attacker.value < defender.value:
        return BattleResult.DEFENDER_WIN
    else:
        return BattleResult.BOTH_DIE  # 同级同归于尽


class Piece:
    """棋子实例"""
    __slots__ = ('piece_type', 'player', 'piece_id', 'alive', 'revealed')

    def __init__(self, piece_type: PieceType, player: int, piece_id: int):
        """
        Args:
            piece_type: 棋子类型
            player: 所属玩家 (0 或 1)
            piece_id: 唯一标识（每方0-24）
        """
        self.piece_type = piece_type
        self.player = player
        self.piece_id = piece_id
        self.alive = True
        self.revealed = False  # 是否已被对方知晓类型

    def is_movable(self) -> bool:
        """该棋子是否可以移动"""
        return self.alive and self.piece_type in MOVABLE_TYPES

    def __repr__(self):
        status = "alive" if self.alive else "dead"
        return f"Piece({PIECE_NAMES[self.piece_type]}, P{self.player}, #{self.piece_id}, {status})"
