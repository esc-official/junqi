"""
50个固定布阵模板。
每个模板是一个 dict[int, PieceType]，映射位置索引到棋子类型。
模板以 player 0 (底部, rows 6-11) 的视角定义。
使用时可通过镜像翻转来适配 player 1。
"""
import random
from typing import Dict, List
from .pieces import PieceType as PT, PIECE_COUNTS
from .board import Board, HEADQUARTERS, pos, row_col, COLS, ROWS


def _mirror_template(template: Dict[int, PT]) -> Dict[int, PT]:
    """将 player 0 的模板镜像翻转为 player 1 的模板"""
    mirrored = {}
    for position, piece_type in template.items():
        r, c = row_col(position)
        new_r = (ROWS - 1) - r  # 行翻转
        mirrored[pos(new_r, c)] = piece_type
    return mirrored


def _generate_template(seed: int) -> Dict[int, PT]:
    """
    基于种子生成一个合法布阵（player 0 视角, rows 6-11）。
    
    约束:
    - 军旗必须放在大本营 (pos 51 or 53)
    - 地雷只能放在后两排 (rows 10-11)
    - 炸弹不能放在最后一排 (row 11)
    - 每个非行营位置恰好放一个棋子
    """
    rng = random.Random(seed)
    player = 0

    # 获取可布阵位置
    all_positions = Board.get_deployable_positions(player)  # 25个位置
    mine_positions = set(Board.get_mine_positions(player))
    bomb_forbidden = Board.get_bomb_forbidden(player)
    hq_positions = set(HEADQUARTERS[player])

    # 准备棋子清单
    pieces: List[PT] = []
    for pt, count in PIECE_COUNTS.items():
        pieces.extend([pt] * count)
    assert len(pieces) == 25

    template: Dict[int, PT] = {}
    used_positions = set()

    # 1. 放置军旗（必须在大本营）
    flag_pos = rng.choice(list(hq_positions))
    template[flag_pos] = PT.FLAG
    used_positions.add(flag_pos)
    pieces.remove(PT.FLAG)

    # 2. 放置地雷（只能在后两排）
    mine_available = [p for p in mine_positions if p not in used_positions]
    rng.shuffle(mine_available)
    mine_count = PIECE_COUNTS[PT.LANDMINE]  # 3
    for i in range(mine_count):
        mp = mine_available[i]
        template[mp] = PT.LANDMINE
        used_positions.add(mp)
        pieces.remove(PT.LANDMINE)

    # 3. 放置炸弹（不能在最后一排）
    bomb_available = [p for p in all_positions
                      if p not in used_positions and p not in bomb_forbidden]
    rng.shuffle(bomb_available)
    bomb_count = PIECE_COUNTS[PT.BOMB]  # 2
    for i in range(bomb_count):
        bp = bomb_available[i]
        template[bp] = PT.BOMB
        used_positions.add(bp)
        pieces.remove(PT.BOMB)

    # 4. 放置剩余棋子到剩余位置
    remaining_positions = [p for p in all_positions if p not in used_positions]
    rng.shuffle(remaining_positions)
    rng.shuffle(pieces)
    assert len(remaining_positions) == len(pieces)
    for position, piece_type in zip(remaining_positions, pieces):
        template[position] = piece_type

    return template


def _generate_strategic_template(seed: int, strategy: str) -> Dict[int, PT]:
    """
    基于策略偏好生成布阵模板。
    
    策略:
    - 'aggressive': 大子靠前，准备进攻
    - 'defensive': 大子护旗，地雷封死大本营
    - 'balanced': 攻守兼备
    - 'tricky': 小子靠前伪装大子，大子隐藏
    """
    rng = random.Random(seed)
    player = 0

    all_positions = Board.get_deployable_positions(player)
    mine_positions = set(Board.get_mine_positions(player))
    bomb_forbidden = Board.get_bomb_forbidden(player)
    hq_positions = list(HEADQUARTERS[player])

    template: Dict[int, PT] = {}
    used_positions = set()
    
    # ── 军旗 ──
    flag_pos = rng.choice(hq_positions)
    template[flag_pos] = PT.FLAG
    used_positions.add(flag_pos)
    other_hq = [h for h in hq_positions if h != flag_pos][0]

    # ── 地雷布局 ──
    mine_slots = [p for p in mine_positions if p not in used_positions]
    
    if strategy in ('defensive', 'balanced'):
        # 防守型：地雷优先保护军旗周围
        flag_r, flag_c = row_col(flag_pos)
        # 优先放在军旗同列或相邻列的后排
        priority = sorted(mine_slots,
                          key=lambda p: abs(row_col(p)[1] - flag_c))
        mine_chosen = priority[:3]
    else:
        rng.shuffle(mine_slots)
        mine_chosen = mine_slots[:3]

    for mp in mine_chosen:
        template[mp] = PT.LANDMINE
        used_positions.add(mp)

    # ── 炸弹布局 ──
    bomb_slots = [p for p in all_positions
                  if p not in used_positions and p not in bomb_forbidden]

    if strategy == 'defensive':
        # 炸弹护旗：放在大本营旁边
        bomb_slots.sort(key=lambda p: abs(row_col(p)[0] - row_col(flag_pos)[0])
                        + abs(row_col(p)[1] - row_col(flag_pos)[1]))
    elif strategy == 'aggressive':
        # 炸弹靠前：准备同归于尽
        bomb_slots.sort(key=lambda p: row_col(p)[0])  # 越前排越优先
    else:
        rng.shuffle(bomb_slots)

    for bp in bomb_slots[:2]:
        template[bp] = PT.BOMB
        used_positions.add(bp)

    # ── 分配棋子池 ──
    remaining_pieces: List[PT] = []
    for pt, count in PIECE_COUNTS.items():
        if pt in (PT.FLAG, PT.LANDMINE, PT.BOMB):
            continue
        remaining_pieces.extend([pt] * count)

    remaining_positions = [p for p in all_positions if p not in used_positions]

    # 按策略排序棋子和位置
    big_pieces = [PT.COMMANDER, PT.CORPS, PT.DIVISION, PT.BRIGADIER]
    mid_pieces = [PT.COLONEL, PT.CAPTAIN]
    small_pieces = [PT.LIEUTENANT, PT.PRIVATE, PT.ENGINEER]

    if strategy == 'aggressive':
        # 大子靠前排
        remaining_positions.sort(key=lambda p: row_col(p)[0])
        # 大子先放
        order = big_pieces + mid_pieces + small_pieces
    elif strategy == 'defensive':
        # 大子靠后排
        remaining_positions.sort(key=lambda p: -row_col(p)[0])
        order = big_pieces + mid_pieces + small_pieces
    elif strategy == 'tricky':
        # 小子占据前排，大子隐藏在中间
        remaining_positions.sort(key=lambda p: row_col(p)[0])
        order = small_pieces + big_pieces + mid_pieces
    else:  # balanced
        rng.shuffle(remaining_positions)
        order = []
        # 交替大小子
        bigs = [p for p in remaining_pieces if p in big_pieces]
        smalls = [p for p in remaining_pieces if p in small_pieces]
        mids = [p for p in remaining_pieces if p in mid_pieces]
        rng.shuffle(bigs)
        rng.shuffle(smalls)
        rng.shuffle(mids)
        order = bigs + mids + smalls

    # 按 order 排列 remaining_pieces
    sorted_pieces = []
    piece_pool = list(remaining_pieces)
    for pt in order:
        if pt in piece_pool:
            sorted_pieces.append(pt)
            piece_pool.remove(pt)
    sorted_pieces.extend(piece_pool)  # 剩余的追加
    rng.shuffle(sorted_pieces)  # 加入一点随机性

    # 工兵特殊处理：至少一个工兵靠前（用于排雷）
    engineer_indices = [i for i, p in enumerate(sorted_pieces) if p == PT.ENGINEER]
    front_positions = [i for i, p in enumerate(remaining_positions)
                       if row_col(p)[0] <= 7]
    if engineer_indices and front_positions:
        # 把第一个工兵换到前排位置
        ei = engineer_indices[0]
        fi = front_positions[0] if front_positions[0] != ei else (
            front_positions[1] if len(front_positions) > 1 else ei)
        if ei != fi and fi < len(sorted_pieces):
            sorted_pieces[ei], sorted_pieces[fi] = sorted_pieces[fi], sorted_pieces[ei]

    for position, piece_type in zip(remaining_positions, sorted_pieces):
        template[position] = piece_type

    assert len(template) == 25
    return template


# ──────────────────────────────────────────────
#  生成50个固定模板
# ──────────────────────────────────────────────

_STRATEGIES = ['aggressive', 'defensive', 'balanced', 'tricky']

TEMPLATES: List[Dict[int, PT]] = []

for i in range(50):
    strategy = _STRATEGIES[i % len(_STRATEGIES)]
    seed = 42_0000 + i * 137  # 确定性种子
    tmpl = _generate_strategic_template(seed, strategy)
    TEMPLATES.append(tmpl)


def get_template(index: int, player: int = 0) -> Dict[int, PT]:
    """
    获取指定索引的布阵模板。
    
    Args:
        index: 模板索引 (0-49)
        player: 玩家编号 (0=底部, 1=顶部)
    
    Returns:
        位置→棋子类型的映射
    """
    tmpl = TEMPLATES[index % len(TEMPLATES)]
    if player == 1:
        return _mirror_template(tmpl)
    return dict(tmpl)


def get_random_template(player: int = 0, rng: random.Random = None) -> Dict[int, PT]:
    """随机选择一个布阵模板"""
    if rng is None:
        idx = random.randint(0, len(TEMPLATES) - 1)
    else:
        idx = rng.randint(0, len(TEMPLATES) - 1)
    return get_template(idx, player)


def validate_template(template: Dict[int, PT], player: int = 0) -> bool:
    """
    验证布阵模板是否合法。
    
    Returns:
        True if valid
    """
    deployable = set(Board.get_deployable_positions(player))
    mine_positions = set(Board.get_mine_positions(player))
    bomb_forbidden = Board.get_bomb_forbidden(player)
    hq_positions = set(HEADQUARTERS[player])

    # 检查位置合法性
    if set(template.keys()) != deployable:
        return False

    # 检查棋子数量
    counts: Dict[PT, int] = {}
    for pt in template.values():
        counts[pt] = counts.get(pt, 0) + 1
    if counts != PIECE_COUNTS:
        return False

    # 军旗在大本营
    flag_positions = [p for p, t in template.items() if t == PT.FLAG]
    if len(flag_positions) != 1 or flag_positions[0] not in hq_positions:
        return False

    # 地雷在后两排
    for p, t in template.items():
        if t == PT.LANDMINE and p not in mine_positions:
            return False

    # 炸弹不在最后一排
    for p, t in template.items():
        if t == PT.BOMB and p in bomb_forbidden:
            return False

    return True
