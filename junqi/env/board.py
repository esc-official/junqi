"""
棋盘拓扑定义。
两人对战军棋棋盘：5列×12行 = 60个位置。
定义所有位置属性、邻接关系、铁路/公路连接。
"""
from typing import List, Set, Dict, Tuple, Optional
from collections import deque

# ──────────────────────────────────────────────
#  棋盘常量
# ──────────────────────────────────────────────
ROWS = 12
COLS = 5
NUM_POSITIONS = ROWS * COLS  # 60

def pos(row: int, col: int) -> int:
    """(row, col) → 位置索引"""
    return row * COLS + col

def row_col(position: int) -> Tuple[int, int]:
    """位置索引 → (row, col)"""
    return position // COLS, position % COLS


# ──────────────────────────────────────────────
#  特殊位置
# ──────────────────────────────────────────────

# 大本营（每方2个，军旗必须放在其中一个）
# 对方 (player 1) 的大本营在上方，己方 (player 0) 在下方
HEADQUARTERS = {
    0: (pos(10, 1), pos(10, 3)),  # player 0 底部
    1: (pos(1, 1), pos(1, 3)),    # player 1 顶部
}

# 行营（安全位置，共5个/每方）—— 棋子进入后不能被攻击
CAMPS = {
    0: frozenset({pos(7, 1), pos(7, 3), pos(8, 2), pos(9, 1), pos(9, 3)}),
    1: frozenset({pos(2, 1), pos(2, 3), pos(3, 2), pos(4, 1), pos(4, 3)}),
}
ALL_CAMPS = CAMPS[0] | CAMPS[1]

# 每方领地行范围
TERRITORY_ROWS = {
    0: range(6, 12),   # player 0: rows 6-11 (底部)
    1: range(0, 6),    # player 1: rows 0-5  (顶部)
}

# 后两排（地雷放置区域）
BACK_TWO_ROWS = {
    0: {r for r in range(10, 12)},  # player 0: rows 10-11
    1: {r for r in range(0, 2)},    # player 1: rows 0-1
}

# 最后一排（炸弹禁止放置区域）
LAST_ROW = {
    0: 11,  # player 0 最后一排
    1: 0,   # player 1 最后一排
}


# ──────────────────────────────────────────────
#  铁路线定义
# ──────────────────────────────────────────────

def _build_railway_positions() -> Set[int]:
    """铁路上的所有位置"""
    positions = set()
    # 外围矩形铁路
    for c in range(COLS):
        positions.add(pos(0, c))    # 顶边
        positions.add(pos(5, c))    # 上方前沿
        positions.add(pos(6, c))    # 下方前沿
        positions.add(pos(11, c))   # 底边
    for r in range(ROWS):
        positions.add(pos(r, 0))    # 左边
        positions.add(pos(r, 4))    # 右边
    # 中间纵向铁路 (第3列 col=2, 覆盖 rows 0-11)
    for r in range(ROWS):
        positions.add(pos(r, 2))
    return positions

RAILWAY_POSITIONS = _build_railway_positions()


def _build_railway_segments() -> List[List[int]]:
    """
    铁路线段：每条线段是一组相邻铁路位置的有序列表。
    棋子可沿线段方向移动任意距离（无障碍时）。
    """
    segments = []
    
    # 水平铁路线段
    # Row 0: 全行
    segments.append([pos(0, c) for c in range(COLS)])
    # Row 5: 全行
    segments.append([pos(5, c) for c in range(COLS)])
    # Row 6: 全行
    segments.append([pos(6, c) for c in range(COLS)])
    # Row 11: 全行
    segments.append([pos(11, c) for c in range(COLS)])

    # 垂直铁路线段
    # Col 0: 全列
    segments.append([pos(r, 0) for r in range(ROWS)])
    # Col 4: 全列
    segments.append([pos(r, 4) for r in range(ROWS)])
    # Col 2: 全列（中间纵向铁路）
    segments.append([pos(r, 2) for r in range(ROWS)])

    return segments

RAILWAY_SEGMENTS = _build_railway_segments()


# ──────────────────────────────────────────────
#  邻接关系
# ──────────────────────────────────────────────

def _build_road_adjacency() -> Dict[int, Set[int]]:
    """
    公路邻接（一步可达的正交相邻位置）。
    不含行营对角线连接和铁路远距离移动。
    """
    adj: Dict[int, Set[int]] = {i: set() for i in range(NUM_POSITIONS)}

    for r in range(ROWS):
        for c in range(COLS):
            p = pos(r, c)
            # 上
            if r > 0:
                adj[p].add(pos(r - 1, c))
            # 下
            if r < ROWS - 1:
                adj[p].add(pos(r + 1, c))
            # 左
            if c > 0:
                adj[p].add(pos(r, c - 1))
            # 右
            if c < COLS - 1:
                adj[p].add(pos(r, c + 1))

    return adj


def _build_camp_diagonals() -> Dict[int, Set[int]]:
    """
    行营对角线连接。
    行营位置与其对角方向的邻居相连。
    """
    diag: Dict[int, Set[int]] = {i: set() for i in range(NUM_POSITIONS)}

    for camp_pos in ALL_CAMPS:
        r, c = row_col(camp_pos)
        # 四个对角方向
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                neighbor = pos(nr, nc)
                diag[camp_pos].add(neighbor)
                diag[neighbor].add(camp_pos)

    return diag


# 预编译邻接表
_ROAD_ADJ = _build_road_adjacency()
_CAMP_DIAG = _build_camp_diagonals()

# 合并的一步邻接（公路 + 行营对角线）
ONE_STEP_ADJ: Dict[int, Set[int]] = {}
for i in range(NUM_POSITIONS):
    ONE_STEP_ADJ[i] = _ROAD_ADJ[i] | _CAMP_DIAG[i]


# ──────────────────────────────────────────────
#  Board 类
# ──────────────────────────────────────────────

class Board:
    """棋盘拓扑查询工具（无状态，纯拓扑信息）"""

    @staticmethod
    def is_camp(position: int) -> bool:
        return position in ALL_CAMPS

    @staticmethod
    def is_headquarters(position: int, player: int) -> bool:
        return position in HEADQUARTERS[player]

    @staticmethod
    def is_railway(position: int) -> bool:
        return position in RAILWAY_POSITIONS

    @staticmethod
    def in_territory(position: int, player: int) -> bool:
        r, _ = row_col(position)
        return r in TERRITORY_ROWS[player]

    @staticmethod
    def in_back_two_rows(position: int, player: int) -> bool:
        r, _ = row_col(position)
        return r in BACK_TWO_ROWS[player]

    @staticmethod
    def in_last_row(position: int, player: int) -> bool:
        r, _ = row_col(position)
        return r == LAST_ROW[player]

    @staticmethod
    def get_one_step_neighbors(position: int) -> Set[int]:
        """获取一步可达邻居（公路 + 行营对角线）"""
        return ONE_STEP_ADJ[position]

    @staticmethod
    def get_railway_reachable(
        position: int,
        occupied: Set[int],
        is_engineer: bool = False,
    ) -> Set[int]:
        """
        获取从 position 出发沿铁路可达的所有位置。
        
        Args:
            position: 起始位置
            occupied: 当前有棋子的所有位置集合（阻挡移动）
            is_engineer: 是否为工兵（工兵可在铁路上拐弯）
        
        Returns:
            铁路可达位置集合（不含起始位置）
        """
        if position not in RAILWAY_POSITIONS:
            return set()

        if is_engineer:
            return Board._engineer_railway_bfs(position, occupied)
        else:
            return Board._normal_railway_reach(position, occupied)

    @staticmethod
    def _normal_railway_reach(position: int, occupied: Set[int]) -> Set[int]:
        """
        非工兵棋子的铁路移动：只能沿线段直行，不能拐弯。
        在每条经过 position 的线段上，向两个方向延伸直到遇到障碍。
        """
        reachable = set()

        for segment in RAILWAY_SEGMENTS:
            if position not in segment:
                continue
            idx = segment.index(position)

            # 向前（index增大方向）
            for i in range(idx + 1, len(segment)):
                p = segment[i]
                if p in ALL_CAMPS:
                    # 不能穿越行营（但可以停在行营？
                    # 标准规则：铁路上行营位置也是经过的，但棋子不能在行营上\"经过\"
                    # 简化处理：行营阻断铁路
                    break
                if p in occupied:
                    break
                reachable.add(p)

            # 向后（index减小方向）
            for i in range(idx - 1, -1, -1):
                p = segment[i]
                if p in ALL_CAMPS:
                    break
                if p in occupied:
                    break
                reachable.add(p)

        return reachable

    @staticmethod
    def _engineer_railway_bfs(position: int, occupied: Set[int]) -> Set[int]:
        """
        工兵的铁路移动：可以在铁路上自由移动，包括拐弯。
        使用 BFS 在铁路网络上搜索所有可达位置。
        """
        reachable = set()
        visited = {position}
        queue = deque([position])

        # 构建铁路邻接表（只含铁路位置之间的相邻连接）
        while queue:
            curr = queue.popleft()
            r, c = row_col(curr)

            # 检查相邻铁路位置
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    neighbor = pos(nr, nc)
                    if neighbor in visited:
                        continue
                    if neighbor not in RAILWAY_POSITIONS:
                        continue
                    if neighbor in ALL_CAMPS:
                        continue
                    visited.add(neighbor)
                    if neighbor in occupied:
                        # 被阻挡但标记已访问，不继续扩展
                        continue
                    reachable.add(neighbor)
                    queue.append(neighbor)

        return reachable

    @staticmethod
    def get_all_reachable(
        position: int,
        occupied_by_self: Set[int],
        occupied_by_opponent: Set[int],
        is_engineer: bool = False,
    ) -> Set[int]:
        """
        获取从 position 出发的所有合法目标位置。
        
        Args:
            position: 起始位置
            occupied_by_self: 己方棋子占据的位置集合
            occupied_by_opponent: 对方棋子占据的位置集合
            is_engineer: 是否为工兵
        
        Returns:
            合法目标位置集合（可移动到的空位 + 可攻击的对方棋子位置）
        """
        all_occupied = occupied_by_self | occupied_by_opponent
        targets = set()

        # 1. 一步可达（公路 + 行营对角线）
        for neighbor in Board.get_one_step_neighbors(position):
            if neighbor in occupied_by_self:
                continue
            # 不能进入对方大本营外的安全区域... 
            # 实际上行营任何人都能进，但进入行营后的棋子不能被攻击
            targets.add(neighbor)

        # 2. 铁路可达（多步直行或工兵BFS）
        rail_targets = Board.get_railway_reachable(
            position, all_occupied - {position}, is_engineer
        )
        # 铁路终点不能是己方占据的位置
        for rt in rail_targets:
            if rt not in occupied_by_self:
                targets.add(rt)
        
        # 从目标中移除：不能进入己方大本营（大本营只在布阵时使用）
        # 注意：可以进入对方大本营去夺旗
        # 这里的逻辑取决于具体规则变体
        
        return targets

    @staticmethod
    def get_deployable_positions(player: int) -> List[int]:
        """
        获取指定玩家的所有可布阵位置（领地内非行营位置）。
        恰好25个位置放25个棋子。
        """
        positions = []
        for r in TERRITORY_ROWS[player]:
            for c in range(COLS):
                p = pos(r, c)
                if p not in CAMPS[player]:
                    positions.append(p)
        return sorted(positions)

    @staticmethod
    def get_mine_positions(player: int) -> List[int]:
        """获取地雷可放置的位置（后两排非行营位置）"""
        positions = []
        for r in BACK_TWO_ROWS[player]:
            for c in range(COLS):
                p = pos(r, c)
                if p not in CAMPS[player]:
                    positions.append(p)
        return sorted(positions)

    @staticmethod
    def get_bomb_forbidden(player: int) -> Set[int]:
        """获取炸弹不能放置的位置（最后一排）"""
        return {pos(LAST_ROW[player], c) for c in range(COLS)}

    @staticmethod
    def display_positions():
        """打印棋盘位置编号（调试用）"""
        print("=" * 35)
        for r in range(ROWS):
            row_str = []
            for c in range(COLS):
                p = pos(r, c)
                marker = ""
                if p in ALL_CAMPS:
                    marker = "*"
                elif p in HEADQUARTERS[0] or p in HEADQUARTERS[1]:
                    marker = "#"
                row_str.append(f"{p:2d}{marker:<1s}")
            sep = "---" if r == 5 else "   "
            print(f"  Row{r:2d}:  {'   '.join(row_str)}")
            if r == 5:
                print("  " + "-" * 31)
        print("=" * 35)
        print("  * = 行营(Camp)  # = 大本营(HQ)")
