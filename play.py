"""
人机对战脚本。
训练完成后，使用此脚本与 AI 对弈。

使用方法:
    python play.py checkpoints/junqi_final.pt
    python play.py checkpoints/junqi_final.pt --you-first
    python play.py checkpoints/junqi_final.pt --template 5
    python play.py --no-model        # 对战随机 AI（测试用）
"""
import argparse
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from junqi.env.junqi_env import JunqiEnv
from junqi.env.game import GameState, GameResult
from junqi.env.pieces import PieceType, PIECE_NAMES, BattleResult, PIECE_BASE_VALUE
from junqi.env.board import (
    Board, NUM_POSITIONS, ROWS, COLS, ALL_CAMPS, HEADQUARTERS,
    pos, row_col, RAILWAY_POSITIONS,
)
from junqi.env.templates import TEMPLATES, get_template
from junqi.model.network import JunqiNetwork
from junqi.training.ppo import PPOAgent


# ──────────────────────────────────────────────
#  颜色 / 样式 (ANSI escape codes)
# ──────────────────────────────────────────────
class C:
    """ANSI color codes for terminal display."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[48;5;235m"
    BG_CAMP = "\033[48;5;22m"
    BG_HQ   = "\033[48;5;52m"
    BG_SEL  = "\033[48;5;24m"


# ──────────────────────────────────────────────
#  棋子显示名称（2字符宽）
# ──────────────────────────────────────────────
SHORT_NAMES = {
    PieceType.FLAG:       "旗",
    PieceType.LANDMINE:   "雷",
    PieceType.BOMB:       "炸",
    PieceType.ENGINEER:   "工",
    PieceType.PRIVATE:    "排",
    PieceType.LIEUTENANT: "连",
    PieceType.CAPTAIN:    "营",
    PieceType.COLONEL:    "团",
    PieceType.BRIGADIER:  "旅",
    PieceType.DIVISION:   "师",
    PieceType.CORPS:      "军",
    PieceType.COMMANDER:  "令",
}


# ──────────────────────────────────────────────
#  棋盘渲染
# ──────────────────────────────────────────────

def render_board(game: GameState, human_player: int = 0,
                 highlight_from: int = -1, highlight_targets: set = None):
    """
    渲染棋盘到终端。
    
    - 己方棋子：显示类型名（绿色）
    - 对方棋子：显示 "？"（红色），已暴露的显示类型名
    - 行营：深绿背景
    - 大本营：深红背景
    - 被选中的棋子：蓝色高亮
    - 可移动目标：黄色标记
    """
    if highlight_targets is None:
        highlight_targets = set()

    opponent = 1 - human_player

    print()
    # 列号标题
    print(f"  {'':>6s}", end="")
    for c in range(COLS):
        print(f"  {c}   ", end="")
    print()
    print(f"  {'':>6s}" + "------" * COLS + "-")

    for r in range(ROWS):
        # 行号
        print(f"  R{r:>2d}  |", end="")

        for c in range(COLS):
            p = pos(r, c)
            piece = game.board[p]

            # 背景色
            bg = ""
            if p == highlight_from:
                bg = C.BG_SEL
            elif p in highlight_targets:
                bg = C.BG_SEL
            elif p in ALL_CAMPS:
                bg = C.BG_CAMP
            elif any(p in hqs for hqs in HEADQUARTERS.values()):
                bg = C.BG_HQ

            if piece is None:
                if p in highlight_targets:
                    cell = f"{bg}{C.YELLOW} ·→ {C.RESET}"
                elif p in ALL_CAMPS:
                    cell = f"{bg}{C.DIM} 营  {C.RESET}"
                elif any(p in hqs for hqs in HEADQUARTERS.values()):
                    cell = f"{bg}{C.DIM} 本  {C.RESET}"
                else:
                    cell = f"{bg}  ·  {C.RESET}"
            else:
                name = SHORT_NAMES.get(piece.piece_type, "？")
                if piece.player == human_player:
                    # 己方棋子 — 始终可见
                    cell = f"{bg}{C.GREEN}{C.BOLD} {name}  {C.RESET}"
                else:
                    # 对方棋子
                    if piece.revealed:
                        cell = f"{bg}{C.RED}{C.BOLD} {name}  {C.RESET}"
                    else:
                        cell = f"{bg}{C.RED} ？ {C.RESET}"

                # 如果是可攻击目标，加标记
                if p in highlight_targets and piece.player == opponent:
                    cell = f"{bg}{C.YELLOW}{C.BOLD} ×{name}{C.RESET}"

            print(cell, end="|")

        # 行分隔
        side = ""
        if human_player == 0:
            if r <= 5:
                side = " ← AI"
            elif r == 6:
                side = " ← YOU"
        else:
            if r <= 5:
                side = " ← YOU"
            elif r == 6:
                side = " ← AI"

        print(f"  {C.DIM}{side}{C.RESET}")

        if r == 5:
            print(f"  {'':>6s}" + "======" * COLS + "=  中线")
        else:
            print(f"  {'':>6s}" + "------" * COLS + "-")

    # 列号底部
    print(f"  {'':>6s}", end="")
    for c in range(COLS):
        print(f"  {c}   ", end="")
    print()


def render_status(game: GameState, human_player: int):
    """显示游戏状态栏"""
    opponent = 1 - human_player

    # 己方存活棋子
    my_alive = []
    for pt in PieceType:
        if pt == PieceType.NONE:
            continue
        count = game.alive_counts[human_player].get(pt, 0)
        if count > 0:
            my_alive.append(f"{SHORT_NAMES[pt]}×{count}")

    # 对方已知阵亡
    opp_dead = []
    for pt in PieceType:
        if pt == PieceType.NONE:
            continue
        from junqi.env.pieces import PIECE_COUNTS
        full = PIECE_COUNTS.get(pt, 0)
        alive = game.alive_counts[opponent].get(pt, 0)
        dead = full - alive
        if dead > 0:
            opp_dead.append(f"{SHORT_NAMES[pt]}×{dead}")

    print(f"\n  {C.CYAN}回合 {game.turn_count}{C.RESET} | "
          f"{'你先手' if game.current_player == human_player else 'AI先手'}")
    print(f"  {C.GREEN}己方存活: {' '.join(my_alive)}{C.RESET}")
    if opp_dead:
        print(f"  {C.RED}对方阵亡: {' '.join(opp_dead)}{C.RESET}")


# ──────────────────────────────────────────────
#  交互逻辑
# ──────────────────────────────────────────────

def parse_position(s: str) -> int:
    """
    解析位置输入。支持格式:
    - "r,c"  例如 "6,0" → pos(6,0) = 30
    - "rc"   例如 "60"  → pos(6,0) = 30 (如果 r<10)
    - 直接数字 "30"     → position 30
    """
    s = s.strip()

    # 格式: r,c
    if "," in s:
        parts = s.split(",")
        if len(parts) == 2:
            try:
                r, c = int(parts[0].strip()), int(parts[1].strip())
                if 0 <= r < ROWS and 0 <= c < COLS:
                    return pos(r, c)
            except ValueError:
                pass

    # 直接数字 → position index
    try:
        p = int(s)
        if 0 <= p < NUM_POSITIONS:
            return p
    except ValueError:
        pass

    return -1


def show_help():
    """显示帮助信息"""
    print(f"""
  {C.BOLD}=== 军棋 人机对战 帮助 ==={C.RESET}

  {C.CYAN}位置输入格式:{C.RESET}
    行,列    例如: 6,0  (第6行第0列)
    位置编号  例如: 30   (直接输入位置号)

  {C.CYAN}操作流程:{C.RESET}
    1. 输入你要移动的棋子位置 (from)
    2. 系统显示可移动的目标位置
    3. 输入目标位置 (to)

  {C.CYAN}特殊命令:{C.RESET}
    help / h    显示帮助
    quit / q    退出游戏
    back / b    重新选择棋子（在选择目标时）
    show / s    重新显示棋盘
    legal / l   显示所有合法动作

  {C.CYAN}棋子说明:{C.RESET}
    令=司令  军=军长  师=师长  旅=旅长  团=团长
    营=营长  连=连长  排=排长  工=工兵  炸=炸弹
    雷=地雷  旗=军旗

  {C.CYAN}规则提示:{C.RESET}
    · 大子吃小子，相同同归于尽
    · 炸弹碰任何子都同归于尽
    · 工兵可排地雷，其余碰地雷同归于尽
    · 铁路上可走多步（工兵可拐弯）
    · 行营内的棋子不能被攻击
    · 夺取对方军旗即获胜
""")


def show_legal_actions(game: GameState, player: int):
    """显示所有合法动作"""
    actions = game.get_legal_actions(player)
    print(f"\n  {C.CYAN}合法动作 ({len(actions)} 个):{C.RESET}")
    for from_pos, to_pos in sorted(actions):
        fr, fc = row_col(from_pos)
        tr, tc = row_col(to_pos)
        piece = game.board[from_pos]
        target = game.board[to_pos]
        name = SHORT_NAMES.get(piece.piece_type, "?") if piece else "?"
        arrow = "→" if target is None else "×"
        print(f"    {name} ({fr},{fc})={from_pos:2d} {arrow} ({tr},{tc})={to_pos:2d}", end="")
        if target and target.player != player:
            if target.revealed:
                print(f"  [攻击 {SHORT_NAMES.get(target.piece_type, '?')}]", end="")
            else:
                print(f"  [攻击 ？]", end="")
        print()


def describe_event(event, human_player: int):
    """描述移动/交战事件"""
    fr, fc = row_col(event.from_pos)
    tr, tc = row_col(event.to_pos)

    who = "你" if event.player == human_player else "AI"
    atk_name = SHORT_NAMES.get(event.attacker.piece_type, "?") if event.attacker else "?"

    # 隐藏 AI 棋子类型（除非交战后暴露）
    if event.player != human_player and event.battle_result is None:
        atk_display = "？"
    else:
        atk_display = atk_name

    msg = f"  {C.BOLD}{who}{C.RESET} 移动 {atk_display} ({fr},{fc}) → ({tr},{tc})"

    if event.is_rail_move:
        msg += f" {C.CYAN}[铁路]{C.RESET}"
        if event.is_rail_turn:
            msg += f" {C.CYAN}[拐弯]{C.RESET}"

    print(msg)

    if event.battle_result is not None:
        def_name = SHORT_NAMES.get(event.defender.piece_type, "?") if event.defender else "?"
        atk_name_full = SHORT_NAMES.get(event.attacker.piece_type, "?")

        if event.battle_result == BattleResult.ATTACKER_WIN:
            print(f"  [>>] {C.GREEN}{atk_name_full}{C.RESET} 吃掉了 "
                  f"{C.RED}{def_name}{C.RESET}!")
        elif event.battle_result == BattleResult.DEFENDER_WIN:
            print(f"  [>>] {C.RED}{def_name}{C.RESET} 吃掉了 "
                  f"{C.GREEN}{atk_name_full}{C.RESET}!")
        elif event.battle_result == BattleResult.BOTH_DIE:
            print(f"  [!!] {atk_name_full} 与 {def_name} 同归于尽!")

        if event.flag_captured:
            print(f"  [FLAG] {C.BOLD}{C.YELLOW}军旗被夺!{C.RESET}")


# ──────────────────────────────────────────────
#  主游戏循环
# ──────────────────────────────────────────────

def play_game(
    model_path: str = None,
    human_player: int = 0,
    human_template: int = None,
    ai_template: int = None,
    device: str = "cpu",
    net_hidden_dim: int = 128,
    net_res_blocks: int = 12,
):
    """
    运行一局人机对战。
    
    Args:
        model_path: 模型检查点路径，None 则 AI 随机下
        human_player: 人类玩家编号 (0=先手底部, 1=后手顶部)
        human_template: 人类玩家指定布阵模板 (0-49), None 则随机
        ai_template: AI 指定布阵模板, None 则随机
        device: 推理设备
    """
    ai_player = 1 - human_player

    # 加载 AI 模型
    agent = None
    if model_path and os.path.exists(model_path):
        print(f"\n  {C.CYAN}加载 AI 模型: {model_path}{C.RESET}")
        network = JunqiNetwork(
            hidden_dim=net_hidden_dim,
            num_res_blocks=net_res_blocks,
        )
        agent = PPOAgent(network=network, device=device)
        agent.load(model_path)
        network.eval()
        print(f"  {C.GREEN}模型加载成功! ({network.count_parameters():,} 参数){C.RESET}")
    else:
        if model_path:
            print(f"  {C.YELLOW}模型文件不存在: {model_path}{C.RESET}")
        print(f"  {C.YELLOW}AI 将使用随机策略{C.RESET}")

    # 选择布阵
    if human_template is None:
        human_template = random.randint(0, 49)
    if ai_template is None:
        ai_template = random.randint(0, 49)

    print(f"\n  你的布阵模板: #{human_template}")
    print(f"  AI 布阵模板: #{ai_template} (隐藏)")

    # 初始化环境
    env = JunqiEnv(max_turns=500)

    if human_player == 0:
        obs, info = env.reset(
            template_idx_p0=human_template,
            template_idx_p1=ai_template,
        )
    else:
        obs, info = env.reset(
            template_idx_p0=ai_template,
            template_idx_p1=human_template,
        )

    game = env.game

    # 欢迎信息
    print(f"\n{'='*50}")
    print(f"  {C.BOLD}{C.CYAN}军棋 人机对战{C.RESET}")
    print(f"  你是 P{human_player} ({'先手' if human_player == 0 else '后手'})")
    print(f"  输入 {C.CYAN}help{C.RESET} 查看帮助")
    print(f"{'='*50}")

    # 主循环
    while game.result == GameResult.ONGOING:
        current = game.current_player

        if current == human_player:
            # ── 人类回合 ──
            render_board(game, human_player)
            render_status(game, human_player)

            action = human_turn(game, human_player)
            if action is None:
                print(f"\n  {C.YELLOW}游戏中止{C.RESET}")
                return

            event = env.step(action)
            obs = env._get_obs(human_player)
            describe_event(env.game.move_history[-1], human_player)

        else:
            # ── AI 回合 ──
            print(f"\n  {C.MAGENTA}AI 思考中...{C.RESET}")

            action_mask = env.get_action_mask(ai_player)

            if agent is not None:
                ai_obs = env._get_obs(ai_player)
                action, _, _ = agent.select_action(
                    ai_obs, action_mask, deterministic=True
                )
            else:
                # 随机 AI
                legal = np.where(action_mask > 0)[0]
                action = int(np.random.choice(legal))

            obs, reward, done, trunc, info = env.step(action)
            describe_event(env.game.move_history[-1], human_player)

    # 游戏结束
    render_board(game, human_player)
    print(f"\n{'='*50}")
    if game.result == GameResult.PLAYER0_WIN:
        winner = "你" if human_player == 0 else "AI"
    elif game.result == GameResult.PLAYER1_WIN:
        winner = "你" if human_player == 1 else "AI"
    else:
        winner = None

    if winner == "你":
        print(f"  *** {C.GREEN}{C.BOLD}恭喜, 你赢了!{C.RESET} ({game.turn_count} 回合)")
    elif winner == "AI":
        print(f"  --- {C.RED}{C.BOLD}AI 获胜{C.RESET} ({game.turn_count} 回合)")
    else:
        print(f"  === {C.YELLOW}{C.BOLD}平局{C.RESET} ({game.turn_count} 回合)")
    print(f"{'='*50}\n")


def human_turn(game: GameState, player: int) -> int:
    """
    处理人类玩家的一个回合。
    
    Returns:
        action index (int), 或 None 表示退出
    """
    actions = game.get_legal_actions(player)
    if not actions:
        print(f"  {C.RED}你没有可移动的棋子！{C.RESET}")
        return None

    # 按起始位置分组
    from_actions = {}
    for f, t in actions:
        from_actions.setdefault(f, []).append(t)

    while True:
        print(f"\n  {C.CYAN}选择要移动的棋子位置 (行,列 或 位置号):{C.RESET} ", end="")
        inp = input().strip().lower()

        if inp in ("q", "quit", "exit"):
            return None
        if inp in ("h", "help"):
            show_help()
            continue
        if inp in ("s", "show"):
            render_board(game, player)
            render_status(game, player)
            continue
        if inp in ("l", "legal"):
            show_legal_actions(game, player)
            continue

        from_pos = parse_position(inp)
        if from_pos < 0:
            print(f"  {C.RED}无效输入，请输入 行,列 (如 6,0) 或位置号 (如 30){C.RESET}")
            continue

        if from_pos not in from_actions:
            piece = game.board[from_pos]
            if piece is None:
                print(f"  {C.RED}该位置没有棋子{C.RESET}")
            elif piece.player != player:
                print(f"  {C.RED}这是对方的棋子{C.RESET}")
            else:
                print(f"  {C.RED}该棋子当前无法移动{C.RESET}")
            continue

        # 显示可移动目标
        targets = set(from_actions[from_pos])
        piece = game.board[from_pos]
        piece_name = SHORT_NAMES.get(piece.piece_type, "?")
        fr, fc = row_col(from_pos)

        print(f"\n  已选择: {C.GREEN}{piece_name}{C.RESET} 在 ({fr},{fc})")
        print(f"  可移动到 {len(targets)} 个位置:")
        for t in sorted(targets):
            tr, tc = row_col(t)
            target_piece = game.board[t]
            if target_piece and target_piece.player != player:
                if target_piece.revealed:
                    tname = SHORT_NAMES.get(target_piece.piece_type, "?")
                    print(f"    ({tr},{tc})={t:2d}  {C.RED}[攻击 {tname}]{C.RESET}")
                else:
                    print(f"    ({tr},{tc})={t:2d}  {C.RED}[攻击 ？]{C.RESET}")
            else:
                print(f"    ({tr},{tc})={t:2d}")

        render_board(game, player, highlight_from=from_pos, highlight_targets=targets)

        # 选择目标
        while True:
            print(f"\n  {C.CYAN}选择目标位置 (back=重选):{C.RESET} ", end="")
            inp2 = input().strip().lower()

            if inp2 in ("b", "back"):
                render_board(game, player)
                break
            if inp2 in ("q", "quit", "exit"):
                return None

            to_pos = parse_position(inp2)
            if to_pos < 0:
                print(f"  {C.RED}无效输入{C.RESET}")
                continue

            if to_pos not in targets:
                print(f"  {C.RED}不是合法目标位置{C.RESET}")
                continue

            # 确认
            return game.action_to_index(from_pos, to_pos)


# ──────────────────────────────────────────────
#  选择布阵
# ──────────────────────────────────────────────

def choose_template_interactive() -> int:
    """让玩家选择或预览布阵模板"""
    print(f"\n  {C.CYAN}选择布阵模板 (0-49):{C.RESET}")
    print(f"  直接输入编号，或输入 {C.CYAN}preview N{C.RESET} 预览模板 N")
    print(f"  输入 {C.CYAN}random{C.RESET} 随机选择\n")

    while True:
        inp = input(f"  模板编号: ").strip().lower()

        if inp in ("r", "random"):
            idx = random.randint(0, 49)
            print(f"  随机选择模板 #{idx}")
            return idx

        if inp.startswith("preview") or inp.startswith("p"):
            parts = inp.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    if 0 <= idx < 50:
                        preview_template(idx)
                    else:
                        print(f"  {C.RED}模板编号须在 0-49 之间{C.RESET}")
                except ValueError:
                    print(f"  {C.RED}无效编号{C.RESET}")
            continue

        try:
            idx = int(inp)
            if 0 <= idx < 50:
                return idx
            else:
                print(f"  {C.RED}模板编号须在 0-49 之间{C.RESET}")
        except ValueError:
            print(f"  {C.RED}请输入数字{C.RESET}")


def preview_template(idx: int):
    """预览一个布阵模板"""
    template = get_template(idx, player=0)
    print(f"\n  {C.CYAN}模板 #{idx} 预览:{C.RESET}")
    for r in range(6, 12):
        row_str = f"  R{r:2d}: "
        for c in range(COLS):
            p = pos(r, c)
            if p in ALL_CAMPS:
                row_str += " [营] "
            elif p in template:
                name = SHORT_NAMES.get(template[p], "?")
                row_str += f" {C.GREEN}{name}{C.RESET}   "
            else:
                row_str += "  ·   "
        print(row_str)
    print()


# ──────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="军棋 AI 人机对战",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python play.py checkpoints/junqi_final.pt
  python play.py checkpoints/junqi_final.pt --you-first
  python play.py checkpoints/junqi_final.pt --template 5
  python play.py --no-model
        """,
    )
    parser.add_argument("model", nargs="?", default=None,
                        help="AI 模型检查点路径")
    parser.add_argument("--no-model", action="store_true",
                        help="不加载模型，使用随机 AI")
    parser.add_argument("--you-first", action="store_true",
                        help="你先手 (默认)")
    parser.add_argument("--ai-first", action="store_true",
                        help="AI 先手")
    parser.add_argument("--template", type=int, default=None,
                        help="指定你的布阵模板 (0-49)")
    parser.add_argument("--ai-template", type=int, default=None,
                        help="指定 AI 的布阵模板 (0-49)")
    parser.add_argument("--choose", action="store_true",
                        help="交互式选择布阵模板")
    parser.add_argument("--device", type=str, default="cpu",
                        help="推理设备 (cpu/cuda)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="网络隐藏层维度 (需匹配训练时的配置)")
    parser.add_argument("--res-blocks", type=int, default=12,
                        help="残差块数量 (需匹配训练时的配置)")
    args = parser.parse_args()

    # 确定先手
    human_player = 0
    if args.ai_first:
        human_player = 1

    # 模型路径
    model_path = None if args.no_model else args.model

    # 选择布阵
    human_template = args.template
    if args.choose:
        human_template = choose_template_interactive()

    try:
        play_game(
            model_path=model_path,
            human_player=human_player,
            human_template=human_template,
            ai_template=args.ai_template,
            device=args.device,
            net_hidden_dim=args.hidden_dim,
            net_res_blocks=args.res_blocks,
        )
    except KeyboardInterrupt:
        print(f"\n\n  {C.YELLOW}游戏中止{C.RESET}\n")


if __name__ == "__main__":
    main()
