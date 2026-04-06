"""Automated test of play.py - simulates a few turns."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from junqi.env.junqi_env import JunqiEnv
from junqi.env.game import GameResult
from junqi.env.board import pos, row_col, NUM_POSITIONS
from junqi.env.pieces import PIECE_NAMES
import numpy as np

# Simulate the play flow without interactive input
env = JunqiEnv(max_turns=500)
obs, info = env.reset(template_idx_p0=3, template_idx_p1=7, seed=99)
game = env.game

# Import render
from play import render_board, render_status, describe_event, SHORT_NAMES

print("=== Play Script Smoke Test ===\n")

# Show initial board from human (P0) perspective
render_board(game, human_player=0)
render_status(game, human_player=0)

# Show legal actions
actions = game.get_legal_actions(0)
print(f"\nP0 has {len(actions)} legal moves")
print("First 5 moves:")
for f, t in sorted(actions)[:5]:
    fr, fc = row_col(f)
    tr, tc = row_col(t)
    piece = game.board[f]
    name = SHORT_NAMES.get(piece.piece_type, "?")
    print(f"  {name} ({fr},{fc}) -> ({tr},{tc})")

# Highlight a selected piece
if actions:
    first_from = actions[0][0]
    targets = {t for f, t in actions if f == first_from}
    render_board(game, 0, highlight_from=first_from, highlight_targets=targets)

# Do one move via env
action_mask = env.get_action_mask(0)
legal = np.where(action_mask > 0)[0]
action = int(legal[0])
obs, reward, done, trunc, info = env.step(action)
describe_event(env.game.move_history[-1], human_player=0)

print("\n=== Smoke Test Passed ===")
