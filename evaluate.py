"""
评估脚本。
用于评估训练好的模型 vs 规则 AI 或自我对弈。
"""
import argparse
import os
import sys
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from junqi.env.junqi_env import JunqiEnv
from junqi.env.game import GameResult
from junqi.model.network import JunqiNetwork
from junqi.training.ppo import PPOAgent
from junqi.training.self_play import RuleBasedAI


def evaluate(
    model_path: str,
    num_games: int = 100,
    opponent: str = 'rule',
    device: str = 'cpu',
    render: bool = False,
):
    """
    评估模型。
    
    Args:
        model_path: 模型检查点路径
        num_games: 评估局数
        opponent: 'rule' or 'self'
        device: 设备
        render: 是否显示棋盘
    """
    print(f"\n{'='*50}")
    print(f"  军棋 AI 评估")
    print(f"  模型: {model_path}")
    print(f"  对手: {opponent}")
    print(f"  局数: {num_games}")
    print(f"{'='*50}\n")

    # 加载模型
    network = JunqiNetwork(hidden_dim=128, num_res_blocks=12)
    agent = PPOAgent(network=network, device=device)
    agent.load(model_path)
    network.eval()

    # 加载对手
    if opponent == 'rule':
        opp = RuleBasedAI()
        opp_type = 'rule'
    else:
        opp = JunqiNetwork(hidden_dim=128, num_res_blocks=12)
        opp_agent = PPOAgent(network=opp, device=device)
        opp_agent.load(model_path)
        opp.eval()
        opp_type = 'network'

    # 环境
    env = JunqiEnv(max_turns=500)

    # 统计
    results = defaultdict(int)
    game_lengths = []
    import random

    for game_idx in range(num_games):
        t0 = random.randint(0, 49)
        t1 = random.randint(0, 49)
        obs, info = env.reset(template_idx_p0=t0, template_idx_p1=t1)
        done = False

        while not done:
            player = env.get_current_player()
            action_mask = env.get_action_mask(player)

            if player == 0:
                current_obs = env._get_obs(player)
                action, _, _ = agent.select_action(
                    current_obs, action_mask, deterministic=True
                )
            else:
                if opp_type == 'rule':
                    action = opp.select_action(env.game, action_mask)
                else:
                    current_obs = env._get_obs(player)
                    state_t = torch.FloatTensor(current_obs).unsqueeze(0).to(device)
                    mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(device)
                    with torch.no_grad():
                        act, _, _, _ = opp.get_action_and_value(
                            state_t, mask_t, deterministic=True
                        )
                    action = act.item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if render and game_idx < 3:
                env.render(perspective=0)

        result = env.game.result
        results[result.name] += 1
        game_lengths.append(env.game.turn_count)

        if (game_idx + 1) % 10 == 0:
            print(f"  Game {game_idx+1}/{num_games} | "
                  f"W:{results.get('PLAYER0_WIN',0)} "
                  f"L:{results.get('PLAYER1_WIN',0)} "
                  f"D:{results.get('DRAW',0)}")

    # 最终统计
    print(f"\n{'='*50}")
    print(f"  最终结果 ({num_games} 局)")
    print(f"{'='*50}")
    wins = results.get('PLAYER0_WIN', 0)
    losses = results.get('PLAYER1_WIN', 0)
    draws = results.get('DRAW', 0)
    print(f"  胜: {wins} ({wins/num_games:.1%})")
    print(f"  负: {losses} ({losses/num_games:.1%})")
    print(f"  平: {draws} ({draws/num_games:.1%})")
    print(f"  平均局长: {np.mean(game_lengths):.1f} 步")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='军棋 AI 评估')
    parser.add_argument('model', type=str, help='模型检查点路径')
    parser.add_argument('--games', type=int, default=100, help='评估局数')
    parser.add_argument('--opponent', type=str, default='rule',
                        choices=['rule', 'self'], help='对手类型')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--render', action='store_true', help='显示棋盘')
    args = parser.parse_args()

    evaluate(args.model, args.games, args.opponent, args.device, args.render)


if __name__ == '__main__':
    main()
